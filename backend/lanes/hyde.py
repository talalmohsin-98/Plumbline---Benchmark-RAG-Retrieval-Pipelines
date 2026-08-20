"""Lane 5: HyDE — retrieve with a hypothetical answer instead of the question.

The idea (Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without
Relevance Labels"): a question and the passage that answers it are written in
different registers, so their embeddings sit further apart than they should.
Ask an LLM to *write* the answer first, embed that instead, and you are
comparing a passage against a passage.

It is the one lane here that makes an LLM call during retrieval, so it is the
one lane with a latency tail and a non-zero cost per query. Whether it buys
anything for that price is the question the benchmark exists to answer, and
the answer is allowed to be no.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from backend.config import get_settings
from backend.lanes.base import Lane, LaneResult, StageTrace, rank_chunks, stopwatch
from backend.lanes.hybrid import HybridLane
from backend.retrieval.corpus import Corpus
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.retrieval.rerank import get_cross_encoder, rerank

if TYPE_CHECKING:
    from groq import Groq

SYSTEM_PROMPT = (
    "You write short passages of technical documentation. "
    "Given a question, write the passage that would answer it, as if it were "
    "an excerpt from the manual. Do not answer conversationally, do not "
    "address the reader, and do not say whether you are certain. Write only "
    "the passage."
)

USER_PROMPT = "Question: {question}\n\nWrite the documentation passage that answers it."

# The passage itself only needs to be a paragraph -- it has to land in the
# right neighbourhood of the embedding space, and bge-small truncates at 512
# tokens anyway, so generating more spends latency on text the embedder
# discards.
#
# But this budget is not the passage length. gpt-oss-20b is a *reasoning*
# model: it emits reasoning tokens first and they count against max_tokens.
# At 160 the model spent the entire budget thinking, returned
# finish_reason="length" with an empty `content`, and the lane failed every
# query. Measured at reasoning_effort="low": 82 completion tokens for a
# typical question, so 400 is roughly 5x headroom without inviting an essay.
MAX_TOKENS = 400

# Reasoning models bill their thinking, and lane 5's whole purpose is to find
# out whether HyDE earns its price. Measured on one question:
#
#   effort="low"     82 completion tokens    368 ms
#   effort="medium" 267 completion tokens    765 ms
#
# "medium" alone blows the 700 ms HyDE budget in ARCHITECTURE §8 before the
# retrieval it precedes has started. Writing one documentation paragraph is
# not a task that needs deliberation, so "low" is the honest setting rather
# than a thumb on the scale.
REASONING_EFFORT = "low"

# temperature=0 is the most reproducible setting available, but it is not the
# same thing as deterministic, and this lane is the one place in the project
# where that distinction shows up in a published number.
#
# Two back-to-back runs over the identical test split, same corpus, same
# prompt: recall@5 and recall@10 came back byte-identical (0.8571 / 0.9429)
# while MRR moved 0.6286 -> 0.6310 and measured cost moved $5.043e-05 ->
# $5.025e-05. Same chunks retrieved, slightly different generated passage, so
# slightly different ranking and token count. Batched GPU inference at a
# provider is not bit-reproducible however the sampler is configured.
#
# So: temperature stays 0 because it minimises the drift, HyDE is deliberately
# NOT run with sampling and several drafts fused (which would likely score
# better and drift further), and the Known Limitations section says plainly
# that lane 5's third decimal place is not reproducible.
TEMPERATURE = 0.0


class HydeGenerationError(RuntimeError):
    """The hypothetical document could not be generated. The lane has no query."""


class HydeLane(Lane):
    """Hybrid + RRF + rerank, with the dense arm searching on a generated passage.

    Three decisions worth defending, none of them forced by the paper:

    1. **The generated passage is embedded with no bge query prefix.** The
       prefix marks a *query*; the whole point of HyDE is that this text is a
       pseudo-passage. Prefixing it would undo the symmetry the lane is buying.

    2. **The BM25 arm keeps the original question.** The generated passage is
       plausible but invented, and its identifiers are invented with it -- a
       hallucinated `--auto-reload` flag would be matched literally by BM25
       and pull the lexical arm somewhere the corpus never went. Dense
       retrieval degrades gracefully under a wrong word; BM25 does not.

    3. **The reranker scores against the original question.** The
       cross-encoder is judging whether a real passage answers the user's
       actual question. Handing it the hypothetical instead would score
       passages for resembling a guess.

    Together these mean HyDE changes exactly one thing versus lane 4 -- the
    vector the dense arm searches with -- which is what makes the lane 4 to
    lane 5 delta attributable.
    """

    id = "hyde"
    label = "Hybrid + RRF + rerank + HyDE"

    def __init__(
        self,
        corpus: Corpus,
        *,
        model: str | None = None,
        use_query_prefix: bool = True,
    ) -> None:
        self.corpus = corpus
        settings = get_settings()
        # The cheap fast model (`groq_model`), not the gold-set drafting model
        # (`goldset_model`): this call is on the latency path with a 700 ms
        # budget, and the drafting model's daily token budget is spoken for.
        self.model = model or settings.groq_model
        self.retrieve_depth = settings.retrieve_depth
        self.rerank_depth = settings.rerank_depth
        self.rrf_k = settings.rrf_k
        self.reranker = settings.reranker_base
        self.use_query_prefix = use_query_prefix
        self._hybrid = HybridLane(corpus, use_query_prefix=use_query_prefix)

    def warm(self) -> None:
        self._hybrid.warm()
        get_cross_encoder(self.reranker)

    # ----------------------------------------------------------------
    # Generation
    # ----------------------------------------------------------------

    def _client(self) -> Groq:
        from backend.goldset.generate import get_client

        return get_client()

    def generate_hypothetical(self, question: str) -> tuple[str, int, int]:
        """Return (passage, prompt_tokens, completion_tokens).

        Retries once on any API failure, because Groq's per-minute limit is
        transient and a single blip would otherwise score this lane a miss on
        a question it can answer. It does not retry twice: past that, the
        failure is the daily budget or the service, and both mean the lane is
        genuinely unavailable, which is a fact about the lane worth reporting
        rather than papering over.

        Never falls back to the plain question. A silent fallback would make
        lane 5 secretly lane 4 and publish the result as HyDE.
        """
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self._client().chat.completions.create(
                    model=self.model,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    reasoning_effort=REASONING_EFFORT,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT.format(question=question)},
                    ],
                )
                choice = response.choices[0]
                passage = (choice.message.content or "").strip()
                if not passage:
                    # Names the reasoning-token trap explicitly. An empty
                    # `content` with finish_reason="length" does not look like
                    # a truncation until you know the budget was spent before
                    # the model started writing.
                    detail = (
                        f"empty passage, finish_reason={choice.finish_reason!r}"
                        + (
                            f" -- max_tokens={MAX_TOKENS} was consumed by reasoning "
                            f"tokens before any content was emitted"
                            if choice.finish_reason == "length"
                            else ""
                        )
                    )
                    raise HydeGenerationError(detail)
                usage = response.usage
                return (
                    passage,
                    int(getattr(usage, "prompt_tokens", 0) or 0),
                    int(getattr(usage, "completion_tokens", 0) or 0),
                )
            except Exception as exc:  # re-raised below as one type
                last_error = exc
                if attempt == 0:
                    time.sleep(2.0)
        raise HydeGenerationError(
            f"could not generate a hypothetical document with {self.model}: {last_error}"
        ) from last_error

    # ----------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------

    def retrieve(self, query: str, k: int = 10) -> LaneResult:
        with stopwatch() as total:
            with stopwatch() as generation:
                passage, prompt_tokens, completion_tokens = self.generate_hypothetical(query)

            with stopwatch() as lexical:
                bm25_hits = self.corpus.bm25_search(query, self.retrieve_depth)
            with stopwatch() as embed:
                # prefix=False: this is a passage, not a query. See the class
                # docstring, decision 1.
                vector = self.corpus.embed_query(passage, prefix=False)
            with stopwatch() as vector_search:
                dense_hits = self.corpus.dense_search_vector(vector, self.retrieve_depth)
            with stopwatch() as fusion:
                fused = reciprocal_rank_fusion(
                    [
                        [chunk_id for chunk_id, _ in bm25_hits],
                        [chunk_id for chunk_id, _ in dense_hits],
                    ],
                    k=self.rrf_k,
                )
            shortlist = fused[: self.rerank_depth]
            candidates = [(chunk_id, self.corpus.text_of(chunk_id)) for chunk_id, _ in shortlist]
            with stopwatch() as scoring:
                # `query`, not `passage`. See the class docstring, decision 3.
                rescored = rerank(query, candidates, get_cross_encoder(self.reranker))
            chunks = rank_chunks(rescored, self.corpus.texts, self.corpus.sources, k)

        preview = " ".join(passage.split())[:80]
        return LaneResult(
            chunks=chunks,
            latency_ms=total.ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            stages=[
                StageTrace(
                    name="hyde",
                    latency_ms=generation.ms,
                    candidates_in=1,
                    candidates_out=1,
                    detail=f"{self.model}: {preview}...",
                ),
                StageTrace(
                    name="bm25",
                    latency_ms=lexical.ms,
                    candidates_in=len(self.corpus.chunks),
                    candidates_out=len(bm25_hits),
                    detail=f"top-{self.retrieve_depth} on the original question",
                ),
                StageTrace(
                    name="embed",
                    latency_ms=embed.ms,
                    candidates_in=1,
                    candidates_out=1,
                    detail="bge-small-en-v1.5 on the hypothetical passage, no query prefix",
                ),
                StageTrace(
                    name="pgvector",
                    latency_ms=vector_search.ms,
                    candidates_in=len(self.corpus.chunks),
                    candidates_out=len(dense_hits),
                    detail=f"top-{self.retrieve_depth} by cosine",
                ),
                StageTrace(
                    name="rrf",
                    latency_ms=fusion.ms,
                    candidates_in=len(bm25_hits) + len(dense_hits),
                    candidates_out=len(fused),
                    detail=f"reciprocal rank fusion, k={self.rrf_k}",
                ),
                StageTrace(
                    name="rerank",
                    latency_ms=scoring.ms,
                    candidates_in=len(candidates),
                    candidates_out=len(chunks),
                    detail=f"cross-encoder {self.reranker} on the original question",
                ),
            ],
        )
