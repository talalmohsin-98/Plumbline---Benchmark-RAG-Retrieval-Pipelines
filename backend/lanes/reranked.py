"""Lanes 4 and 6: hybrid retrieval plus a cross-encoder rerank.

One class, two checkpoints. Lane 4 loads the stock MS MARCO MiniLM; lane 6
(Day 3) loads the fine-tuned one. Sharing the class is what makes their
comparison honest -- if the tuned lane had its own pipeline, any difference
between them could be the pipeline rather than the model.
"""

from __future__ import annotations

from backend.config import get_settings
from backend.lanes.base import Lane, LaneResult, StageTrace, rank_chunks, stopwatch
from backend.lanes.hybrid import HybridLane
from backend.retrieval.corpus import Corpus
from backend.retrieval.rerank import get_cross_encoder, rerank


class RerankedLane(Lane):
    """Fused candidates, rescored by a cross-encoder reading query and passage together.

    A reranker does not retrieve: it can never surface a chunk the fused list
    did not already contain, so recall@`rerank_depth` is a hard ceiling it
    cannot cross.

    But recall@10 is *not* that ceiling, and an earlier version of this
    docstring said it was. Because `rerank_depth` (20) is larger than the
    depth the metrics score at (10), the reranker reorders 20 candidates into
    a new top 10 -- so a gold chunk sitting at fused rank 11-20 can be
    promoted *into* the scored window. recall@10 can rise, and on the test
    split it does: 0.8571 to 0.9143, two questions pulled up from the 11-20
    band.

    That also inverts the prediction in EVALUATION_SPEC §2, which expects a
    reranker to "barely move recall@10 while noticeably lifting MRR".
    Measured here it is the other way round -- recall@10 +0.0571, MRR +0.0078
    -- and the reason is visible in the numbers: fusion had already put a gold
    chunk in the top 10 for 30 of 35 questions, so there was little ranking
    left to win and the gains had to come from the 11-20 band instead. The
    spec's expectation holds where first-stage recall is the bottleneck; here
    it is not. Recorded rather than reconciled.
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        model_name: str | None = None,
        lane_id: str = "hybrid_rerank",
        label: str = "Hybrid + RRF + reranker",
        use_query_prefix: bool = True,
    ) -> None:
        self.corpus = corpus
        self.id = lane_id
        self.label = label
        self.model_name = model_name or get_settings().reranker_base
        self.rerank_depth = get_settings().rerank_depth
        # Composition rather than inheritance: this lane *uses* the hybrid
        # retrieval, it is not a kind of hybrid lane. Subclassing would make
        # `retrieve` an override that silently has to remember not to call
        # super(), which is the kind of thing that breaks quietly later.
        self._hybrid = HybridLane(corpus, use_query_prefix=use_query_prefix)

    def warm(self) -> None:
        self._hybrid.warm()
        get_cross_encoder(self.model_name)

    def retrieve(self, query: str, k: int = 10) -> LaneResult:
        with stopwatch() as total:
            fused, stages = self._hybrid.fuse(query)
            # Only the top `rerank_depth` (20) is rescored. Cross-encoder cost
            # is linear in candidates and is the largest CPU item in the
            # system; reranking all 50 would roughly double the lane's latency
            # for whatever sits at fused ranks 21-50, which is below the depth
            # any reported metric reads.
            shortlist = fused[: self.rerank_depth]
            candidates = [(chunk_id, self.corpus.text_of(chunk_id)) for chunk_id, _ in shortlist]
            with stopwatch() as scoring:
                rescored = rerank(query, candidates, get_cross_encoder(self.model_name))
            chunks = rank_chunks(rescored, self.corpus.texts, self.corpus.sources, k)
        stages.append(
            StageTrace(
                name="rerank",
                latency_ms=scoring.ms,
                candidates_in=len(candidates),
                candidates_out=len(chunks),
                detail=f"cross-encoder {self.model_name}, depth {self.rerank_depth}",
            )
        )
        return LaneResult(chunks=chunks, latency_ms=total.ms, stages=stages)
