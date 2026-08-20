"""Cross-encoder scoring.

A cross-encoder reads (query, passage) as one sequence and scores the pair
directly, rather than embedding each side independently and comparing vectors.
That is what makes it more accurate than the bi-encoder that produced the
candidates -- and also why it cannot be used for first-stage retrieval: it
scores one pair per forward pass, so ranking 1,480 chunks would mean 1,480
passes per query.

The whole lane design follows from that: retrieve broadly and cheaply, then
rerank a shortlist. `rerank_depth` (20) is the shortlist.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

# bge-small tops out at 512 tokens and so does MiniLM; a (query, chunk) pair
# where the chunk is already 512 tokens will be truncated from the right. That
# is the model's limit, not a choice, and it is stated here because a chunk
# whose answer sits in its final sentence can be truncated away from the
# reranker while remaining perfectly retrievable by the lanes that do not
# rerank. It belongs in Known Limitations.
MAX_LENGTH = 512


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str) -> CrossEncoder:
    """The process-wide cross-encoder, loaded once per checkpoint.

    maxsize=2 because Day 3 adds the fine-tuned checkpoint alongside the stock
    one and both must stay resident: evicting and reloading a model between
    lanes would put a multi-second load inside a timed retrieval and land in
    the published p95.

    Loaded on first use rather than at import so the test suite and anything
    that only touches the lexical lanes need neither the model files nor a
    network. `backend.main` warms it at import time so the serving path never
    pays the load.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=MAX_LENGTH)


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    model: CrossEncoder,
) -> list[tuple[str, float]]:
    """Score (chunk_id, text) candidates against `query`; best first.

    Returns every candidate rescored, not a truncated list -- the caller owns
    the cutoff, because lane 4 wants the top 10 of a reranked 20 while a
    future lane might want all 20 with their scores.

    Scores are raw model outputs (logits for this checkpoint), not
    probabilities. They are used only to sort within one lane, and squashing
    them through a sigmoid would change nothing about the order while
    inviting a reader to compare them against another lane's cosine.

    Ties break on chunk_id, for the same determinism reason as everywhere else
    in this package. Exact ties are rare from a cross-encoder but not
    impossible: the corpus contains byte-identical duplicated chunks, which
    score identically by construction.
    """
    if not candidates:
        return []
    pairs = [(query, text) for _, text in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    chunk_ids = [chunk_id for chunk_id, _ in candidates]
    return sorted(
        zip(chunk_ids, (float(s) for s in scores), strict=True),
        key=lambda pair: (-pair[1], pair[0]),
    )
