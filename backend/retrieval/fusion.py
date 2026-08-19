"""Reciprocal Rank Fusion of ranked result lists."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# Cormack et al. (2009), "Reciprocal Rank Fusion outperforms Condorcet and
# individual Rank Learning Methods". k=60 is the value from that paper and the
# default here; it is passed in rather than read from config so this module
# stays pure and testable. ARCHITECTURE §11 records the alternative: k controls
# how sharply rank 1 dominates -- small k (say 10) lets a single list's top hit
# overwhelm the other list, large k (say 100) flattens both toward a plain
# vote. Sweeping k ∈ {10, 30, 60, 100} is a measurement this project can make
# rather than an assumption it has to defend.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[str]],
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one, scoring each id `Σ 1/(k + rank)`.

    Ranks are 1-based: the first element of each list contributes `1/(k+1)`.

    Rank-based rather than score-based on purpose. BM25 returns unbounded
    term-frequency scores and pgvector returns a cosine similarity in [0, 1];
    adding those together would mean whichever scale happens to be larger wins
    every time, and normalising them requires knowing each list's range, which
    changes per query. Ranks are the one thing the two lists genuinely share.

    An id absent from a list simply contributes nothing from it -- there is no
    penalty term. An id found by both retrievers therefore outranks one found
    by either alone at the same position, which is the entire point of the
    hybrid lane.

    Returns (id, score) pairs sorted by score descending, id ascending. The id
    tie-break makes the output a deterministic function of the input: without
    it, two ids fused from symmetric positions would order by dict insertion
    and the split's recall@10 could differ between runs.
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")

    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        for position, chunk_id in enumerate(ranked, start=1):
            # A retriever that returns the same id twice must not be paid
            # twice for it. Defensive rather than observed: both current
            # retrievers key by chunk_id, but a future lane fusing three lists
            # from one source is exactly where this would silently inflate.
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)

    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
