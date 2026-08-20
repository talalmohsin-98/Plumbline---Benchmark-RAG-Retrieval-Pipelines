"""The lane registry.

Adding a lane means a new file in this package and one entry in `REGISTRY`.
Nothing in `graph.py`, `metrics.py` or `evaluate.py` names a lane, so nothing
in them has to change. That property is the point of the abstraction and is
tested in `test_lanes.py`.

Lane 6 (the fine-tuned reranker) is Day 3: it is the same `RerankedLane` class
with a different checkpoint, which is what makes its comparison with lane 4
attributable to the model rather than to the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable

from backend.config import get_settings
from backend.lanes.base import Lane, LaneResult, RetrievedChunk, StageTrace
from backend.lanes.bm25 import BM25Lane
from backend.lanes.dense import DenseLane
from backend.lanes.hybrid import HybridLane
from backend.lanes.hyde import HydeLane
from backend.lanes.reranked import RerankedLane
from backend.retrieval.corpus import Corpus

LaneFactory = Callable[[Corpus], Lane]

# Ordered as the leaderboard reads: cheapest and simplest first, so the table
# tells the story of what each addition buys.
REGISTRY: dict[str, LaneFactory] = {
    "bm25": BM25Lane,
    "dense": DenseLane,
    "hybrid_rrf": HybridLane,
    "hybrid_rerank": RerankedLane,
    "hyde": HydeLane,
}

# Lanes that call an external API during retrieval. Kept here rather than as a
# flag on the class so a run can be restricted to the offline lanes with
# `--skip-remote` -- the daily Groq budget is finite and the other four lanes
# must stay runnable when it is gone.
REMOTE_LANES: frozenset[str] = frozenset({"hyde"})


def lane_six_factory() -> LaneFactory:
    """Lane 6: the same reranking lane pointed at the fine-tuned checkpoint.

    Not in `REGISTRY` until the checkpoint exists on the Hub (Day 3); an entry
    here would make every evaluation run fail on a 404 for a model that has
    not been trained yet.
    """

    def build(corpus: Corpus) -> Lane:
        return RerankedLane(
            corpus,
            model_name=get_settings().reranker_tuned,
            lane_id="hybrid_rerank_tuned",
            label="Hybrid + RRF + fine-tuned reranker",
        )

    return build


def build_lanes(corpus: Corpus, lane_ids: list[str] | None = None) -> list[Lane]:
    """Instantiate lanes by id, in registry order.

    Registry order rather than the caller's order, so `--lanes dense,bm25` and
    `--lanes bm25,dense` produce the same results file.
    """
    wanted = set(lane_ids) if lane_ids is not None else set(REGISTRY)
    unknown = wanted - set(REGISTRY)
    if unknown:
        raise KeyError(f"unknown lane(s): {sorted(unknown)}. Known: {sorted(REGISTRY)}")
    return [factory(corpus) for lane_id, factory in REGISTRY.items() if lane_id in wanted]


__all__ = [
    "REGISTRY",
    "REMOTE_LANES",
    "BM25Lane",
    "Corpus",
    "DenseLane",
    "HybridLane",
    "HydeLane",
    "Lane",
    "LaneResult",
    "RerankedLane",
    "RetrievedChunk",
    "StageTrace",
    "build_lanes",
    "lane_six_factory",
]
