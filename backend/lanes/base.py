"""The Lane ABC plus the LaneResult and StageTrace dataclasses: the single lane interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk as a lane hands it back: identity, text, and where it ranked.

    Deliberately not `ingest.Chunk` or `dense_store.ScoredChunk`. Those carry
    fields that only make sense on one side -- character offsets before
    embedding, cosine similarity after a vector search -- and a lane that
    fuses two lists has neither. What every lane can honestly report is: which
    chunk, what text, what score my final stage gave it, and where it landed.

    `score` is comparable *within* one lane's result and meaningless across
    lanes: it is a cosine similarity in lane 2, an RRF score in lane 3, and a
    cross-encoder logit in lane 4. Nothing in `metrics.py` reads it -- the
    metrics are rank-based -- and nothing should start to.
    """

    chunk_id: str
    source_doc: str
    text: str
    score: float
    rank: int


@dataclass(frozen=True)
class StageTrace:
    """What one stage of a pipeline did, for the inspector and for debugging.

    The point of a benchmark that shows its working is that a reader can see
    *where* a lane spent its time and what it passed on. A lane that reports
    only a final list is a black box with a number attached.
    """

    name: str
    latency_ms: float
    candidates_in: int
    candidates_out: int
    detail: str = ""


@dataclass
class LaneResult:
    """One lane's answer to one query: the ranked chunks and how they were got."""

    chunks: list[RetrievedChunk]
    latency_ms: float
    stages: list[StageTrace] = field(default_factory=list)
    # Tokens billed by an LLM call made *during retrieval*. Zero for every lane
    # that does not make one, which is all of them except HyDE. That zero is a
    # finding, not a gap -- see EVALUATION_SPEC §2, "Cost per query".
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def chunk_ids(self) -> list[str]:
        """The ranked ids, best first. This is all `metrics.py` ever needs."""
        return [c.chunk_id for c in self.chunks]

    @property
    def tokens_used(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Lane(ABC):
    """One retrieval pipeline, comparable with every other by construction.

    Subclasses implement `retrieve` and nothing else. Adding a seventh lane
    must require only a new file here plus a registry entry in `__init__.py`;
    if it needs an edit to `graph.py` or `metrics.py`, this abstraction is
    wrong and the fix belongs here rather than in a special case there.
    """

    id: str
    label: str

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> LaneResult:
        """Return the top `k` chunks for `query`, best first.

        Implementations must set `latency_ms` from their own measurement of the
        whole method (see `stopwatch`), because only the lane knows what work
        it legitimately owns.
        """

    def warm(self) -> None:
        """Do any one-time work -- index build, model load -- before timing.

        Called once by the evaluation runner and excluded from the reported
        latencies. Without it the first query of each lane pays for loading a
        cross-encoder or building a BM25 index over 1,480 chunks, and with
        n=35 questions that one outlier *is* the p95. Default is a no-op; a
        lane that has nothing to warm inherits it.
        """
        return


class stopwatch:
    """Measure a block in milliseconds off the monotonic clock.

    `perf_counter`, not `time()`: a wall-clock adjustment mid-race would
    otherwise show up as a negative latency in the published numbers.
    """

    def __init__(self) -> None:
        self.ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> bool:
        self.ms = (time.perf_counter() - self._start) * 1000.0
        return False


def rank_chunks(
    scored: list[tuple[str, float]],
    texts: dict[str, str],
    sources: dict[str, str],
    k: int,
) -> list[RetrievedChunk]:
    """Turn (chunk_id, score) pairs into the top-`k` ranked `RetrievedChunk`s.

    Sorted by score descending with `chunk_id` as the tie-break. The tie-break
    is not cosmetic: BM25 gives whole groups of chunks a score of exactly 0.0
    when a query shares no terms with them, and without a deterministic second
    key the order of those ties depends on dict iteration order. That would
    make recall@10 vary between runs on the same data.
    """
    ordered = sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:k]
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            source_doc=sources.get(chunk_id, ""),
            text=texts.get(chunk_id, ""),
            score=score,
            rank=index + 1,
        )
        for index, (chunk_id, score) in enumerate(ordered)
    ]
