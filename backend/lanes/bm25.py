"""Lane 1: BM25-only lexical retrieval."""

from __future__ import annotations

from backend.lanes.base import Lane, LaneResult, StageTrace, rank_chunks, stopwatch
from backend.retrieval.corpus import Corpus


class BM25Lane(Lane):
    """Term-frequency retrieval and nothing else.

    The floor of the benchmark, and not a strawman: BM25 is what a great many
    production systems still run, it needs no model and no GPU, and on
    documentation full of exact identifiers -- `python-multipart`,
    `HTTPException`, `--reload` -- matching the literal token is often exactly
    right. Any lane that cannot beat it has not earned its latency.
    """

    id = "bm25"
    label = "BM25 only"

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def warm(self) -> None:
        # No embedder: this lane must be runnable with no model files present.
        self.corpus.warm(embedder=False)

    def retrieve(self, query: str, k: int = 10) -> LaneResult:
        with stopwatch() as total:
            with stopwatch() as search:
                scored = self.corpus.bm25_search(query, k)
            chunks = rank_chunks(scored, self.corpus.texts, self.corpus.sources, k)
        return LaneResult(
            chunks=chunks,
            latency_ms=total.ms,
            stages=[
                StageTrace(
                    name="bm25",
                    latency_ms=search.ms,
                    candidates_in=len(self.corpus.chunks),
                    candidates_out=len(chunks),
                    detail=f"top-{k} by BM25 (k1=1.5, b=0.75)",
                )
            ],
        )
