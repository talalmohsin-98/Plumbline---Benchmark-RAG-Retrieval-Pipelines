"""Lane 2: dense vector retrieval with bge-small-en-v1.5."""

from __future__ import annotations

from backend.lanes.base import Lane, LaneResult, StageTrace, rank_chunks, stopwatch
from backend.retrieval.corpus import Corpus


class DenseLane(Lane):
    """Embedding-only retrieval: cosine nearest neighbours over pgvector.

    The complement of lane 1. It finds passages that *mean* the same thing as
    the question without sharing its words, and misses exact identifiers that
    BM25 walks straight to. That the two fail on different questions is the
    premise of the hybrid lane.

    Runs entirely locally -- bge-small is 133 MB and embeds one query in under
    40 ms on CPU. No API, no key, no per-query cost.
    """

    id = "dense"
    label = "Dense only (bge-small)"

    def __init__(self, corpus: Corpus, *, use_query_prefix: bool = True) -> None:
        """`use_query_prefix` controls bge's asymmetric query instruction.

        Default True: bge-*-en-v1.5 is trained with the instruction on the
        query side and nothing on the passage side, so omitting it embeds the
        question in a slightly different space from the passages it is being
        matched against. The flag exists so the evaluation runner can measure
        that claim on this corpus instead of restating the model card, and the
        delta is published as a methodology note.
        """
        self.corpus = corpus
        self.use_query_prefix = use_query_prefix

    def warm(self) -> None:
        self.corpus.warm()

    def retrieve(self, query: str, k: int = 10) -> LaneResult:
        with stopwatch() as total:
            with stopwatch() as embed:
                vector = self.corpus.embed_query(query, prefix=self.use_query_prefix)
            with stopwatch() as search:
                scored = self.corpus.dense_search_vector(vector, k)
            chunks = rank_chunks(scored, self.corpus.texts, self.corpus.sources, k)
        prefix_note = "with bge query prefix" if self.use_query_prefix else "no query prefix"
        return LaneResult(
            chunks=chunks,
            latency_ms=total.ms,
            stages=[
                StageTrace(
                    name="embed",
                    latency_ms=embed.ms,
                    candidates_in=1,
                    candidates_out=1,
                    detail=f"bge-small-en-v1.5, {prefix_note}",
                ),
                StageTrace(
                    name="pgvector",
                    latency_ms=search.ms,
                    candidates_in=len(self.corpus.chunks),
                    candidates_out=len(chunks),
                    detail=f"top-{k} by cosine",
                ),
            ],
        )
