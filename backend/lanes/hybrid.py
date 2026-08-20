"""Lane 3: BM25 and dense retrieval fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from backend.config import get_settings
from backend.lanes.base import Lane, LaneResult, StageTrace, rank_chunks, stopwatch
from backend.retrieval.corpus import Corpus
from backend.retrieval.fusion import reciprocal_rank_fusion


class HybridLane(Lane):
    """Both retrievers, fused by rank.

    The load-bearing lane of the benchmark. Lanes 1 and 2 fail on *different*
    questions -- BM25 misses paraphrase, dense misses exact identifiers -- so
    fusing their rankings should beat both, and the build plan treats it as a
    gate: if lane 3 does not beat lanes 1 and 2 on recall@10, fusion is broken
    and nothing downstream of it means anything.

    Each retriever contributes its top `retrieve_depth` (50) rather than its
    top k. Fusing only the top 10 of each would throw away exactly the
    evidence RRF exists to use: a chunk ranked 1st by dense and 30th by BM25 is
    a strong hybrid result and is invisible if BM25's list stops at 10.
    """

    id = "hybrid_rrf"
    label = "Hybrid + RRF"

    def __init__(self, corpus: Corpus, *, use_query_prefix: bool = True) -> None:
        self.corpus = corpus
        self.use_query_prefix = use_query_prefix
        settings = get_settings()
        self.retrieve_depth = settings.retrieve_depth
        self.rrf_k = settings.rrf_k

    def warm(self) -> None:
        self.corpus.warm()

    def fuse(self, query: str) -> tuple[list[tuple[str, float]], list[StageTrace]]:
        """Retrieve from both arms and fuse. Shared with lanes 4 and 6.

        Returns the full fused list, not a truncated one: the reranking lanes
        take the top `rerank_depth` of it and would otherwise be limited by a
        cutoff chosen here.
        """
        with stopwatch() as lexical:
            bm25_hits = self.corpus.bm25_search(query, self.retrieve_depth)
        with stopwatch() as embed:
            vector = self.corpus.embed_query(query, prefix=self.use_query_prefix)
        with stopwatch() as vector_search:
            dense_hits = self.corpus.dense_search_vector(vector, self.retrieve_depth)
        with stopwatch() as fusion:
            fused = reciprocal_rank_fusion(
                [[chunk_id for chunk_id, _ in bm25_hits], [chunk_id for chunk_id, _ in dense_hits]],
                k=self.rrf_k,
            )
        stages = [
            StageTrace(
                name="bm25",
                latency_ms=lexical.ms,
                candidates_in=len(self.corpus.chunks),
                candidates_out=len(bm25_hits),
                detail=f"top-{self.retrieve_depth} lexical",
            ),
            StageTrace(
                name="embed",
                latency_ms=embed.ms,
                candidates_in=1,
                candidates_out=1,
                detail="bge-small-en-v1.5"
                + (", with query prefix" if self.use_query_prefix else ", no query prefix"),
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
        ]
        return fused, stages

    def retrieve(self, query: str, k: int = 10) -> LaneResult:
        with stopwatch() as total:
            fused, stages = self.fuse(query)
            chunks = rank_chunks(fused, self.corpus.texts, self.corpus.sources, k)
        return LaneResult(chunks=chunks, latency_ms=total.ms, stages=stages)
