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

    What this is expected to do, from EVALUATION_SPEC §2: barely move
    recall@10 and noticeably lift MRR. It cannot improve recall beyond what
    the fused list already contains -- it reorders a shortlist, it does not
    retrieve -- so recall@10 can only fall (if a gold chunk in the fused top
    10 is pushed out of the reranked top 10) or stay flat. MRR is where a
    reranker earns its 600 ms.
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
