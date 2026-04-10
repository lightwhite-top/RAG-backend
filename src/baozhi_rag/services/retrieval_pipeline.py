"""多 lane 检索执行与融合流水线。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.retrieval_plan import RetrievalLanePlan, RetrievalPlan
from baozhi_rag.services.term_matching import MaximumMatchingTermMatcher

if TYPE_CHECKING:
    from baozhi_rag.services.chunk_search import (
        ChunkSearchHit,
        ChunkSearchStore,
    )


@dataclass(frozen=True, slots=True)
class RetrievalLaneExecution:
    """单条 lane 的执行结果。"""

    lane: RetrievalLanePlan
    hits: list[ChunkSearchHit]


@dataclass(frozen=True, slots=True)
class RetrievalPipelineExecutionResult:
    """完整检索流水线执行结果。"""

    plan: RetrievalPlan
    lane_executions: list[RetrievalLaneExecution]
    hits: list[ChunkSearchHit]


class RetrievalPipelineService:
    """执行多 lane 检索计划并做全局融合。"""

    def __init__(
        self,
        *,
        term_matcher: MaximumMatchingTermMatcher,
        store: ChunkSearchStore,
        chunk_embedding_service: ChunkEmbeddingService,
    ) -> None:
        self._term_matcher = term_matcher
        self._store = store
        self._chunk_embedding_service = chunk_embedding_service

    def execute(
        self,
        *,
        retrieval_plan: RetrievalPlan,
        viewer_user_id: str,
    ) -> RetrievalPipelineExecutionResult:
        """执行多 lane 检索并做全局 RRF 融合。"""
        query_embedding_cache: dict[str, list[float]] = {}
        lane_executions: list[RetrievalLaneExecution] = []

        for lane in retrieval_plan.lanes:
            from baozhi_rag.services.chunk_search import ChunkSearchRequest

            terms = self._term_matcher.extract_terms(lane.query_text)
            query_embedding: list[float] = []
            if lane.vector_candidate_size > 0 and (
                lane.vector_rrf_weight is None or lane.vector_rrf_weight > 0.0
            ):
                cached_embedding = query_embedding_cache.get(lane.query_text)
                if cached_embedding is None:
                    cached_embedding = self._chunk_embedding_service.embed_query(lane.query_text)
                    query_embedding_cache[lane.query_text] = cached_embedding
                query_embedding = cached_embedding

            request = ChunkSearchRequest(
                query_text=lane.query_text,
                size=lane.result_size,
                merged_terms=terms.merged_terms,
                query_embedding=query_embedding,
                query_intent=lane.query_intent,
                lexical_candidate_size=lane.lexical_candidate_size,
                vector_candidate_size=lane.vector_candidate_size,
                lexical_rrf_weight=lane.lexical_rrf_weight,
                vector_rrf_weight=lane.vector_rrf_weight,
                viewer_user_id=viewer_user_id,
            )
            lane_executions.append(
                RetrievalLaneExecution(
                    lane=lane,
                    hits=self._store.search(request),
                )
            )

        hits = self._fuse_lane_results(
            lane_executions=lane_executions,
            result_size=retrieval_plan.result_size,
        )
        return RetrievalPipelineExecutionResult(
            plan=retrieval_plan,
            lane_executions=lane_executions,
            hits=hits,
        )

    def _fuse_lane_results(
        self,
        *,
        lane_executions: list[RetrievalLaneExecution],
        result_size: int,
    ) -> list[ChunkSearchHit]:
        """对多 lane 结果做全局 RRF 融合。"""
        fused_scores: dict[str, float] = {}
        hit_map: dict[str, ChunkSearchHit] = {}

        for lane_execution in lane_executions:
            for rank, hit in enumerate(lane_execution.hits, start=1):
                fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + (
                    lane_execution.lane.lane_weight / (60 + rank)
                )
                if hit.chunk_id not in hit_map:
                    hit_map[hit.chunk_id] = hit

        ordered_chunk_ids = [
            chunk_id
            for chunk_id, _ in sorted(
                fused_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        return [
            replace(hit_map[chunk_id], score=round(fused_scores[chunk_id], 6))
            for chunk_id in ordered_chunk_ids[:result_size]
            if chunk_id in hit_map
        ]
