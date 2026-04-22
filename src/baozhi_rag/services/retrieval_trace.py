"""检索链路追踪构造服务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from baozhi_rag.services.evidence_sufficiency import EvidenceAssessment

if TYPE_CHECKING:
    from baozhi_rag.services.chunk_search import ChunkSearchHit
    from baozhi_rag.services.retrieval_pipeline import (
        RetrievalLaneExecution,
        RetrievalPipelineExecutionResult,
    )


@dataclass(frozen=True, slots=True)
class RetrievalTraceLane:
    """单条 lane 的追踪摘要。"""

    lane_id: str
    query_text: str
    lane_weight: float
    result_count: int
    top_chunk_ids: list[str] = field(default_factory=list)
    source_strategy: str = "original"
    source_query: str | None = None
    strategy_label: str | None = None
    expansion_id: str | None = None
    embedding_source_summary: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """检索链路追踪摘要。"""

    mode: str
    query_intent: str
    lane_count: int
    final_hit_count: int
    evidence_sufficient: bool
    evidence_reason: str
    deep_rerank_triggered: bool = False
    lanes: list[RetrievalTraceLane] = field(default_factory=list)


class RetrievalTraceService:
    """把检索计划与执行结果转换为 trace 摘要。"""

    _EMBEDDING_SUMMARY_LIMIT = 96

    def build(
        self,
        *,
        execution_result: RetrievalPipelineExecutionResult,
        final_hits: list[ChunkSearchHit],
        evidence_assessment: EvidenceAssessment,
    ) -> RetrievalTrace:
        """构造可下发到 API trace 的结构化摘要。"""
        return RetrievalTrace(
            mode=execution_result.plan.mode,
            query_intent=execution_result.plan.query_intent,
            lane_count=len(execution_result.lane_executions),
            final_hit_count=len(final_hits),
            evidence_sufficient=evidence_assessment.sufficient,
            evidence_reason=evidence_assessment.reason_code,
            lanes=[
                self._build_lane_trace(lane_execution)
                for lane_execution in execution_result.lane_executions
            ],
        )

    def _build_lane_trace(self, lane_execution: RetrievalLaneExecution) -> RetrievalTraceLane:
        """构造单条 lane 的 trace。"""
        lane = lane_execution.lane
        lane_query_text = lane.lexical_query_text or lane.query_text
        embedding_source_text = lane.embedding_source_text or lane_query_text
        return RetrievalTraceLane(
            lane_id=lane.lane_id,
            query_text=lane_query_text,
            lane_weight=lane.lane_weight,
            result_count=len(lane_execution.hits),
            top_chunk_ids=[hit.chunk_id for hit in lane_execution.hits[:5]],
            source_strategy=lane.source_strategy,
            source_query=lane.source_query,
            strategy_label=lane.strategy_label,
            expansion_id=lane.expansion_id,
            embedding_source_summary=self._summarize_embedding_source(
                embedding_source_text=embedding_source_text,
                query_text=lane_query_text,
            ),
        )

    def _summarize_embedding_source(
        self,
        *,
        embedding_source_text: str,
        query_text: str,
    ) -> str | None:
        """在 embedding 文本与查询不一致时，返回可观测摘要。"""
        normalized_embedding_source = embedding_source_text.strip()
        if not normalized_embedding_source:
            return None

        normalized_query_text = query_text.strip()
        if normalized_embedding_source == normalized_query_text:
            return None

        compact_text = " ".join(normalized_embedding_source.split())
        if len(compact_text) <= self._EMBEDDING_SUMMARY_LIMIT:
            return compact_text
        return f"{compact_text[: self._EMBEDDING_SUMMARY_LIMIT - 3]}..."
