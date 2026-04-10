from __future__ import annotations

from baozhi_rag.services.chunk_search import ChunkSearchHit
from baozhi_rag.services.evidence_sufficiency import EvidenceSufficiencyService
from baozhi_rag.services.retrieval_pipeline import (
    RetrievalLaneExecution,
    RetrievalPipelineExecutionResult,
)
from baozhi_rag.services.retrieval_plan import RetrievalLanePlan, RetrievalPlan
from baozhi_rag.services.retrieval_trace import RetrievalTraceService


def _build_hit(chunk_id: str, score: float | None) -> ChunkSearchHit:
    return ChunkSearchHit(
        chunk_id=chunk_id,
        file_id="file-1",
        chunk_type="text",
        segment_id="seg-1",
        source_filename="guide.md",
        storage_key="guide.md",
        chunk_index=0,
        char_count=10,
        content="content",
        merged_terms=[],
        score=score,
    )


def test_evidence_sufficiency_reports_no_evidence() -> None:
    service = EvidenceSufficiencyService()

    result = service.assess([])

    assert result.sufficient is False
    assert result.reason_code == "no_evidence"


def test_evidence_sufficiency_reports_low_score() -> None:
    service = EvidenceSufficiencyService(min_score=0.05)

    result = service.assess([_build_hit("chunk-1", 0.01)])

    assert result.sufficient is False
    assert result.reason_code == "low_score"


def test_retrieval_trace_service_builds_lane_summary() -> None:
    plan = RetrievalPlan(
        mode="chat",
        query_text="什么是向量检索",
        query_intent="definition",
        result_size=3,
        lanes=[
            RetrievalLanePlan(
                lane_id="lane-1",
                query_text="什么是向量检索",
                lane_weight=1.0,
                result_size=3,
                lexical_candidate_size=10,
                vector_candidate_size=10,
                query_intent="definition",
            )
        ],
    )
    lane_hits = [_build_hit("chunk-1", 0.9), _build_hit("chunk-2", 0.8)]
    execution_result = RetrievalPipelineExecutionResult(
        plan=plan,
        lane_executions=[
            RetrievalLaneExecution(
                lane=plan.lanes[0],
                hits=lane_hits,
            )
        ],
        hits=lane_hits,
    )
    assessment = EvidenceSufficiencyService().assess(lane_hits)

    trace = RetrievalTraceService().build(
        execution_result=execution_result,
        final_hits=lane_hits,
        evidence_assessment=assessment,
    )

    assert trace.mode == "chat"
    assert trace.query_intent == "definition"
    assert trace.lane_count == 1
    assert trace.evidence_sufficient is True
    assert trace.lanes[0].top_chunk_ids == ["chunk-1", "chunk-2"]


def test_evidence_sufficiency_reports_conflicting_evidence() -> None:
    service = EvidenceSufficiencyService()
    hits = [
        ChunkSearchHit(
            chunk_id="chunk-1",
            file_id="file-1",
            chunk_type="text",
            segment_id="seg-1",
            source_filename="guide.md",
            storage_key="guide.md",
            chunk_index=0,
            char_count=10,
            content="该功能支持启用。",
            merged_terms=[],
            score=0.9,
        ),
        ChunkSearchHit(
            chunk_id="chunk-2",
            file_id="file-1",
            chunk_type="text",
            segment_id="seg-2",
            source_filename="guide.md",
            storage_key="guide.md",
            chunk_index=1,
            char_count=10,
            content="该功能当前禁用，不支持启用。",
            merged_terms=[],
            score=0.88,
        ),
    ]

    result = service.assess(hits)

    assert result.sufficient is False
    assert result.reason_code == "conflicting_evidence"
