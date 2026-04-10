from __future__ import annotations

from baozhi_rag.services.chat import ChatCitation
from baozhi_rag.services.context_packing import ContextPackingService
from baozhi_rag.services.evidence_sufficiency import EvidenceAssessment


def _build_citation() -> ChatCitation:
    return ChatCitation(
        citation_id="cit-1",
        chunk_id="chunk-1",
        file_id="file-1",
        chunk_type="text",
        segment_id="seg-1",
        source_filename="guide.md",
        storage_key="guide.md",
        chunk_index=0,
        char_count=100,
        content="这里是证据正文。",
        merged_terms=["证据"],
        score=0.9,
    )


def test_context_packing_builds_context_prompt() -> None:
    service = ContextPackingService(max_context_chars=50)

    prompt = service.build_context_prompt(
        retrieval_query="什么是向量检索",
        citations=[_build_citation()],
    )

    assert "用户当前问题：什么是向量检索" in prompt
    assert "[1] 文件：guide.md" in prompt


def test_context_packing_builds_evidence_guidance_prompt() -> None:
    service = ContextPackingService()
    prompt = service.build_evidence_guidance_prompt(
        evidence_assessment=EvidenceAssessment(
            sufficient=False,
            reason_code="low_score",
            citation_count=1,
            top_score=0.01,
        )
    )

    assert "当前证据充分性判断：证据不足。" in prompt
    assert "不足原因：low_score" in prompt
