from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from baozhi_rag.domain.knowledge_file import FileStorageProvider, FileVisibilityScope, KnowledgeFile
from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest, ChunkSearchService
from baozhi_rag.services.retrieval_expansion import (
    ExpansionAwareRetrievalPlanner,
    ExpansionCandidate,
    ExpansionPlannerRequest,
    RetrievalExpansionOrchestrator,
)
from baozhi_rag.services.retrieval_plan import RetrievalPlanner
from baozhi_rag.services.term_matching import TermMatchResult


class _StubTermMatcher:
    """返回固定术语匹配结果。"""

    def extract_terms(self, text: str) -> TermMatchResult:
        return TermMatchResult(merged_terms=["term-a"])


class _StubEmbeddingService:
    """返回固定查询向量。"""

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


@dataclass
class _StubStore:
    """记录最后一次检索请求。"""

    requests: list[ChunkSearchRequest] | None = None
    hits: list[ChunkSearchHit] | None = None

    def __post_init__(self) -> None:
        if self.requests is None:
            self.requests = []
        if self.hits is None:
            self.hits = []

    def ensure_index(self) -> None:  # pragma: no cover - 协议占位
        return None

    def index_chunks(self, chunks: list[object]) -> int:  # pragma: no cover - 协议占位
        return len(chunks)

    def delete_chunks_by_file_id(self, file_id: str) -> None:  # pragma: no cover - 协议占位
        return None

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        self.requests.append(request)
        return list(self.hits or [])


@dataclass
class _StubKnowledgeFileRepository:
    files: list[KnowledgeFile]

    def get_files_by_ids(self, file_ids: list[str]) -> list[KnowledgeFile]:
        file_id_set = set(file_ids)
        return [item for item in self.files if item.id in file_id_set]


def _build_service(store: _StubStore) -> ChunkSearchService:
    return ChunkSearchService(
        term_matcher=_StubTermMatcher(),
        store=store,
        chunk_embedding_service=_StubEmbeddingService(),  # type: ignore[arg-type]
        retrieval_planner=RetrievalPlanner(
            lexical_candidate_size=40,
            vector_candidate_size=40,
            lexical_rrf_weight=1.0,
            vector_rrf_weight=1.1,
        ),
    )


def test_chunk_search_detects_definition_intent() -> None:
    store = _StubStore()
    service = _build_service(store)

    service.search("什么是向量检索", 5)

    assert store.requests
    assert all(request.query_intent == "definition" for request in store.requests)
    assert len(store.requests) == 3


def test_chunk_search_detects_document_location_intent() -> None:
    store = _StubStore()
    service = _build_service(store)

    service.search("这个内容在哪一章", 5)

    assert store.requests
    assert all(request.query_intent == "document_location" for request in store.requests)
    assert len(store.requests) == 4


def test_chunk_search_detects_structured_intent() -> None:
    store = _StubStore()
    service = _build_service(store)

    service.search("表格里第二列是什么意思", 5)

    assert store.requests
    assert all(request.query_intent == "structured" for request in store.requests)
    assert len(store.requests) == 4


def test_chunk_search_detects_image_scan_queries_as_structured() -> None:
    store = _StubStore()
    service = _build_service(store)

    service.search("影像或扫描内容里，影像补传截止时点是多少", 5)

    assert store.requests
    assert all(request.query_intent == "structured" for request in store.requests)


def test_chunk_search_detects_precise_numeric_document_location_intent() -> None:
    store = _StubStore()
    service = _build_service(store)

    service.search("样例 1 的处理时限为 5 小时", 5)

    assert store.requests
    assert all(request.query_intent == "general" for request in store.requests)
    assert len(store.requests) == 3


def test_chunk_search_suppresses_weak_hits_for_negative_evidence_query() -> None:
    store = _StubStore(
        hits=[
            ChunkSearchHit(
                chunk_id="file-1-chunk-0",
                file_id="file-1",
                chunk_type="text",
                segment_id="seg-1",
                source_filename="06_培训与上岗认证手册_V2.pdf",
                storage_key="knowledge-files/admin/file-1/06_培训与上岗认证手册_V2.pdf",
                page_number=None,
                source_anchor="chunk:0",
                chunk_index=0,
                char_count=32,
                content="新人上岗前理论培训时长：16小时；考试通过线：85分。",
                merged_terms=[],
                score=0.31,
            )
        ]
    )
    service = _build_service(store)

    result = service.search_with_trace("培训手册是否规定新人考试满分必须达到99分？", 5)

    assert result.hits == []
    assert result.evidence_assessment.sufficient is False
    assert result.evidence_assessment.reason_code == "no_evidence"


def test_chunk_search_keeps_hits_for_guidance_query_even_if_terms_are_sparse() -> None:
    store = _StubStore(
        hits=[
            ChunkSearchHit(
                chunk_id="file-1-chunk-0",
                file_id="file-1",
                chunk_type="text",
                segment_id="seg-1",
                source_filename="01_财务报销制度_V2.docx",
                storage_key="knowledge-files/admin/file-1/01_财务报销制度_V2.docx",
                page_number=None,
                source_anchor="chunk:0",
                chunk_index=0,
                char_count=42,
                content="处理问答时需要将角色、版本、申请编号、时间边界和材料完整度一起核对。",
                merged_terms=[],
                score=0.33,
            )
        ]
    )
    service = _build_service(store)

    result = service.search_with_trace(
        "如果只基于 01_财务报销制度_V2.docx 回答，遇到证据不足时应该怎么说？",
        5,
    )

    assert result.hits


def test_chunk_search_adds_filename_anchor_lanes_for_explicit_file_queries() -> None:
    store = _StubStore()
    service = _build_service(store)

    service.search(
        "如果只基于 28_财务_发票真伪核验说明_scanned.pdf 回答，遇到证据不足时应该怎么说？", 5
    )

    assert store.requests is not None
    request_queries = [request.query_text for request in store.requests]
    assert (
        "如果只基于 28_财务_发票真伪核验说明_scanned.pdf 回答，遇到证据不足时应该怎么说？"
        in request_queries
    )
    assert "28_财务_发票真伪核验说明_scanned.pdf" in request_queries
    assert any("财务 发票真伪核验说明 scanned" in query for query in request_queries)


def test_chunk_search_filters_owner_only_hits_for_other_user() -> None:
    store = _StubStore(
        hits=[
            ChunkSearchHit(
                chunk_id="file-private-chunk-0",
                file_id="file-private",
                chunk_type="text",
                segment_id="seg-1",
                source_filename="private.docx",
                storage_key="knowledge-files/user-a/file-private/private.docx",
                page_number=None,
                source_anchor="chunk:0",
                chunk_index=0,
                char_count=12,
                content="private content",
                merged_terms=[],
                score=0.91,
            ),
            ChunkSearchHit(
                chunk_id="file-global-chunk-0",
                file_id="file-global",
                chunk_type="text",
                segment_id="seg-2",
                source_filename="global.docx",
                storage_key="knowledge-files/admin/file-global/global.docx",
                page_number=None,
                source_anchor="chunk:0",
                chunk_index=0,
                char_count=11,
                content="global content",
                merged_terms=[],
                score=0.72,
            ),
        ]
    )
    now = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    knowledge_file_repository = _StubKnowledgeFileRepository(
        files=[
            KnowledgeFile(
                id="file-private",
                uploader_user_id="user-a",
                original_filename="private.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size=128,
                storage_provider=FileStorageProvider.ALIYUN_OSS,
                storage_key="knowledge-files/user-a/file-private/private.docx",
                visibility_scope=FileVisibilityScope.OWNER_ONLY,
                chunk_count=1,
                uploaded_at=now,
                updated_at=now,
                raw_sha256="raw-private",
                text_sha256="text-private",
                content_sha256="content-private",
            ),
            KnowledgeFile(
                id="file-global",
                uploader_user_id="admin-1",
                original_filename="global.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size=128,
                storage_provider=FileStorageProvider.ALIYUN_OSS,
                storage_key="knowledge-files/admin/file-global/global.docx",
                visibility_scope=FileVisibilityScope.GLOBAL,
                chunk_count=1,
                uploaded_at=now,
                updated_at=now,
                raw_sha256="raw-global",
                text_sha256="text-global",
                content_sha256="content-global",
            ),
        ]
    )
    service = ChunkSearchService(
        term_matcher=_StubTermMatcher(),
        store=store,
        chunk_embedding_service=_StubEmbeddingService(),  # type: ignore[arg-type]
        knowledge_file_repository=knowledge_file_repository,  # type: ignore[arg-type]
        retrieval_planner=RetrievalPlanner(
            lexical_candidate_size=40,
            vector_candidate_size=40,
            lexical_rrf_weight=1.0,
            vector_rrf_weight=1.1,
        ),
    )

    hits = service.search("private content", 5, viewer_user_id="user-b")

    assert [hit.file_id for hit in hits] == ["file-global"]


def test_chunk_search_trace_filters_owner_only_lane_hits_for_other_user() -> None:
    store = _StubStore(
        hits=[
            ChunkSearchHit(
                chunk_id="file-private-chunk-0",
                file_id="file-private",
                chunk_type="text",
                segment_id="seg-1",
                source_filename="private.docx",
                storage_key="knowledge-files/user-a/file-private/private.docx",
                page_number=None,
                source_anchor="chunk:0",
                chunk_index=0,
                char_count=12,
                content="private content",
                merged_terms=[],
                score=0.91,
            ),
            ChunkSearchHit(
                chunk_id="file-global-chunk-0",
                file_id="file-global",
                chunk_type="text",
                segment_id="seg-2",
                source_filename="global.docx",
                storage_key="knowledge-files/admin/file-global/global.docx",
                page_number=None,
                source_anchor="chunk:0",
                chunk_index=0,
                char_count=11,
                content="global content",
                merged_terms=[],
                score=0.72,
            ),
        ]
    )
    now = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    knowledge_file_repository = _StubKnowledgeFileRepository(
        files=[
            KnowledgeFile(
                id="file-private",
                uploader_user_id="user-a",
                original_filename="private.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size=128,
                storage_provider=FileStorageProvider.ALIYUN_OSS,
                storage_key="knowledge-files/user-a/file-private/private.docx",
                visibility_scope=FileVisibilityScope.OWNER_ONLY,
                chunk_count=1,
                uploaded_at=now,
                updated_at=now,
                raw_sha256="raw-private",
                text_sha256="text-private",
                content_sha256="content-private",
            ),
            KnowledgeFile(
                id="file-global",
                uploader_user_id="admin-1",
                original_filename="global.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size=128,
                storage_provider=FileStorageProvider.ALIYUN_OSS,
                storage_key="knowledge-files/admin/file-global/global.docx",
                visibility_scope=FileVisibilityScope.GLOBAL,
                chunk_count=1,
                uploaded_at=now,
                updated_at=now,
                raw_sha256="raw-global",
                text_sha256="text-global",
                content_sha256="content-global",
            ),
        ]
    )
    service = ChunkSearchService(
        term_matcher=_StubTermMatcher(),
        store=store,
        chunk_embedding_service=_StubEmbeddingService(),  # type: ignore[arg-type]
        knowledge_file_repository=knowledge_file_repository,  # type: ignore[arg-type]
        retrieval_planner=RetrievalPlanner(
            lexical_candidate_size=40,
            vector_candidate_size=40,
            lexical_rrf_weight=1.0,
            vector_rrf_weight=1.1,
        ),
    )

    result = service.search_with_trace("private content", 5, viewer_user_id="user-b")

    assert [hit.file_id for hit in result.hits] == ["file-global"]
    assert result.retrieval_trace.lanes
    assert all(
        lane.top_chunk_ids == ["file-global-chunk-0"] for lane in result.retrieval_trace.lanes
    )


def test_chunk_search_prefers_explicit_target_filename_over_same_chain_newer_version() -> None:
    store = _StubStore(
        hits=[
            ChunkSearchHit(
                chunk_id="file-v2-chunk-0",
                file_id="file-v2",
                chunk_type="text",
                segment_id="seg-v2",
                source_filename="01_财务报销制度_V2.docx",
                storage_key="knowledge-files/admin/file-v2/01_财务报销制度_V2.docx",
                page_number=None,
                source_anchor="chunk:0",
                chunk_index=0,
                char_count=20,
                content="单笔差旅自动审批上限 2000 元。",
                merged_terms=[],
                score=0.92,
            ),
            ChunkSearchHit(
                chunk_id="file-v1-chunk-0",
                file_id="file-v1",
                chunk_type="text",
                segment_id="seg-v1",
                source_filename="01_财务报销制度_V1.docx",
                storage_key="knowledge-files/admin/file-v1/01_财务报销制度_V1.docx",
                page_number=None,
                source_anchor="chunk:0",
                chunk_index=0,
                char_count=20,
                content="单笔差旅自动审批上限 1500 元。",
                merged_terms=[],
                score=0.71,
            ),
        ]
    )
    now = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    knowledge_file_repository = _StubKnowledgeFileRepository(
        files=[
            KnowledgeFile(
                id="file-v1",
                uploader_user_id="admin-1",
                original_filename="01_财务报销制度_V1.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size=128,
                storage_provider=FileStorageProvider.ALIYUN_OSS,
                storage_key="knowledge-files/admin/file-v1/01_财务报销制度_V1.docx",
                visibility_scope=FileVisibilityScope.GLOBAL,
                chunk_count=1,
                uploaded_at=now,
                updated_at=now,
                raw_sha256="raw-v1",
                text_sha256="text-v1",
                content_sha256="content-v1",
            ),
            KnowledgeFile(
                id="file-v2",
                uploader_user_id="admin-1",
                original_filename="01_财务报销制度_V2.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size=128,
                storage_provider=FileStorageProvider.ALIYUN_OSS,
                storage_key="knowledge-files/admin/file-v2/01_财务报销制度_V2.docx",
                visibility_scope=FileVisibilityScope.GLOBAL,
                chunk_count=1,
                uploaded_at=now,
                updated_at=now,
                raw_sha256="raw-v2",
                text_sha256="text-v2",
                content_sha256="content-v2",
            ),
        ]
    )
    service = ChunkSearchService(
        term_matcher=_StubTermMatcher(),
        store=store,
        chunk_embedding_service=_StubEmbeddingService(),  # type: ignore[arg-type]
        knowledge_file_repository=knowledge_file_repository,  # type: ignore[arg-type]
        retrieval_planner=RetrievalPlanner(
            lexical_candidate_size=40,
            vector_candidate_size=40,
            lexical_rrf_weight=1.0,
            vector_rrf_weight=1.1,
        ),
    )

    hits = service.search(
        "01_财务报销制度_V1.docx 中的单笔差旅自动审批上限是什么？",
        5,
        viewer_user_id="user-a",
    )

    assert [hit.file_id for hit in hits] == ["file-v1"]


def test_chunk_search_marks_explicit_private_target_as_inaccessible() -> None:
    store = _StubStore(
        hits=[
            ChunkSearchHit(
                chunk_id="file-private-chunk-0",
                file_id="file-private",
                chunk_type="text",
                segment_id="seg-private",
                source_filename="13_FAQ_员工出差报销常见问答.docx",
                storage_key="knowledge-files/user-a/file-private/13_FAQ_员工出差报销常见问答.docx",
                page_number=None,
                source_anchor="chunk:0",
                chunk_index=0,
                char_count=20,
                content="发票抬头错误补正时限：2个工作日。",
                merged_terms=[],
                score=0.91,
            ),
            ChunkSearchHit(
                chunk_id="file-global-chunk-0",
                file_id="file-global",
                chunk_type="text",
                segment_id="seg-global",
                source_filename="01_财务报销制度_V2.docx",
                storage_key="knowledge-files/admin/file-global/01_财务报销制度_V2.docx",
                page_number=None,
                source_anchor="chunk:0",
                chunk_index=0,
                char_count=20,
                content="单笔差旅自动审批上限 2000 元。",
                merged_terms=[],
                score=0.62,
            ),
        ]
    )
    now = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    knowledge_file_repository = _StubKnowledgeFileRepository(
        files=[
            KnowledgeFile(
                id="file-private",
                uploader_user_id="user-a",
                original_filename="13_FAQ_员工出差报销常见问答.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size=128,
                storage_provider=FileStorageProvider.ALIYUN_OSS,
                storage_key="knowledge-files/user-a/file-private/13_FAQ_员工出差报销常见问答.docx",
                visibility_scope=FileVisibilityScope.OWNER_ONLY,
                chunk_count=1,
                uploaded_at=now,
                updated_at=now,
                raw_sha256="raw-private",
                text_sha256="text-private",
                content_sha256="content-private",
            ),
            KnowledgeFile(
                id="file-global",
                uploader_user_id="admin-1",
                original_filename="01_财务报销制度_V2.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size=128,
                storage_provider=FileStorageProvider.ALIYUN_OSS,
                storage_key="knowledge-files/admin/file-global/01_财务报销制度_V2.docx",
                visibility_scope=FileVisibilityScope.GLOBAL,
                chunk_count=1,
                uploaded_at=now,
                updated_at=now,
                raw_sha256="raw-global",
                text_sha256="text-global",
                content_sha256="content-global",
            ),
        ]
    )
    service = ChunkSearchService(
        term_matcher=_StubTermMatcher(),
        store=store,
        chunk_embedding_service=_StubEmbeddingService(),  # type: ignore[arg-type]
        knowledge_file_repository=knowledge_file_repository,  # type: ignore[arg-type]
        retrieval_planner=RetrievalPlanner(
            lexical_candidate_size=40,
            vector_candidate_size=40,
            lexical_rrf_weight=1.0,
            vector_rrf_weight=1.1,
        ),
    )

    result = service.search_with_trace(
        "请直接告诉我 13_FAQ_员工出差报销常见问答.docx 里发票抬头错误补正时限的值。",
        5,
        viewer_user_id="user-b",
    )

    assert result.hits == []
    assert result.evidence_assessment.reason_code == "target_file_inaccessible"
    assert result.retrieval_trace.final_hit_count == 0


@dataclass
class _StubExpansionStrategy:
    """返回固定扩展候选，模拟统一扩展检索框架中的策略。"""

    calls: list[ExpansionPlannerRequest] | None = None
    strategy_name: str = "mqe"

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def expand(self, request: ExpansionPlannerRequest) -> list[ExpansionCandidate]:
        self.calls.append(request)
        return [
            ExpansionCandidate(
                strategy_name="mqe",
                query_text=f"{request.query_text} 扩展问法",
                embedding_source_text=f"{request.query_text} 扩展问法",
                lane_weight=0.85,
            ),
            ExpansionCandidate(
                strategy_name="hyde",
                query_text=request.query_text,
                embedding_source_text=f"{request.query_text} 的假设答案",
                lane_weight=0.75,
                vector_only=True,
            ),
        ]


def test_chunk_search_uses_expansion_aware_planner_for_search_and_chat() -> None:
    store = _StubStore()
    strategy = _StubExpansionStrategy()
    service = ChunkSearchService(
        term_matcher=_StubTermMatcher(),
        store=store,
        chunk_embedding_service=_StubEmbeddingService(),  # type: ignore[arg-type]
        retrieval_planner=ExpansionAwareRetrievalPlanner(
            lexical_candidate_size=40,
            vector_candidate_size=40,
            lexical_rrf_weight=1.0,
            vector_rrf_weight=1.1,
            expansion_orchestrator=RetrievalExpansionOrchestrator(
                enabled=True,
                lexical_candidate_size=40,
                vector_candidate_size=40,
                candidate_pool_multiplier=1.0,
                strategies=[strategy],
            ),
        ),
    )

    service.search("什么是向量检索", 5, retrieval_mode="search")
    service.search("什么是向量检索", 5, retrieval_mode="chat")

    assert strategy.calls is not None
    assert [call.mode for call in strategy.calls] == ["search", "chat"]
    assert all(call.query_intent == "definition" for call in strategy.calls)
    assert store.requests is not None
    assert any(request.query_text == "什么是向量检索 扩展问法" for request in store.requests)
