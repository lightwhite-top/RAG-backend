from __future__ import annotations

from dataclasses import dataclass

from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest, ChunkSearchService
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

    def __post_init__(self) -> None:
        if self.requests is None:
            self.requests = []

    def ensure_index(self) -> None:  # pragma: no cover - 协议占位
        return None

    def index_chunks(self, chunks: list[object]) -> int:  # pragma: no cover - 协议占位
        return len(chunks)

    def delete_chunks_by_file_id(self, file_id: str) -> None:  # pragma: no cover - 协议占位
        return None

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        self.requests.append(request)
        return []


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
