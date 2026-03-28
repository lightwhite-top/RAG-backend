"""chunk 检索服务与抽象。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.term_matching import MaximumMatchingTermMatcher

if TYPE_CHECKING:
    from baozhi_rag.services.document_chunking import DocumentChunk


class ChunkSearchValidationError(AppError):
    """检索请求参数非法。"""

    default_message = "检索参数非法"
    default_error_code = "chunk_search_validation_error"
    default_status_code = status.HTTP_400_BAD_REQUEST


@dataclass(frozen=True, slots=True)
class ChunkSearchRequest:
    """chunk 检索请求。"""

    query_text: str
    size: int
    fmm_terms: list[str]
    bmm_terms: list[str]
    merged_terms: list[str]
    query_embedding: list[float]


@dataclass(frozen=True, slots=True)
class ChunkSearchHit:
    """chunk 检索命中结果。"""

    chunk_id: str
    file_id: str
    source_filename: str
    storage_key: str
    chunk_index: int
    char_count: int
    content: str
    fmm_terms: list[str]
    bmm_terms: list[str]
    merged_terms: list[str]
    score: float | None


class ChunkSearchStore(Protocol):
    """chunk 混合检索存储抽象。"""

    def ensure_index(self) -> None:
        """确保索引存在。"""
        ...

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """写入 chunk 文档。"""
        ...

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """按文件标识删除 chunk。"""
        ...

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        """执行 chunk 检索。"""
        ...


class ChunkSearchService:
    """编排查询词分解、向量化与混合检索。"""

    def __init__(
        self,
        term_matcher: MaximumMatchingTermMatcher,
        store: ChunkSearchStore,
        chunk_embedding_service: ChunkEmbeddingService,
    ) -> None:
        """初始化检索服务。"""
        self._term_matcher = term_matcher
        self._store = store
        self._chunk_embedding_service = chunk_embedding_service

    def search(self, query_text: str, size: int) -> list[ChunkSearchHit]:
        """执行基于 ES 词法与 Milvus 语义的混合检索。"""
        normalized_query = query_text.strip()
        if not normalized_query:
            msg = "查询文本不能为空"
            raise ChunkSearchValidationError(msg)
        if size <= 0:
            msg = "size 必须大于 0"
            raise ChunkSearchValidationError(msg)

        terms = self._term_matcher.extract_terms(normalized_query)
        request = ChunkSearchRequest(
            query_text=normalized_query,
            size=size,
            fmm_terms=terms.fmm_terms,
            bmm_terms=terms.bmm_terms,
            merged_terms=terms.merged_terms,
            query_embedding=self._chunk_embedding_service.embed_query(normalized_query),
        )
        return self._store.search(request)
