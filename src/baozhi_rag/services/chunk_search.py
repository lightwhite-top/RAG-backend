"""chunk 检索服务与抽象。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.domain.knowledge_file_repository import KnowledgeFileRepository
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
    merged_terms: list[str]
    query_embedding: list[float]
    viewer_user_id: str = ""


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
    merged_terms: list[str]
    score: float | None
    uploader_user_id: str = ""
    visibility_scope: str = ""


class ChunkSearchStore(Protocol):
    """chunk 混合检索存储抽象。"""

    def ensure_index(self) -> None:
        """确保底层检索索引或集合已经就绪。"""
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
    """编排查询词分解、向量化、权限过滤与元数据补齐。"""

    def __init__(
        self,
        term_matcher: MaximumMatchingTermMatcher,
        store: ChunkSearchStore,
        chunk_embedding_service: ChunkEmbeddingService,
        knowledge_file_repository: KnowledgeFileRepository | None = None,
    ) -> None:
        """初始化检索服务。"""
        self._term_matcher = term_matcher
        self._store = store
        self._chunk_embedding_service = chunk_embedding_service
        self._knowledge_file_repository = knowledge_file_repository

    def search(
        self,
        query_text: str,
        size: int,
        *,
        viewer_user_id: str = "",
    ) -> list[ChunkSearchHit]:
        """执行基于 ES 词法与 Milvus 语义的混合检索。"""
        normalized_query = query_text.strip()

        if not normalized_query:
            raise ChunkSearchValidationError("查询文本不能为空")
        if size <= 0:
            raise ChunkSearchValidationError("size 必须大于 0")

        terms = self._term_matcher.extract_terms(normalized_query)
        request = ChunkSearchRequest(
            query_text=normalized_query,
            size=size,
            merged_terms=terms.merged_terms,
            query_embedding=self._chunk_embedding_service.embed_query(normalized_query),
            viewer_user_id=viewer_user_id,
        )
        hits = self._store.search(request)
        return self._hydrate_file_metadata(hits)

    def _hydrate_file_metadata(self, hits: list[ChunkSearchHit]) -> list[ChunkSearchHit]:
        """用数据库中的最新文件元数据覆盖索引内的旧标题。"""
        if not hits or self._knowledge_file_repository is None:
            return hits

        file_ids = list(dict.fromkeys(hit.file_id for hit in hits))
        knowledge_files = self._knowledge_file_repository.get_files_by_ids(file_ids)
        file_map = {knowledge_file.id: knowledge_file for knowledge_file in knowledge_files}

        hydrated_hits: list[ChunkSearchHit] = []
        for hit in hits:
            knowledge_file = file_map.get(hit.file_id)
            if knowledge_file is None:
                continue
            hydrated_hits.append(
                replace(
                    hit,
                    source_filename=knowledge_file.original_filename,
                    storage_key=knowledge_file.storage_key,
                    uploader_user_id=knowledge_file.uploader_user_id,
                    visibility_scope=knowledge_file.visibility_scope.value,
                )
            )
        return hydrated_hits
