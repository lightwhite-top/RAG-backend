"""API 依赖构造函数。"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.infra.llm.aliyun_model_studio import AlibabaModelStudioClient
from baozhi_rag.infra.retrieval.elasticsearch_chunk_store import ElasticsearchChunkStore
from baozhi_rag.infra.storage.local_file_store import LocalFileStore
from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.chunk_search import ChunkSearchService
from baozhi_rag.services.document_chunking import DocumentChunkService
from baozhi_rag.services.document_preview import DocumentPreviewService
from baozhi_rag.services.file_upload import FileUploadService
from baozhi_rag.services.term_matching import build_default_term_matcher


def _build_chunk_embedding_service(settings: Settings) -> ChunkEmbeddingService | None:
    """按配置构造 chunk 向量化服务。"""
    if not settings.chunk_embedding_enabled:
        return None
    return ChunkEmbeddingService(AlibabaModelStudioClient.from_settings(settings))


def get_document_preview_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentPreviewService:
    """构造文件上传与切块预览服务。"""
    file_store = LocalFileStore(settings.upload_root_dir)
    term_matcher = build_default_term_matcher(settings.domain_dictionary_path)
    chunk_store = ElasticsearchChunkStore.from_settings(settings)
    chunk_embedding_service = _build_chunk_embedding_service(settings)

    return DocumentPreviewService(
        file_upload_service=FileUploadService(file_store),
        chunk_service=DocumentChunkService(
            chunk_size=settings.doc_chunk_size,
            chunk_overlap=settings.doc_chunk_overlap,
            convert_temp_dir=settings.doc_convert_temp_dir,
            term_matcher=term_matcher,
        ),
        file_store=file_store,
        chunk_store=chunk_store,
        chunk_embedding_service=chunk_embedding_service,
    )


def get_chunk_search_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChunkSearchService:
    """构造 chunk 检索服务。"""
    return ChunkSearchService(
        term_matcher=build_default_term_matcher(settings.domain_dictionary_path),
        store=ElasticsearchChunkStore.from_settings(settings),
        chunk_embedding_service=_build_chunk_embedding_service(settings),
    )
