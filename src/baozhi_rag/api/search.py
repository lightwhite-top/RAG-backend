"""chunk 检索接口。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from baozhi_rag.api.dependencies import get_chunk_search_service
from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.infra.retrieval.elasticsearch_chunk_store import (
    ElasticsearchDependencyError,
    ElasticsearchSearchError,
    ElasticsearchStoreError,
)
from baozhi_rag.schemas.search import ChunkSearchHitItem, ChunkSearchResponse
from baozhi_rag.services.chunk_search import ChunkSearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/chunks", response_model=ChunkSearchResponse, summary="检索 chunk")
def search_chunks(
    q: Annotated[str, Query(min_length=1, description="查询文本")],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[ChunkSearchService, Depends(get_chunk_search_service)],
    size: Annotated[int | None, Query(gt=0, le=50, description="返回条数")] = None,
) -> ChunkSearchResponse:
    """执行基于全文和领域词的 chunk 混合检索。"""
    if not settings.es_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ES 检索未启用，请先配置 ES_ENABLED=true",
        )

    try:
        hits = service.search(q, size or settings.search_default_size)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ElasticsearchDependencyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except (ElasticsearchSearchError, ElasticsearchStoreError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return ChunkSearchResponse(
        query=q,
        size=len(hits),
        hits=[
            ChunkSearchHitItem(
                chunk_id=hit.chunk_id,
                file_id=hit.file_id,
                source_filename=hit.source_filename,
                storage_key=hit.storage_key,
                chunk_index=hit.chunk_index,
                char_count=hit.char_count,
                content=hit.content,
                fmm_terms=hit.fmm_terms,
                bmm_terms=hit.bmm_terms,
                merged_terms=hit.merged_terms,
                score=hit.score,
            )
            for hit in hits
        ],
    )
