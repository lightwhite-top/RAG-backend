"""chunk 检索接口。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from baozhi_rag.api.dependencies import get_chunk_search_service
from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.core.request_context import ensure_request_id
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.schemas.search import ChunkSearchHitItem, ChunkSearchResponse
from baozhi_rag.services.chunk_search import ChunkSearchService

router = APIRouter(prefix="/search", tags=["search"])


def get_search_default_size(
    settings: Annotated[Settings, Depends(get_settings)],
) -> int:
    """读取检索接口默认返回条数。"""
    return settings.search_default_size


@router.get(
    "/chunks",
    response_model=SuccessResponse[ChunkSearchResponse],
    summary="检索 chunk",
)
def search_chunks(
    request: Request,
    q: Annotated[str, Query(min_length=1, description="查询文本")],
    default_size: Annotated[int, Depends(get_search_default_size)],
    service: Annotated[ChunkSearchService, Depends(get_chunk_search_service)],
    size: Annotated[int | None, Query(gt=0, le=50, description="返回条数")] = None,
) -> SuccessResponse[ChunkSearchResponse]:
    """执行基于全文和领域词的 chunk 混合检索。"""
    request_id = ensure_request_id(request)
    hits = service.search(q, size or default_size)

    return SuccessResponse[ChunkSearchResponse].success(
        message="检索成功",
        request_id=request_id,
        data=ChunkSearchResponse(
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
        ),
    )
