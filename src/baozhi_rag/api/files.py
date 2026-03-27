"""文件上传路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from baozhi_rag.api.dependencies import get_document_preview_service
from baozhi_rag.infra.retrieval.elasticsearch_chunk_store import ElasticsearchStoreError
from baozhi_rag.schemas.files import ChunkPreviewItem, UploadedFileItem, UploadFilesResponse
from baozhi_rag.services.document_chunking import (
    DocumentChunkingError,
    UnsupportedDocumentTypeError,
)
from baozhi_rag.services.document_preview import DocumentPreviewService
from baozhi_rag.services.file_upload import (
    FileStorageError,
    FileUploadInput,
    InvalidUploadFileError,
)

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload", response_model=UploadFilesResponse, summary="上传文件")
async def upload_files(
    files: Annotated[list[UploadFile], File(description="待上传文件列表")],
    service: Annotated[DocumentPreviewService, Depends(get_document_preview_service)],
) -> UploadFilesResponse:
    """接收多个文件、落盘并返回切块预览。

    参数:
        files: 通过 `multipart/form-data` 上传的文件列表。
        service: 上传、切块与可选 ES 入库编排服务。

    返回:
        包含上传元数据、切块状态、切块数量和预览摘要的响应对象。

    异常:
        HTTPException: 当文件名非法、文件格式不支持、存储失败或切块失败时抛出。
    """
    try:
        results = service.upload_and_chunk_files(
            [
                FileUploadInput(
                    filename=file.filename or "",
                    content_type=file.content_type,
                    stream=file.file,
                )
                for file in files
            ]
        )
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except InvalidUploadFileError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except DocumentChunkingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except ElasticsearchStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    finally:
        for file in files:
            await file.close()

    return UploadFilesResponse(
        files=[
            UploadedFileItem(
                file_id=result.upload.file_id,
                original_filename=result.upload.original_filename,
                content_type=result.upload.content_type,
                size=result.upload.size,
                storage_key=result.upload.storage_key,
                uploaded_at=result.upload.uploaded_at,
                chunk_status="success",
                chunk_count=len(result.chunks),
                chunk_preview=[
                    ChunkPreviewItem(
                        chunk_index=chunk.chunk_index,
                        char_count=chunk.char_count,
                        preview_text=chunk.content.replace("\n", " ")[:160],
                        merged_terms=chunk.merged_terms,
                    )
                    for chunk in result.chunks[:3]
                ],
            )
            for result in results
        ]
    )
