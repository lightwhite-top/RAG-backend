"""文件上传路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from baozhi_rag.api.dependencies import get_document_preview_service
from baozhi_rag.infra.retrieval.elasticsearch_chunk_store import ElasticsearchStoreError
from baozhi_rag.schemas.common import MessageResponse
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


@router.post(
    "/upload",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传文件",
)
async def upload_files(
    files: Annotated[list[UploadFile], File(description="待上传文件列表")],
    service: Annotated[DocumentPreviewService, Depends(get_document_preview_service)],
) -> MessageResponse:
    """接收多个文件、落盘并返回通用成功消息。

    参数:
        files: 通过 `multipart/form-data` 上传的文件列表。
        service: 上传、切块、向量化与 ES 入库编排服务。

    返回:
        表示文件上传和入库完成的通用消息响应。

    异常:
        HTTPException: 当文件名非法、文件格式不支持、存储失败或切块失败时抛出。
    """
    try:
        service.upload_and_chunk_files(
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

    return MessageResponse(message="文件上传成功")
