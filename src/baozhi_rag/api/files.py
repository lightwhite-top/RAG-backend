"""文件上传路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

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


@router.post("/upload", response_model=MessageResponse, summary="上传文件")
async def upload_files(
    files: Annotated[list[UploadFile], File(description="待上传文件列表")],
    service: Annotated[DocumentPreviewService, Depends(get_document_preview_service)],
) -> MessageResponse:
    """接收多个文件、落盘并返回通用业务响应。

    参数:
        files: 通过 `multipart/form-data` 上传的文件列表。
        service: 上传、切块、向量化与 ES 入库编排服务。

    返回:
        包含成功失败标识、业务码和提示消息的通用响应。
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
        return MessageResponse.failure_response(message=str(exc))
    except InvalidUploadFileError as exc:
        return MessageResponse.failure_response(message=str(exc))
    except FileStorageError as exc:
        return MessageResponse.failure_response(message=str(exc))
    except DocumentChunkingError as exc:
        return MessageResponse.failure_response(message=str(exc))
    except ElasticsearchStoreError as exc:
        return MessageResponse.failure_response(message=str(exc))
    finally:
        for file in files:
            await file.close()

    return MessageResponse.success_response(message="文件上传成功")
