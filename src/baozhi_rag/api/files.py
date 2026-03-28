"""文件上传路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile

from baozhi_rag.api.dependencies import get_document_preview_service
from baozhi_rag.core.request_context import ensure_request_id
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.schemas.files import FileUploadResponseData, UploadedFileItem
from baozhi_rag.services.document_preview import DocumentPreviewService
from baozhi_rag.services.file_upload import FileUploadInput

router = APIRouter(prefix="/files", tags=["files"])


@router.post(
    "/upload",
    response_model=SuccessResponse[FileUploadResponseData],
    summary="上传文件",
)
async def upload_files(
    request: Request,
    files: Annotated[list[UploadFile], File(description="待上传文件列表")],
    service: Annotated[DocumentPreviewService, Depends(get_document_preview_service)],
) -> SuccessResponse[FileUploadResponseData]:
    """接收多个文件、落盘并返回通用业务响应。

    参数:
        files: 通过 `multipart/form-data` 上传的文件列表。
        service: 上传、切块、向量化与 ES 入库编排服务。

    返回:
        包含成功失败标识、业务码和提示消息的通用响应。
    """
    request_id = ensure_request_id(request)

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
    finally:
        for file in files:
            await file.close()

    response_data = FileUploadResponseData(
        file_count=len(results),
        files=[
            UploadedFileItem(
                file_id=result.upload.file_id,
                original_filename=result.upload.original_filename,
                content_type=result.upload.content_type,
                size=result.upload.size,
                storage_key=result.upload.storage_key,
                chunk_count=len(result.chunks),
                uploaded_at=result.upload.uploaded_at,
            )
            for result in results
        ],
    )
    return SuccessResponse[FileUploadResponseData].success(
        message="文件上传成功",
        request_id=request_id,
        data=response_data,
    )
