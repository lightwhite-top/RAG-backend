"""文件上传路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile

from baozhi_rag.api.dependencies import get_current_user, get_document_preview_service
from baozhi_rag.core.request_context import ensure_request_id
from baozhi_rag.domain.user import CurrentUser
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.schemas.files import FileUploadResponseData, UploadedFileItem
from baozhi_rag.services.document_preview import DocumentPreviewService
from baozhi_rag.services.file_upload import FileUploadInput, UploadedFileResult

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
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[FileUploadResponseData]:
    """接收多个文件并完成 OSS 上传、切块和索引。"""
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
            ],
            current_user=current_user,
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
                storage_provider=result.upload.storage_provider,
                deduplicated=result.upload.deduplicated,
                replaced=result.upload.replaced,
                chunk_count=result.upload.chunk_count,
                uploaded_at=result.upload.uploaded_at,
            )
            for result in results
        ],
    )
    return SuccessResponse[FileUploadResponseData].success(
        message=_resolve_upload_message([result.upload for result in results]),
        request_id=request_id,
        data=response_data,
    )


def _resolve_upload_message(results: list[UploadedFileResult]) -> str:
    """根据上传结果生成统一提示语。"""
    if len(results) != 1:
        return "文件上传成功"

    result = results[0]
    if result.title_updated:
        return "文件内容重复入库，已更新标题"
    if result.deduplicated:
        return "文件重复入库"
    return "文件上传成功"
