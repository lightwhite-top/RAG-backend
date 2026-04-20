"""文件上传、列表查询与同源访问路由。"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from pathlib import PurePath
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Path, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from baozhi_rag.api.chat import _resolve_extension
from baozhi_rag.api.dependencies import (
    get_current_user,
    get_knowledge_file_access_service,
    get_knowledge_file_delete_service,
    get_knowledge_file_query_service,
    get_knowledge_upload_service,
)
from baozhi_rag.api.file_access import build_backend_file_access_payload
from baozhi_rag.core.request_context import ensure_request_id
from baozhi_rag.domain.knowledge_upload_task import KnowledgeUploadTask
from baozhi_rag.domain.user import CurrentUser
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.schemas.files import (
    FileUploadSubmitResponseData,
    KnowledgeFileItem,
    KnowledgeFileListResponseData,
    KnowledgeFilePurgeResponseData,
    UploadTaskItem,
    UploadTaskListResponseData,
)
from baozhi_rag.services.document_chunking import UnsupportedDocumentTypeError
from baozhi_rag.services.file_upload import AsyncFileUploadInput
from baozhi_rag.services.knowledge_file_access import KnowledgeFileAccessService
from baozhi_rag.services.knowledge_file_delete import KnowledgeFileDeleteService
from baozhi_rag.services.knowledge_file_query import (
    KnowledgeFileListItemResult,
    KnowledgeFileListResult,
    KnowledgeFileQueryService,
)
from baozhi_rag.services.upload_tasks import KnowledgeUploadService

router = APIRouter(prefix="/files", tags=["files"])
_SUPPORTED_UPLOAD_SUFFIXES = {".docx", ".doc", ".pdf"}


@router.get(
    "/global",
    response_model=SuccessResponse[KnowledgeFileListResponseData],
    summary="分页查询全局文件",
)
def list_global_files(
    request: Request,
    service: Annotated[KnowledgeFileQueryService, Depends(get_knowledge_file_query_service)],
    page: Annotated[int, Query(ge=1, description="页码")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="每页数量")] = 20,
) -> SuccessResponse[KnowledgeFileListResponseData]:
    """分页查询管理员上传的全局文件。"""
    result = service.list_global_files(page=page, page_size=page_size)
    return SuccessResponse[KnowledgeFileListResponseData].success(
        message="获取全局文件列表成功",
        request_id=ensure_request_id(request),
        data=KnowledgeFileListResponseData(
            items=[_to_knowledge_file_item(request, item) for item in result.items]
        ),
        meta=_build_page_meta(result),
    )


@router.get(
    "/mine",
    response_model=SuccessResponse[KnowledgeFileListResponseData],
    summary="分页查询我的文件",
)
def list_my_files(
    request: Request,
    service: Annotated[KnowledgeFileQueryService, Depends(get_knowledge_file_query_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    page: Annotated[int, Query(ge=1, description="页码")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="每页数量")] = 20,
) -> SuccessResponse[KnowledgeFileListResponseData]:
    """分页查询当前用户自己上传的文件。"""
    result = service.list_my_files(current_user=current_user, page=page, page_size=page_size)
    return SuccessResponse[KnowledgeFileListResponseData].success(
        message="获取我的文件列表成功",
        request_id=ensure_request_id(request),
        data=KnowledgeFileListResponseData(
            items=[_to_knowledge_file_item(request, item) for item in result.items]
        ),
        meta=_build_page_meta(result),
    )


@router.post(
    "/upload",
    response_model=SuccessResponse[FileUploadSubmitResponseData],
    summary="提交文件上传任务",
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_files(
    request: Request,
    files: Annotated[list[UploadFile], File(description="待上传文件列表")],
    service: Annotated[KnowledgeUploadService, Depends(get_knowledge_upload_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[FileUploadSubmitResponseData]:
    """接收多个文件并创建或复用后台上传任务。"""
    request_id = ensure_request_id(request)
    for file in files:
        suffix = PurePath(file.filename or "").suffix.lower()
        if suffix not in _SUPPORTED_UPLOAD_SUFFIXES:
            msg = f"暂不支持的文件格式: {suffix or 'unknown'}"
            raise UnsupportedDocumentTypeError(msg)

    try:
        tasks = await service.submit_files(
            [
                AsyncFileUploadInput(
                    filename=file.filename or "",
                    content_type=file.content_type,
                    stream=file,
                )
                for file in files
            ],
            current_user=current_user,
            request_id=request_id,
        )
    finally:
        for file in files:
            await file.close()

    response_data = FileUploadSubmitResponseData(
        file_count=len(tasks),
        tasks=[_to_upload_task_item(task) for task in tasks],
    )
    return SuccessResponse[FileUploadSubmitResponseData].success(
        message="上传任务已创建",
        request_id=request_id,
        data=response_data,
    )


@router.get(
    "/upload-tasks",
    response_model=SuccessResponse[UploadTaskListResponseData],
    summary="查询上传任务列表",
)
def list_upload_tasks(
    request: Request,
    service: Annotated[KnowledgeUploadService, Depends(get_knowledge_upload_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[UploadTaskListResponseData]:
    """查询当前用户最近的上传任务。"""
    request_id = ensure_request_id(request)
    tasks = service.list_tasks(current_user=current_user)
    return SuccessResponse[UploadTaskListResponseData].success(
        message="查询上传任务成功",
        request_id=request_id,
        data=UploadTaskListResponseData(
            task_count=len(tasks),
            tasks=[_to_upload_task_item(task) for task in tasks],
        ),
    )


@router.get(
    "/upload-tasks/{task_id}",
    response_model=SuccessResponse[UploadTaskItem],
    summary="查询单条上传任务",
)
def get_upload_task(
    task_id: str,
    request: Request,
    service: Annotated[KnowledgeUploadService, Depends(get_knowledge_upload_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[UploadTaskItem]:
    """查询当前用户的单条上传任务。"""
    request_id = ensure_request_id(request)
    task = service.get_task(task_id=task_id, current_user=current_user)
    return SuccessResponse[UploadTaskItem].success(
        message="查询上传任务成功",
        request_id=request_id,
        data=_to_upload_task_item(task),
    )


@router.post(
    "/upload-tasks/{task_id}/retry",
    response_model=SuccessResponse[UploadTaskItem],
    summary="重试上传任务",
)
def retry_upload_task(
    task_id: str,
    request: Request,
    service: Annotated[KnowledgeUploadService, Depends(get_knowledge_upload_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[UploadTaskItem]:
    """将失败上传任务重新入队。"""
    request_id = ensure_request_id(request)
    task = service.retry_task(task_id=task_id, current_user=current_user)
    return SuccessResponse[UploadTaskItem].success(
        message="上传任务已重新入队",
        request_id=request_id,
        data=_to_upload_task_item(task),
    )


@router.get(
    "/{file_id}/content",
    summary="访问知识文件内容",
)
def get_file_content(
    file_id: Annotated[str, Path(description="文件 ID")],
    service: Annotated[
        KnowledgeFileAccessService,
        Depends(get_knowledge_file_access_service),
    ],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> StreamingResponse:
    """通过后端同源接口输出知识文件内容，避免前端直连对象存储。"""
    access_result = service.open_file(file_id=file_id, current_user=current_user)
    knowledge_file = access_result.knowledge_file
    normalized_content_type = knowledge_file.content_type.strip() or "application/octet-stream"
    return _build_binary_stream_response(
        filename=knowledge_file.original_filename,
        content_type=normalized_content_type,
        content_iter=access_result.content_iter,
    )


@router.get(
    "/image-assets/{asset_id}/content",
    summary="访问图片原图内容",
)
def get_image_asset_content(
    asset_id: Annotated[str, Path(description="图片资产 ID")],
    service: Annotated[
        KnowledgeFileAccessService,
        Depends(get_knowledge_file_access_service),
    ],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> StreamingResponse:
    """通过后端同源接口输出图片原图内容。"""
    access_result = service.open_image_asset(asset_id=asset_id, current_user=current_user)
    return _build_binary_stream_response(
        filename=access_result.filename,
        content_type=access_result.content_type,
        content_iter=access_result.content_iter,
    )


@router.get(
    "/image-assets/{asset_id}/preview",
    summary="访问图片缩略图内容",
)
def get_image_asset_preview(
    asset_id: Annotated[str, Path(description="图片资产 ID")],
    service: Annotated[
        KnowledgeFileAccessService,
        Depends(get_knowledge_file_access_service),
    ],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> StreamingResponse:
    """通过后端同源接口输出图片缩略图内容。"""
    access_result = service.open_image_asset(
        asset_id=asset_id,
        current_user=current_user,
        use_preview=True,
    )
    return _build_binary_stream_response(
        filename=access_result.filename,
        content_type=access_result.content_type,
        content_iter=access_result.content_iter,
    )


@router.delete(
    "/mine",
    response_model=SuccessResponse[KnowledgeFilePurgeResponseData],
    summary="清空我的知识库数据",
)
def delete_my_all_files(
    request: Request,
    service: Annotated[KnowledgeFileDeleteService, Depends(get_knowledge_file_delete_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[KnowledgeFilePurgeResponseData]:
    """清空当前用户上传的全部知识文件与上传任务。"""
    request_id = ensure_request_id(request)
    result = service.delete_all_files(current_user=current_user)
    return SuccessResponse[KnowledgeFilePurgeResponseData].success(
        message="清空我的知识库数据成功",
        request_id=request_id,
        data=KnowledgeFilePurgeResponseData(
            deleted_file_count=result.deleted_file_count,
            deleted_task_count=result.deleted_task_count,
        ),
    )


@router.delete(
    "/{file_id}",
    response_model=SuccessResponse[None],
    summary="删除我的文件",
)
def delete_my_file(
    request: Request,
    file_id: Annotated[str, Path(description="文件 ID")],
    service: Annotated[KnowledgeFileDeleteService, Depends(get_knowledge_file_delete_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[None]:
    """删除当前用户自己上传的知识文件。"""
    request_id = ensure_request_id(request)
    service.delete_file(file_id=file_id, current_user=current_user)
    return SuccessResponse[None].success(
        message="删除文件成功",
        request_id=request_id,
    )


def _build_content_disposition(filename: str) -> str:
    """构造兼容中文文件名的 Content-Disposition。"""
    safe_filename = filename.strip() or "download.bin"
    ascii_fallback = _build_ascii_filename_fallback(safe_filename)
    encoded_filename = quote(safe_filename, safe="")
    return f"inline; filename=\"{ascii_fallback}\"; filename*=UTF-8''{encoded_filename}"


def _build_ascii_filename_fallback(filename: str) -> str:
    """生成仅包含 ASCII 的文件名兜底值，避免响应头编码失败。"""
    # Starlette 会把响应头按 latin-1 编码；这里必须保证 filename= 部分可安全编码。
    file_path = PurePath(filename)
    normalized_stem = unicodedata.normalize("NFKD", file_path.stem)
    sanitized_stem = (
        normalized_stem.encode("ascii", "ignore")
        .decode("ascii")
        .replace("\\", "_")
        .replace("/", "_")
        .replace('"', "_")
        .replace(";", "_")
        .strip(" ._")
    )
    if not sanitized_stem:
        sanitized_stem = "download"

    safe_suffix = file_path.suffix
    if safe_suffix and safe_suffix.isascii():
        safe_suffix = safe_suffix.replace('"', "").replace(";", "").replace(" ", "")
    else:
        safe_suffix = ".bin"
    return f"{sanitized_stem}{safe_suffix or '.bin'}"


def _build_binary_stream_response(
    *,
    filename: str,
    content_type: str,
    content_iter: Iterator[bytes],
) -> StreamingResponse:
    """构造统一的同源二进制流响应。"""
    return StreamingResponse(
        content_iter,
        media_type=content_type,
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": _build_content_disposition(filename),
        },
    )


def _to_upload_task_item(task: KnowledgeUploadTask) -> UploadTaskItem:
    """把上传任务实体转换为接口响应结构。"""
    return UploadTaskItem(
        task_id=task.id,
        status=task.status.value,
        stage=task.stage.value,
        original_filename=task.original_filename,
        content_type=task.content_type,
        extension=_resolve_extension(None, source_filename=task.original_filename),
        size=task.size,
        file_id=task.file_id,
        chunk_count=task.chunk_count,
        deduplicated=task.deduplicated,
        replaced=task.replaced,
        title_updated=task.title_updated,
        error_code=task.error_code,
        error_message=task.error_message,
        created_at=task.created_at,
        updated_at=task.updated_at,
        completed_at=task.completed_at,
    )


def _to_knowledge_file_item(
    request: Request,
    item: KnowledgeFileListItemResult,
) -> KnowledgeFileItem:
    """把文件列表结果转换为接口模型。"""
    file_access = build_backend_file_access_payload(request, file_id=item.file_id)
    return KnowledgeFileItem(
        file_id=item.file_id,
        uploader_user_id=item.uploader_user_id,
        original_filename=item.original_filename,
        content_type=item.content_type,
        extension=_resolve_extension(None, source_filename=item.original_filename),
        size=item.size,
        storage_key=item.storage_key,
        file_url=file_access["url"] or item.file_url,
        storage_provider=item.storage_provider,
        visibility_scope=item.visibility_scope,
        chunk_count=item.chunk_count,
        uploaded_at=item.uploaded_at,
        updated_at=item.updated_at,
    )


def _build_page_meta(result: KnowledgeFileListResult) -> dict[str, int]:
    """构造统一分页元信息。"""
    return {
        "page": result.page,
        "page_size": result.page_size,
        "total": result.total,
    }
