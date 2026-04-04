"""文件上传与上传任务路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, Query, Request, UploadFile, status

from baozhi_rag.api.dependencies import (
    get_current_user,
    get_knowledge_file_delete_service,
    get_knowledge_file_query_service,
    get_knowledge_upload_service,
)
from baozhi_rag.core.request_context import ensure_request_id
from baozhi_rag.domain.knowledge_upload_task import KnowledgeUploadTask
from baozhi_rag.domain.user import CurrentUser
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.schemas.files import (
    FileUploadSubmitResponseData,
    KnowledgeFileItem,
    KnowledgeFileListResponseData,
    UploadTaskItem,
    UploadTaskListResponseData,
)
from baozhi_rag.services.file_upload import AsyncFileUploadInput
from baozhi_rag.services.knowledge_file_delete import KnowledgeFileDeleteService
from baozhi_rag.services.knowledge_file_query import (
    KnowledgeFileListItemResult,
    KnowledgeFileListResult,
    KnowledgeFileQueryService,
)
from baozhi_rag.services.upload_tasks import KnowledgeUploadService

router = APIRouter(prefix="/files", tags=["files"])


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
            items=[_to_knowledge_file_item(item) for item in result.items]
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
            items=[_to_knowledge_file_item(item) for item in result.items]
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


def _to_upload_task_item(task: KnowledgeUploadTask) -> UploadTaskItem:
    """把上传任务实体转换为接口响应结构。"""
    return UploadTaskItem(
        task_id=task.id,
        status=task.status.value,
        stage=task.stage.value,
        original_filename=task.original_filename,
        content_type=task.content_type,
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


def _to_knowledge_file_item(item: KnowledgeFileListItemResult) -> KnowledgeFileItem:
    """把文件列表结果转换为接口模型。"""
    return KnowledgeFileItem(
        file_id=item.file_id,
        uploader_user_id=item.uploader_user_id,
        original_filename=item.original_filename,
        content_type=item.content_type,
        size=item.size,
        storage_key=item.storage_key,
        file_url=item.file_url,
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
