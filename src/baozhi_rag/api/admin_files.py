"""管理员知识文件管理接口。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from baozhi_rag.api.dependencies import get_current_user, get_knowledge_file_delete_service
from baozhi_rag.core.request_context import ensure_request_id
from baozhi_rag.domain.user import CurrentUser
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.schemas.files import KnowledgeFilePurgeResponseData
from baozhi_rag.services.knowledge_file_delete import KnowledgeFileDeleteService

router = APIRouter(prefix="/admin/files", tags=["admin-files"])


@router.delete(
    "",
    response_model=SuccessResponse[KnowledgeFilePurgeResponseData],
    summary="管理员清空全站知识库数据",
)
def delete_all_files_globally(
    request: Request,
    service: Annotated[KnowledgeFileDeleteService, Depends(get_knowledge_file_delete_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[KnowledgeFilePurgeResponseData]:
    """管理员清空全站知识文件与上传任务。"""
    request_id = ensure_request_id(request)
    result = service.delete_all_files_globally(current_user=current_user)
    return SuccessResponse[KnowledgeFilePurgeResponseData].success(
        message="清空全站知识库数据成功",
        request_id=request_id,
        data=KnowledgeFilePurgeResponseData(
            deleted_file_count=result.deleted_file_count,
            deleted_task_count=result.deleted_task_count,
        ),
    )
