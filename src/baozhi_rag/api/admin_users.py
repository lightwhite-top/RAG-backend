"""管理员用户管理接口。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Request

from baozhi_rag.api.dependencies import get_user_admin_service
from baozhi_rag.core.request_context import ensure_request_id
from baozhi_rag.domain.user import CurrentUser, UserRole
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.schemas.users import (
    AdminCreateUserRequest,
    AdminUpdateUserRequest,
    UserItem,
    UserListResponseData,
    UserRoleItem,
)
from baozhi_rag.services.user_admin import UserAdminService

router = APIRouter(prefix="/admin/users", tags=["admin-users"])


@router.get(
    "",
    response_model=SuccessResponse[UserListResponseData],
    summary="分页查询用户",
)
def list_users(
    request: Request,
    service: Annotated[UserAdminService, Depends(get_user_admin_service)],
    q: Annotated[str | None, Query(description="邮箱或用户名关键词")] = None,
    page: Annotated[int, Query(ge=1, description="页码")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="每页数量")] = 20,
) -> SuccessResponse[UserListResponseData]:
    """分页查询用户。"""
    result = service.list_users(query_text=q, page=page, page_size=page_size)
    return SuccessResponse[UserListResponseData].success(
        message="获取用户列表成功",
        request_id=ensure_request_id(request),
        data=UserListResponseData(items=[_build_user_item(item) for item in result.items]),
        meta={
            "page": result.page,
            "page_size": result.page_size,
            "total": result.total,
        },
    )


@router.get(
    "/{user_id}",
    response_model=SuccessResponse[UserItem],
    summary="获取用户详情",
)
def get_user(
    request: Request,
    user_id: Annotated[str, Path(description="用户 ID")],
    service: Annotated[UserAdminService, Depends(get_user_admin_service)],
) -> SuccessResponse[UserItem]:
    """获取单个用户详情。"""
    user = service.get_user(user_id=user_id)
    return SuccessResponse[UserItem].success(
        message="获取用户详情成功",
        request_id=ensure_request_id(request),
        data=_build_user_item(user),
    )


@router.post(
    "",
    response_model=SuccessResponse[UserItem],
    summary="管理员创建用户",
)
def create_user(
    request: Request,
    payload: AdminCreateUserRequest,
    service: Annotated[UserAdminService, Depends(get_user_admin_service)],
) -> SuccessResponse[UserItem]:
    """管理员创建用户。"""
    user = service.create_user(
        email=str(payload.email),
        password=payload.password,
        username=payload.username,
        role=UserRole(payload.role.value),
    )
    return SuccessResponse[UserItem].success(
        message="创建用户成功",
        request_id=ensure_request_id(request),
        data=_build_user_item(user),
    )


@router.patch(
    "/{user_id}",
    response_model=SuccessResponse[UserItem],
    summary="管理员更新用户",
)
def update_user(
    request: Request,
    user_id: Annotated[str, Path(description="用户 ID")],
    payload: AdminUpdateUserRequest,
    service: Annotated[UserAdminService, Depends(get_user_admin_service)],
) -> SuccessResponse[UserItem]:
    """管理员更新用户信息。"""
    user = service.update_user(
        user_id=user_id,
        email=str(payload.email) if payload.email is not None else None,
        username=payload.username,
        role=UserRole(payload.role.value) if payload.role is not None else None,
        password=payload.password,
    )
    return SuccessResponse[UserItem].success(
        message="更新用户成功",
        request_id=ensure_request_id(request),
        data=_build_user_item(user),
    )


@router.delete(
    "/{user_id}",
    response_model=SuccessResponse[None],
    summary="管理员删除用户",
)
def delete_user(
    request: Request,
    user_id: Annotated[str, Path(description="用户 ID")],
    service: Annotated[UserAdminService, Depends(get_user_admin_service)],
) -> SuccessResponse[None]:
    """管理员删除用户。"""
    service.delete_user(user_id=user_id)
    return SuccessResponse[None].success(
        message="删除用户成功",
        request_id=ensure_request_id(request),
    )


def _build_user_item(user: CurrentUser) -> UserItem:
    """把领域用户对象转换为接口模型。"""
    return UserItem(
        id=user.id,
        email=user.email,
        username=user.username,
        role=UserRoleItem(user.role.value),
        created_at=user.created_at,
        updated_at=user.updated_at,
    )
