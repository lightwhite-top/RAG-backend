"""认证接口。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from baozhi_rag.api.dependencies import get_auth_service, get_current_user
from baozhi_rag.core.request_context import ensure_request_id
from baozhi_rag.domain.user import CurrentUser
from baozhi_rag.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    LoginResponseData,
    RegisterRequest,
    SendRegistrationCodeRequest,
    SendRegistrationCodeResponseData,
    UpdateProfileRequest,
)
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.schemas.users import UserItem, UserRoleItem
from baozhi_rag.services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register/code",
    response_model=SuccessResponse[SendRegistrationCodeResponseData],
    summary="发送注册验证码",
)
def send_registration_code(
    request: Request,
    payload: SendRegistrationCodeRequest,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> SuccessResponse[SendRegistrationCodeResponseData]:
    """发送注册所需的邮箱验证码。"""
    result = service.send_registration_code(email=str(payload.email))
    return SuccessResponse[SendRegistrationCodeResponseData].success(
        message="验证码已发送，请留意邮箱",
        request_id=ensure_request_id(request),
        data=SendRegistrationCodeResponseData(
            expires_in=result.expires_in_seconds,
            expires_at=result.expires_at,
            resend_after=result.resend_interval_seconds,
        ),
    )


@router.post(
    "/register",
    response_model=SuccessResponse[UserItem],
    summary="用户注册",
)
def register(
    request: Request,
    payload: RegisterRequest,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> SuccessResponse[UserItem]:
    """注册普通用户。"""
    user = service.register(
        email=str(payload.email),
        password=payload.password,
        username=payload.username,
        verification_code=payload.verification_code,
    )
    return SuccessResponse[UserItem].success(
        message="注册成功",
        request_id=ensure_request_id(request),
        data=_build_user_item(user),
    )


@router.post(
    "/login",
    response_model=SuccessResponse[LoginResponseData],
    summary="用户登录",
)
def login(
    request: Request,
    payload: LoginRequest,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> SuccessResponse[LoginResponseData]:
    """校验凭证并返回 JWT。"""
    result = service.login(email=str(payload.email), password=payload.password)
    return SuccessResponse[LoginResponseData].success(
        message="登录成功",
        request_id=ensure_request_id(request),
        data=LoginResponseData(
            access_token=result.access_token,
            token_type=result.token_type,
            expires_in=result.expires_in_seconds,
            expires_at=result.expires_at,
            user=_build_user_item(result.user),
        ),
    )


@router.get(
    "/me",
    response_model=SuccessResponse[UserItem],
    summary="获取当前用户信息",
)
def get_me(
    request: Request,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[UserItem]:
    """返回当前登录用户。"""
    return SuccessResponse[UserItem].success(
        message="获取当前用户成功",
        request_id=ensure_request_id(request),
        data=_build_user_item(current_user),
    )


@router.patch(
    "/me",
    response_model=SuccessResponse[UserItem],
    summary="修改当前用户信息",
)
def update_me(
    request: Request,
    payload: UpdateProfileRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> SuccessResponse[UserItem]:
    """更新当前登录用户的用户名。"""
    user = service.update_profile(user_id=current_user.id, username=payload.username)
    return SuccessResponse[UserItem].success(
        message="更新当前用户成功",
        request_id=ensure_request_id(request),
        data=_build_user_item(user),
    )


@router.patch(
    "/password",
    response_model=SuccessResponse[None],
    summary="修改当前用户密码",
)
def change_password(
    request: Request,
    payload: ChangePasswordRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> SuccessResponse[None]:
    """修改当前登录用户密码。"""
    service.change_password(
        user_id=current_user.id,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )
    return SuccessResponse[None].success(
        message="修改密码成功",
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
