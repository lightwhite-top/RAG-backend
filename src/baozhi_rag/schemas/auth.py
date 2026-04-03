"""认证接口模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

from baozhi_rag.schemas.users import UserItem


class RegisterRequest(BaseModel):
    """注册请求体。"""

    email: EmailStr = Field(description="注册邮箱")
    password: str = Field(min_length=8, max_length=128, description="登录密码")
    username: str = Field(min_length=1, max_length=64, description="用户名")
    verification_code: str = Field(
        min_length=4,
        max_length=8,
        description="邮箱验证码",
    )

    @field_validator("username")
    @classmethod
    def strip_username(cls, value: str) -> str:
        """清理用户名首尾空白。"""
        normalized_username = value.strip()
        if not normalized_username:
            raise ValueError("用户名不能为空")
        return normalized_username

    @field_validator("verification_code")
    @classmethod
    def normalize_verification_code(cls, value: str) -> str:
        """清理验证码首尾空白并限制为数字。"""
        normalized_code = value.strip()
        if not normalized_code:
            raise ValueError("验证码不能为空")
        if not normalized_code.isdigit():
            raise ValueError("验证码必须为数字")
        return normalized_code


class SendRegistrationCodeRequest(BaseModel):
    """发送注册验证码请求体。"""

    email: EmailStr = Field(description="待注册邮箱")


class SendRegistrationCodeResponseData(BaseModel):
    """发送注册验证码成功后的响应数据。"""

    expires_in: int = Field(description="验证码剩余有效秒数")
    resend_after: int = Field(description="再次发送前需等待的秒数")


class LoginRequest(BaseModel):
    """登录请求体。"""

    email: EmailStr = Field(description="登录邮箱")
    password: str = Field(min_length=8, max_length=128, description="登录密码")


class LoginResponseData(BaseModel):
    """登录成功后的响应数据。"""

    access_token: str = Field(description="JWT 访问令牌")
    token_type: str = Field(description="令牌类型")
    expires_in: int = Field(description="剩余有效秒数")
    expires_at: datetime = Field(description="令牌到期时间")
    user: UserItem = Field(description="当前登录用户信息")


class UpdateProfileRequest(BaseModel):
    """当前用户资料更新请求。"""

    username: str = Field(min_length=1, max_length=64, description="新的用户名")

    @field_validator("username")
    @classmethod
    def strip_username(cls, value: str) -> str:
        """清理用户名首尾空白。"""
        normalized_username = value.strip()
        if not normalized_username:
            raise ValueError("用户名不能为空")
        return normalized_username


class ChangePasswordRequest(BaseModel):
    """当前用户密码修改请求。"""

    current_password: str = Field(min_length=8, max_length=128, description="当前密码")
    new_password: str = Field(min_length=8, max_length=128, description="新密码")
