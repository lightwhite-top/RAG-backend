"""用户与管理员接口模型。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserRoleItem(StrEnum):
    """接口层可见的用户角色。"""

    ADMIN = "admin"
    USER = "user"


class UserItem(BaseModel):
    """用户信息响应。"""

    id: str = Field(description="用户唯一标识")
    email: str = Field(description="用户邮箱")
    username: str = Field(description="用户名")
    role: UserRoleItem = Field(description="用户角色")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")


class UserListResponseData(BaseModel):
    """用户列表响应数据。"""

    items: list[UserItem] = Field(description="用户列表")


class AdminCreateUserRequest(BaseModel):
    """管理员创建用户请求。"""

    email: EmailStr = Field(description="用户邮箱")
    password: str = Field(min_length=8, max_length=128, description="用户密码")
    username: str = Field(min_length=1, max_length=64, description="用户名")
    role: UserRoleItem = Field(description="用户角色")

    @field_validator("username")
    @classmethod
    def strip_username(cls, value: str) -> str:
        """清理用户名首尾空白。"""
        normalized_username = value.strip()
        if not normalized_username:
            raise ValueError("用户名不能为空")
        return normalized_username


class AdminUpdateUserRequest(BaseModel):
    """管理员更新用户请求。"""

    email: EmailStr | None = Field(default=None, description="用户邮箱")
    username: str | None = Field(default=None, min_length=1, max_length=64, description="用户名")
    role: UserRoleItem | None = Field(default=None, description="用户角色")
    password: str | None = Field(default=None, min_length=8, max_length=128, description="新密码")

    @field_validator("username")
    @classmethod
    def strip_optional_username(cls, value: str | None) -> str | None:
        """清理可选用户名首尾空白。"""
        if value is None:
            return None
        normalized_username = value.strip()
        if not normalized_username:
            raise ValueError("用户名不能为空")
        return normalized_username
