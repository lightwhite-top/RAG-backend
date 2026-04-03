"""用户领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class UserRole(StrEnum):
    """系统内支持的用户角色。"""

    ADMIN = "admin"
    USER = "user"


@dataclass(frozen=True, slots=True)
class User:
    """用户领域实体。"""

    id: str
    email: str
    username: str
    password_hash: str
    role: UserRole
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """鉴权成功后注入请求上下文的当前用户。"""

    id: str
    email: str
    username: str
    role: UserRole
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class UserListPage:
    """用户分页查询结果。"""

    items: list[User]
    total: int
    page: int
    page_size: int


def build_current_user(user: User) -> CurrentUser:
    """从用户实体构造当前用户上下文。"""
    return CurrentUser(
        id=user.id,
        email=user.email,
        username=user.username,
        role=user.role,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )
