"""用户仓储抽象。"""

from __future__ import annotations

from typing import Protocol

from baozhi_rag.domain.user import User, UserListPage, UserRole


class UserRepository(Protocol):
    """用户仓储协议。"""

    def create_user(
        self,
        *,
        email: str,
        username: str,
        password_hash: str,
        role: UserRole,
    ) -> User:
        """创建用户并返回完整实体。"""

    def get_user_by_id(self, user_id: str) -> User | None:
        """按用户 ID 查询用户。"""

    def get_user_by_email(self, email: str) -> User | None:
        """按邮箱查询用户。"""

    def list_users(
        self,
        *,
        query_text: str | None,
        page: int,
        page_size: int,
    ) -> UserListPage:
        """分页查询用户。"""

    def update_user(
        self,
        user_id: str,
        *,
        email: str | None = None,
        username: str | None = None,
        role: UserRole | None = None,
        password_hash: str | None = None,
    ) -> User | None:
        """更新指定用户并返回最新实体。"""

    def delete_user(self, user_id: str) -> bool:
        """删除用户，返回是否删除成功。"""
