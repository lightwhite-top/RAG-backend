"""管理员用户管理服务。"""

from __future__ import annotations

from dataclasses import dataclass

from baozhi_rag.domain.user import CurrentUser, UserRole, build_current_user
from baozhi_rag.domain.user_errors import InvalidUserOperationError, UserNotFoundError
from baozhi_rag.domain.user_repository import UserRepository
from baozhi_rag.infra.security.passwords import PasswordHasherAdapter


@dataclass(frozen=True, slots=True)
class UserListResult:
    """用户列表结果。"""

    items: list[CurrentUser]
    total: int
    page: int
    page_size: int


class UserAdminService:
    """管理员视角的用户管理编排服务。"""

    def __init__(
        self,
        *,
        user_repository: UserRepository,
        password_hasher: PasswordHasherAdapter,
    ) -> None:
        """初始化管理员用户服务。"""
        self._user_repository = user_repository
        self._password_hasher = password_hasher

    def list_users(
        self,
        *,
        query_text: str | None,
        page: int,
        page_size: int,
    ) -> UserListResult:
        """分页查询用户。"""
        page_result = self._user_repository.list_users(
            query_text=self._normalize_query_text(query_text),
            page=page,
            page_size=page_size,
        )
        return UserListResult(
            items=[build_current_user(item) for item in page_result.items],
            total=page_result.total,
            page=page_result.page,
            page_size=page_result.page_size,
        )

    def get_user(self, *, user_id: str) -> CurrentUser:
        """查询单个用户详情。"""
        user = self._user_repository.get_user_by_id(user_id)
        if user is None:
            raise UserNotFoundError()
        return build_current_user(user)

    def create_user(
        self,
        *,
        email: str,
        password: str,
        username: str,
        role: UserRole,
    ) -> CurrentUser:
        """由管理员创建用户。"""
        user = self._user_repository.create_user(
            email=self._normalize_email(email),
            username=self._normalize_username(username),
            password_hash=self._password_hasher.hash_password(password),
            role=role,
        )
        return build_current_user(user)

    def update_user(
        self,
        *,
        user_id: str,
        email: str | None = None,
        username: str | None = None,
        role: UserRole | None = None,
        password: str | None = None,
    ) -> CurrentUser:
        """更新用户资料、角色或密码。"""
        normalized_email = self._normalize_email(email) if email is not None else None
        normalized_username = self._normalize_username(username) if username is not None else None
        password_hash = (
            self._password_hasher.hash_password(password) if password is not None else None
        )
        if (
            normalized_email is None
            and normalized_username is None
            and role is None
            and password_hash is None
        ):
            raise InvalidUserOperationError("至少提供一个待更新字段")

        updated_user = self._user_repository.update_user(
            user_id,
            email=normalized_email,
            username=normalized_username,
            role=role,
            password_hash=password_hash,
        )
        if updated_user is None:
            raise UserNotFoundError()
        return build_current_user(updated_user)

    def delete_user(self, *, user_id: str) -> None:
        """删除指定用户。"""
        if not self._user_repository.delete_user(user_id):
            raise UserNotFoundError()

    def _normalize_email(self, email: str) -> str:
        """统一规整邮箱格式。"""
        normalized_email = email.strip().lower()
        if not normalized_email:
            raise InvalidUserOperationError("邮箱不能为空")
        return normalized_email

    def _normalize_username(self, username: str) -> str:
        """统一规整用户名。"""
        normalized_username = username.strip()
        if not normalized_username:
            raise InvalidUserOperationError("用户名不能为空")
        return normalized_username

    def _normalize_query_text(self, query_text: str | None) -> str | None:
        """规整可选查询关键词。"""
        if query_text is None:
            return None
        normalized_query = query_text.strip()
        return normalized_query or None
