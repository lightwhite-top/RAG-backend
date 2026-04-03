"""基于 SQLAlchemy 的用户仓储实现。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from baozhi_rag.domain.user import User, UserListPage, UserRole
from baozhi_rag.domain.user_errors import EmailAlreadyExistsError, UsernameAlreadyExistsError
from baozhi_rag.infra.database.models import UserModel


class SqlAlchemyUserRepository:
    """用户仓储的 SQLAlchemy 实现。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """初始化仓储。"""
        self._session_factory = session_factory

    def create_user(
        self,
        *,
        email: str,
        username: str,
        password_hash: str,
        role: UserRole,
    ) -> User:
        """创建用户。"""
        user_model = UserModel(
            id=uuid4().hex,
            email=email,
            username=username,
            password_hash=password_hash,
            role=role.value,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        with self._session_factory() as session:
            session.add(user_model)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                self._raise_conflict_error(exc)
            session.refresh(user_model)
            return self._to_domain(user_model)

    def get_user_by_id(self, user_id: str) -> User | None:
        """按 ID 查询用户。"""
        with self._session_factory() as session:
            user_model = session.get(UserModel, user_id)
            return self._to_domain(user_model) if user_model is not None else None

    def get_user_by_email(self, email: str) -> User | None:
        """按邮箱查询用户。"""
        with self._session_factory() as session:
            stmt = select(UserModel).where(UserModel.email == email)
            user_model = session.scalar(stmt)
            return self._to_domain(user_model) if user_model is not None else None

    def list_users(
        self,
        *,
        query_text: str | None,
        page: int,
        page_size: int,
    ) -> UserListPage:
        """分页查询用户。"""
        with self._session_factory() as session:
            base_stmt = select(UserModel)
            count_stmt = select(func.count()).select_from(UserModel)
            if query_text:
                like_pattern = f"%{query_text}%"
                search_predicate = or_(
                    UserModel.email.like(like_pattern),
                    UserModel.username.like(like_pattern),
                )
                base_stmt = base_stmt.where(search_predicate)
                count_stmt = count_stmt.where(search_predicate)

            paged_stmt = (
                base_stmt.order_by(UserModel.created_at.desc(), UserModel.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            items = [self._to_domain(item) for item in session.scalars(paged_stmt).all()]
            total = int(session.scalar(count_stmt) or 0)
            return UserListPage(items=items, total=total, page=page, page_size=page_size)

    def update_user(
        self,
        user_id: str,
        *,
        email: str | None = None,
        username: str | None = None,
        role: UserRole | None = None,
        password_hash: str | None = None,
    ) -> User | None:
        """更新用户信息。"""
        with self._session_factory() as session:
            user_model = session.get(UserModel, user_id)
            if user_model is None:
                return None

            if email is not None:
                user_model.email = email
            if username is not None:
                user_model.username = username
            if role is not None:
                user_model.role = role.value
            if password_hash is not None:
                user_model.password_hash = password_hash
            user_model.updated_at = datetime.now(UTC)

            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                self._raise_conflict_error(exc)
            session.refresh(user_model)
            return self._to_domain(user_model)

    def delete_user(self, user_id: str) -> bool:
        """删除用户。"""
        with self._session_factory() as session:
            user_model = session.get(UserModel, user_id)
            if user_model is None:
                return False
            session.delete(user_model)
            session.commit()
            return True

    def _raise_conflict_error(self, exc: IntegrityError) -> None:
        """根据唯一索引冲突名称转换为稳定业务错误。"""
        message = str(exc.orig).lower()
        if "uq_users_email" in message or "email" in message:
            raise EmailAlreadyExistsError() from exc
        if "uq_users_username" in message or "username" in message:
            raise UsernameAlreadyExistsError() from exc
        raise

    def _to_domain(self, user_model: UserModel) -> User:
        """把 ORM 模型转换为领域实体。"""
        return User(
            id=user_model.id,
            email=user_model.email,
            username=user_model.username,
            password_hash=user_model.password_hash,
            role=UserRole(user_model.role),
            created_at=user_model.created_at,
            updated_at=user_model.updated_at,
        )
