"""基于 SQLAlchemy 的注册验证码仓储实现。"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from baozhi_rag.domain.registration_verification import RegistrationVerificationCode
from baozhi_rag.infra.database.models import RegistrationVerificationCodeModel


class SqlAlchemyRegistrationVerificationRepository:
    """注册验证码仓储的 SQLAlchemy 实现。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """初始化仓储。

        参数:
            session_factory: SQLAlchemy 会话工厂。

        返回:
            None。
        """
        self._session_factory = session_factory

    def create_code(
        self,
        *,
        email: str,
        code_digest: str,
        sent_at: datetime,
        expires_at: datetime,
    ) -> RegistrationVerificationCode:
        """创建新的注册验证码记录。"""
        code_model = RegistrationVerificationCodeModel(
            id=uuid4().hex,
            email=email,
            code_digest=code_digest,
            failed_attempts=0,
            sent_at=sent_at,
            expires_at=expires_at,
        )
        with self._session_factory() as session:
            session.add(code_model)
            session.commit()
            session.refresh(code_model)
            return self._to_domain(code_model)

    def get_latest_code(self, email: str) -> RegistrationVerificationCode | None:
        """按邮箱获取最新验证码记录。"""
        stmt = (
            select(RegistrationVerificationCodeModel)
            .where(RegistrationVerificationCodeModel.email == email)
            .order_by(
                RegistrationVerificationCodeModel.sent_at.desc(),
                RegistrationVerificationCodeModel.id.desc(),
            )
            .limit(1)
        )
        with self._session_factory() as session:
            code_model = session.scalar(stmt)
            return self._to_domain(code_model) if code_model is not None else None

    def invalidate_active_codes(self, *, email: str, invalidated_at: datetime) -> int:
        """把指定邮箱下所有未完成的旧验证码置为失效。"""
        stmt = select(RegistrationVerificationCodeModel).where(
            RegistrationVerificationCodeModel.email == email,
            RegistrationVerificationCodeModel.used_at.is_(None),
            RegistrationVerificationCodeModel.invalidated_at.is_(None),
        )
        with self._session_factory() as session:
            code_models = session.scalars(stmt).all()
            for code_model in code_models:
                code_model.invalidated_at = invalidated_at
            session.commit()
            return len(code_models)

    def invalidate_code(
        self,
        code_id: str,
        *,
        invalidated_at: datetime,
    ) -> RegistrationVerificationCode | None:
        """把指定验证码记录置为失效。"""
        with self._session_factory() as session:
            code_model = session.get(RegistrationVerificationCodeModel, code_id)
            if code_model is None:
                return None
            code_model.invalidated_at = invalidated_at
            session.commit()
            session.refresh(code_model)
            return self._to_domain(code_model)

    def increment_failed_attempts(
        self,
        code_id: str,
    ) -> RegistrationVerificationCode | None:
        """把指定验证码记录的失败次数加一。"""
        with self._session_factory() as session:
            code_model = session.get(RegistrationVerificationCodeModel, code_id)
            if code_model is None:
                return None
            code_model.failed_attempts += 1
            session.commit()
            session.refresh(code_model)
            return self._to_domain(code_model)

    def mark_used(
        self,
        code_id: str,
        *,
        used_at: datetime,
    ) -> RegistrationVerificationCode | None:
        """把指定验证码记录标记为已使用。"""
        with self._session_factory() as session:
            code_model = session.get(RegistrationVerificationCodeModel, code_id)
            if code_model is None:
                return None
            code_model.used_at = used_at
            session.commit()
            session.refresh(code_model)
            return self._to_domain(code_model)

    def _to_domain(
        self,
        code_model: RegistrationVerificationCodeModel,
    ) -> RegistrationVerificationCode:
        """把 ORM 模型转换为领域实体。"""
        return RegistrationVerificationCode(
            id=code_model.id,
            email=code_model.email,
            code_digest=code_model.code_digest,
            failed_attempts=code_model.failed_attempts,
            sent_at=code_model.sent_at,
            expires_at=code_model.expires_at,
            used_at=code_model.used_at,
            invalidated_at=code_model.invalidated_at,
        )
