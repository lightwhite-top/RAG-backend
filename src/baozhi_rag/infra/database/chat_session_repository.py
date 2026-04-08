"""基于 SQLAlchemy 的聊天会话仓储实现。"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from baozhi_rag.domain.chat_session import ChatSession, ChatSessionListPage, ChatSessionStatus
from baozhi_rag.infra.database.models import ChatSessionModel


class SqlAlchemyChatSessionRepository:
    """聊天会话仓储的 SQLAlchemy 实现。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create_session(self, session: ChatSession) -> ChatSession:
        session_model = self._to_model(session)
        with self._session_factory() as db_session:
            db_session.add(session_model)
            db_session.commit()
            db_session.refresh(session_model)
            return self._to_domain(session_model)

    def get_session_by_id(self, session_id: str) -> ChatSession | None:
        with self._session_factory() as db_session:
            session_model = db_session.get(ChatSessionModel, session_id)
            return self._to_domain(session_model) if session_model is not None else None

    def list_sessions(
        self,
        *,
        owner_user_id: str,
        page: int,
        page_size: int,
        status: ChatSessionStatus | None,
    ) -> ChatSessionListPage:
        predicates: list[ColumnElement[bool]] = [ChatSessionModel.owner_user_id == owner_user_id]
        if status is None:
            predicates.append(ChatSessionModel.status != ChatSessionStatus.DELETED.value)
        else:
            predicates.append(ChatSessionModel.status == status.value)

        with self._session_factory() as db_session:
            base_stmt = select(ChatSessionModel).where(*predicates)
            count_stmt = select(func.count()).select_from(ChatSessionModel).where(*predicates)
            paged_stmt = (
                base_stmt.order_by(ChatSessionModel.updated_at.desc(), ChatSessionModel.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            items = [self._to_domain(item) for item in db_session.scalars(paged_stmt).all()]
            total = int(db_session.scalar(count_stmt) or 0)
            return ChatSessionListPage(items=items, total=total, page=page, page_size=page_size)

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        status: ChatSessionStatus | None = None,
    ) -> ChatSession | None:
        with self._session_factory() as db_session:
            session_model = db_session.get(ChatSessionModel, session_id)
            if session_model is None:
                return None

            now = datetime.now(UTC)
            if title is not None:
                session_model.title = title
            if status is not None:
                session_model.status = status.value
                session_model.deleted_at = now if status is ChatSessionStatus.DELETED else None
            session_model.updated_at = now
            db_session.commit()
            db_session.refresh(session_model)
            return self._to_domain(session_model)

    def _to_model(self, session: ChatSession) -> ChatSessionModel:
        return ChatSessionModel(
            id=session.id,
            owner_user_id=session.owner_user_id,
            title=session.title,
            status=session.status.value,
            message_count=session.message_count,
            summary_version=session.summary_version,
            last_message_at=session.last_message_at,
            last_user_message_at=session.last_user_message_at,
            last_assistant_message_at=session.last_assistant_message_at,
            deleted_at=session.deleted_at,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    def _to_domain(self, session_model: ChatSessionModel) -> ChatSession:
        return ChatSession(
            id=session_model.id,
            owner_user_id=session_model.owner_user_id,
            title=session_model.title,
            status=ChatSessionStatus(session_model.status),
            message_count=session_model.message_count,
            summary_version=session_model.summary_version,
            last_message_at=session_model.last_message_at,
            last_user_message_at=session_model.last_user_message_at,
            last_assistant_message_at=session_model.last_assistant_message_at,
            deleted_at=session_model.deleted_at,
            created_at=session_model.created_at,
            updated_at=session_model.updated_at,
        )
