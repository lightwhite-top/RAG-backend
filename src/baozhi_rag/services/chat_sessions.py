"""聊天会话管理服务。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from baozhi_rag.domain.chat_errors import (
    ChatSessionDeletedError,
    ChatSessionForbiddenError,
    ChatSessionNotFoundError,
    ChatSessionValidationError,
)
from baozhi_rag.domain.chat_message import ChatMessageRecord
from baozhi_rag.domain.chat_message_repository import ChatMessageRepository
from baozhi_rag.domain.chat_session import ChatSession, ChatSessionListPage, ChatSessionStatus
from baozhi_rag.domain.chat_session_repository import ChatSessionRepository
from baozhi_rag.domain.user import CurrentUser


@dataclass(frozen=True, slots=True)
class ChatSessionMessageHistoryResult:
    """会话历史消息结果。"""

    session: ChatSession
    items: list[ChatMessageRecord]
    next_before_sequence_no: int | None


class ChatSessionService:
    """聊天会话管理服务。"""

    _DEFAULT_TITLE = "新建会话"

    def __init__(
        self,
        *,
        session_repository: ChatSessionRepository,
        message_repository: ChatMessageRepository,
    ) -> None:
        self._session_repository = session_repository
        self._message_repository = message_repository

    def create_session(
        self,
        *,
        current_user: CurrentUser,
        title: str | None,
    ) -> ChatSession:
        now = datetime.now(UTC)
        normalized_title = self._normalize_optional_title(title) or self._DEFAULT_TITLE
        session = ChatSession(
            id=uuid4().hex,
            owner_user_id=current_user.id,
            title=normalized_title,
            status=ChatSessionStatus.ACTIVE,
            message_count=0,
            summary_version=0,
            last_message_at=None,
            last_user_message_at=None,
            last_assistant_message_at=None,
            deleted_at=None,
            created_at=now,
            updated_at=now,
        )
        return self._session_repository.create_session(session)

    def list_sessions(
        self,
        *,
        current_user: CurrentUser,
        page: int,
        page_size: int,
        status: ChatSessionStatus | None,
    ) -> ChatSessionListPage:
        return self._session_repository.list_sessions(
            owner_user_id=current_user.id,
            page=page,
            page_size=page_size,
            status=status,
        )

    def get_session(
        self,
        *,
        session_id: str,
        current_user: CurrentUser,
    ) -> ChatSession:
        return self._require_accessible_session(session_id=session_id, current_user=current_user)

    def update_session(
        self,
        *,
        session_id: str,
        current_user: CurrentUser,
        title: str | None = None,
        status: ChatSessionStatus | None = None,
    ) -> ChatSession:
        session = self._require_accessible_session(session_id=session_id, current_user=current_user)
        normalized_title = self._normalize_optional_title(title)
        updated_session = self._session_repository.update_session(
            session.id,
            title=normalized_title,
            status=status,
        )
        if updated_session is None:
            raise ChatSessionNotFoundError()
        return updated_session

    def delete_session(
        self,
        *,
        session_id: str,
        current_user: CurrentUser,
    ) -> None:
        session = self._require_accessible_session(session_id=session_id, current_user=current_user)
        deleted_session = self._session_repository.update_session(
            session.id,
            status=ChatSessionStatus.DELETED,
        )
        if deleted_session is None:
            raise ChatSessionNotFoundError()

    def list_messages(
        self,
        *,
        session_id: str,
        current_user: CurrentUser,
        before_sequence_no: int | None,
        limit: int,
    ) -> ChatSessionMessageHistoryResult:
        session = self._require_accessible_session(session_id=session_id, current_user=current_user)
        messages = self._message_repository.list_messages(
            session_id=session.id,
            before_sequence_no=before_sequence_no,
            limit=limit + 1,
        )
        next_before_sequence_no: int | None = None
        if len(messages) > limit:
            next_before_sequence_no = messages[0].sequence_no
            messages = messages[1:]

        return ChatSessionMessageHistoryResult(
            session=session,
            items=messages,
            next_before_sequence_no=next_before_sequence_no,
        )

    def _require_accessible_session(
        self,
        *,
        session_id: str,
        current_user: CurrentUser,
    ) -> ChatSession:
        session = self._session_repository.get_session_by_id(session_id)
        if session is None:
            raise ChatSessionNotFoundError()
        if session.owner_user_id != current_user.id:
            raise ChatSessionForbiddenError()
        if session.status is ChatSessionStatus.DELETED:
            raise ChatSessionDeletedError()
        return session

    def _normalize_optional_title(self, title: str | None) -> str | None:
        if title is None:
            return None
        normalized_title = title.strip()
        if not normalized_title:
            raise ChatSessionValidationError("会话标题不能为空")
        return normalized_title
