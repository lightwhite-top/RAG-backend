"""聊天会话仓储抽象。"""

from __future__ import annotations

from typing import Protocol

from baozhi_rag.domain.chat_session import ChatSession, ChatSessionListPage, ChatSessionStatus


class ChatSessionRepository(Protocol):
    """聊天会话仓储协议。"""

    def create_session(self, session: ChatSession) -> ChatSession:
        """创建聊天会话。"""
        ...

    def get_session_by_id(self, session_id: str) -> ChatSession | None:
        """按 ID 查询会话。"""
        ...

    def list_sessions(
        self,
        *,
        owner_user_id: str,
        page: int,
        page_size: int,
        status: ChatSessionStatus | None,
    ) -> ChatSessionListPage:
        """分页查询会话。"""
        ...

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        status: ChatSessionStatus | None = None,
    ) -> ChatSession | None:
        """更新会话。"""
        ...
