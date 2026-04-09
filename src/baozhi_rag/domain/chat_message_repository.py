"""聊天消息仓储抽象。"""

from __future__ import annotations

from typing import Any, Protocol

from baozhi_rag.domain.chat_message import (
    ChatMessageCitationRecord,
    ChatMessageRecord,
    ChatMessageRole,
    ChatMessageStatus,
)


class ChatMessageRepository(Protocol):
    """聊天消息仓储协议。"""

    def append_message(
        self,
        *,
        session_id: str,
        role: ChatMessageRole,
        status: ChatMessageStatus,
        plain_text: str,
        content_blocks: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        model_name: str | None = None,
        original_query: str | None = None,
        retrieval_query: str | None = None,
        rewrite_applied: bool = False,
        retrieval_size: int | None = None,
        temperature: float | None = None,
        finish_reason: str | None = None,
        latency_ms: int | None = None,
        usage: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> ChatMessageRecord:
        """向会话末尾追加一条消息。"""
        ...

    def update_message(
        self,
        message_id: str,
        *,
        status: ChatMessageStatus | None = None,
        plain_text: str | None = None,
        content_blocks: list[dict[str, Any]] | None = None,
        model_name: str | None = None,
        original_query: str | None = None,
        retrieval_query: str | None = None,
        rewrite_applied: bool | None = None,
        retrieval_size: int | None = None,
        temperature: float | None = None,
        finish_reason: str | None = None,
        latency_ms: int | None = None,
        usage: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        completed: bool = False,
    ) -> ChatMessageRecord | None:
        """更新消息。"""
        ...

    def replace_citations(
        self,
        message_id: str,
        citations: list[ChatMessageCitationRecord],
    ) -> list[ChatMessageCitationRecord]:
        """替换消息引用。"""
        ...

    def list_messages(
        self,
        *,
        session_id: str,
        before_sequence_no: int | None,
        limit: int,
    ) -> list[ChatMessageRecord]:
        """按游标查询历史消息。"""
        ...

    def get_recent_messages(
        self,
        *,
        session_id: str,
        limit: int,
    ) -> list[ChatMessageRecord]:
        """获取最近消息窗口。"""
        ...
