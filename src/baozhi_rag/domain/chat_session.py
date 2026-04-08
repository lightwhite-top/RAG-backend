"""聊天会话领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class ChatSessionStatus(StrEnum):
    """聊天会话状态。"""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class ChatSession:
    """聊天会话实体。"""

    id: str
    owner_user_id: str
    title: str
    status: ChatSessionStatus
    message_count: int
    summary_version: int
    last_message_at: datetime | None
    last_user_message_at: datetime | None
    last_assistant_message_at: datetime | None
    deleted_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ChatSessionListPage:
    """聊天会话分页结果。"""

    items: list[ChatSession]
    total: int
    page: int
    page_size: int
