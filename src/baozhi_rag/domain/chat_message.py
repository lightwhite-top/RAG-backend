"""聊天消息领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class ChatMessageRole(StrEnum):
    """聊天消息角色。"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessageStatus(StrEnum):
    """聊天消息状态。"""

    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class ChatMessageCitationRecord:
    """聊天消息引用记录。"""

    id: str
    message_id: str
    citation_index: int
    chunk_id: str
    file_id: str
    source_filename: str
    storage_key: str
    chunk_index: int
    char_count: int
    content: str
    snippet: str
    merged_terms: list[str]
    score: float | None
    heading_path: list[str]
    section_title: str | None
    content_type: str
    source_anchor: str | None
    created_at: datetime
    image_assets: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ChatMessageRecord:
    """聊天消息记录。"""

    id: str
    session_id: str
    sequence_no: int
    role: ChatMessageRole
    status: ChatMessageStatus
    plain_text: str
    content_blocks: list[dict[str, Any]]
    request_id: str | None
    model_name: str | None
    original_query: str | None
    retrieval_query: str | None
    rewrite_applied: bool
    retrieval_size: int | None
    temperature: float | None
    finish_reason: str | None
    latency_ms: int | None
    usage: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    citations: list[ChatMessageCitationRecord] = field(default_factory=list)
