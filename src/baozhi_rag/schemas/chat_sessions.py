"""聊天会话接口模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from baozhi_rag.schemas.chat import ChatAssistantMessage, ChatTraceItem


class ChatSessionItem(BaseModel):
    """聊天会话条目。"""

    session_id: str = Field(description="会话ID")
    title: str = Field(description="会话标题")
    status: Literal["active", "archived", "deleted"] = Field(description="会话状态")
    message_count: int = Field(description="消息总数")
    summary_version: int = Field(description="摘要版本号")
    last_message_at: datetime | None = Field(default=None, description="最近消息时间")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")


class ChatSessionSummaryItem(BaseModel):
    """历史消息接口需要的会话摘要。"""

    session_id: str = Field(description="会话ID")
    title: str = Field(description="会话标题")
    status: Literal["active", "archived", "deleted"] = Field(description="会话状态")


class CreateChatSessionRequest(BaseModel):
    """创建聊天会话请求。"""

    title: str | None = Field(default=None, max_length=255, description="会话标题")

    @field_validator("title")
    @classmethod
    def strip_optional_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized_title = value.strip()
        if not normalized_title:
            raise ValueError("会话标题不能为空")
        return normalized_title


class UpdateChatSessionRequest(BaseModel):
    """更新聊天会话请求。"""

    title: str | None = Field(default=None, max_length=255, description="会话标题")
    status: Literal["active", "archived"] | None = Field(default=None, description="会话状态")

    @field_validator("title")
    @classmethod
    def strip_optional_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized_title = value.strip()
        if not normalized_title:
            raise ValueError("会话标题不能为空")
        return normalized_title


class ChatSessionDetailResponseData(BaseModel):
    """单个会话详情响应。"""

    session: ChatSessionItem = Field(description="会话详情")


class ChatSessionListResponseData(BaseModel):
    """会话列表响应。"""

    items: list[ChatSessionItem] = Field(description="会话列表")


class ChatHistoryMessageItem(BaseModel):
    """历史消息条目。"""

    message_id: str = Field(description="消息ID")
    session_id: str = Field(description="会话ID")
    sequence_no: int = Field(description="消息序号")
    role: Literal["user", "assistant", "system"] = Field(description="消息角色")
    status: Literal["streaming", "completed", "failed", "aborted"] = Field(description="消息状态")
    plain_text: str = Field(description="消息正文")
    assistant_message: ChatAssistantMessage | None = Field(default=None, description="助手消息详情")
    request_id: str | None = Field(default=None, description="请求ID")
    trace: ChatTraceItem | None = Field(default=None, description="消息追踪信息")
    created_at: datetime = Field(description="创建时间")
    completed_at: datetime | None = Field(default=None, description="完成时间")


class ChatHistoryMessageListResponseData(BaseModel):
    """历史消息响应。"""

    session: ChatSessionSummaryItem = Field(description="会话摘要")
    items: list[ChatHistoryMessageItem] = Field(description="历史消息")
    next_before_sequence_no: int | None = Field(default=None, description="下一页游标")
