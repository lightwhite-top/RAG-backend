"""聊天接口模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessageItem(BaseModel):
    """单条聊天消息。"""

    role: Literal["system", "user", "assistant"] = Field(description="消息角色")
    content: str = Field(min_length=1, description="消息内容")


class ChatCompletionRequest(BaseModel):
    """聊天补全请求体。"""

    messages: list[ChatMessageItem] = Field(
        min_length=1,
        description="会话消息列表，至少包含一条 user 消息",
    )
    stream: bool = Field(default=False, description="是否启用 SSE 流式返回")
    retrieval_size: int = Field(default=5, ge=1, le=20, description="检索召回数量")
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description="可选采样温度",
    )


class ChatCitationItem(BaseModel):
    """聊天回答引用的证据片段。"""

    chunk_id: str = Field(description="chunk 唯一标识")
    file_id: str = Field(description="文件唯一标识")
    source_filename: str = Field(description="原始文件名")
    storage_key: str = Field(description="相对存储路径")
    chunk_index: int = Field(description="chunk 序号")
    char_count: int = Field(description="chunk 字符数")
    content: str = Field(description="chunk 正文")
    merged_terms: list[str] = Field(description="命中的领域词项")
    score: float | None = Field(description="检索得分")


class ChatCompletionResponse(BaseModel):
    """聊天补全响应体。"""

    answer: str = Field(description="最终回答")
    retrieval_query: str = Field(description="实际用于检索的查询文本")
    citation_count: int = Field(description="证据数量")
    citations: list[ChatCitationItem] = Field(description="证据列表")
    finish_reason: str = Field(description="完成原因")
