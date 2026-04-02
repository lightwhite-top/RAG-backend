"""聊天接口模型。"""

from __future__ import annotations

from typing import Any, Literal

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

    id: str = Field(description="回答内稳定的引用标识")
    chunk_id: str = Field(description="chunk 唯一标识")
    file_id: str = Field(description="文件唯一标识")
    source_filename: str = Field(description="原始文件名")
    storage_key: str = Field(description="相对存储路径")
    chunk_index: int = Field(description="chunk 序号")
    char_count: int = Field(description="chunk 字符数")
    content: str = Field(description="chunk 正文")
    snippet: str = Field(description="用于前端展示的证据摘要")
    merged_terms: list[str] = Field(description="命中的领域词项")
    score: float | None = Field(description="检索得分")
    heading_path: list[str] = Field(default_factory=list, description="所属章节路径")
    section_title: str | None = Field(default=None, description="所属末级章节标题")
    content_type: Literal["paragraph", "table"] = Field(
        default="paragraph",
        description="证据内容类型",
    )
    source_anchor: str | None = Field(default=None, description="原文定位锚点")


class ChatContentBlockItem(BaseModel):
    """结构化正文块。"""

    block_id: str = Field(description="正文块唯一标识")
    block_type: Literal["markdown", "notice"] = Field(description="正文块类型")
    text: str = Field(description="正文块文本")
    citation_ids: list[str] = Field(default_factory=list, description="关联引用标识列表")
    sequence: int = Field(description="正文块顺序")


class ChatAssistantMessage(BaseModel):
    """结构化助手消息。"""

    message_id: str = Field(description="消息唯一标识")
    role: Literal["assistant"] = Field(default="assistant", description="消息角色")
    plain_text: str = Field(description="助手回答纯文本")
    content_blocks: list[ChatContentBlockItem] = Field(description="结构化正文块")
    citations: list[ChatCitationItem] = Field(description="结构化引用列表")
    finish_reason: str = Field(description="消息完成原因")


class ChatTraceItem(BaseModel):
    """聊天链路追踪信息。"""

    request_id: str = Field(description="请求链路编号")
    original_query: str = Field(description="用户原始问题")
    retrieval_query: str = Field(description="实际用于检索的查询文本")
    rewrite_applied: bool = Field(description="是否执行了查询改写")
    model: str | None = Field(default=None, description="实际使用的模型名称")
    usage: dict[str, Any] | None = Field(default=None, description="模型调用资源消耗")
    latency_ms: int | None = Field(default=None, description="本次请求耗时（毫秒）")


class ChatCompletionResponse(BaseModel):
    """聊天补全响应体。

    说明:
        `assistant_message` 与 `trace` 是前端应优先消费的新结构；
        `answer`、`citations`、`finish_reason` 等旧字段保留一段兼容期。
    """

    assistant_message: ChatAssistantMessage = Field(description="结构化助手消息")
    trace: ChatTraceItem = Field(description="聊天链路追踪信息")
    answer: str = Field(description="最终回答（兼容字段）")
    retrieval_query: str = Field(description="实际用于检索的查询文本（兼容字段）")
    citation_count: int = Field(description="证据数量（兼容字段）")
    citations: list[ChatCitationItem] = Field(description="证据列表（兼容字段）")
    finish_reason: str = Field(description="完成原因（兼容字段）")
