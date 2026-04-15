"""聊天接口模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from baozhi_rag.schemas.common import NormalizedExtension


class ChatImageAssetItem(BaseModel):
    """聊天引用与正文块中的图片资产。"""

    segment_id: str | None = Field(default=None, description="所属原始片段ID")
    asset_id: str = Field(description="图片资产唯一标识")
    source_anchor: str | None = Field(default=None, description="原文锚点")
    storage_key: str = Field(description="原图存储对象键")
    thumbnail_storage_key: str | None = Field(default=None, description="缩略图存储对象键")
    image_url: str | None = Field(default=None, description="原图访问地址")
    thumbnail_url: str | None = Field(default=None, description="缩略图访问地址")
    image_type: str = Field(default="", description="图片类型")
    summary: str = Field(default="", description="图片语义摘要")
    ocr_text: str = Field(default="", description="图片OCR文本")


class ChatBlockAssetItem(BaseModel):
    """结构化正文块中的统一资产。"""

    asset_id: str = Field(description="资产唯一标识")
    display_name: str = Field(description="资产展示名称")
    storage_key: str = Field(description="资产原始存储对象键")
    content_type: str | None = Field(default=None, description="资产 MIME 类型")
    extension: NormalizedExtension = Field(
        description="标准化资产扩展名，不含点，例如 png、pdf；无法识别时为 null",
        title="资产扩展名",
    )
    size: int | None = Field(default=None, description="资产大小（字节）")
    url: str | None = Field(default=None, description="资产访问地址")
    preview_url: str | None = Field(default=None, description="资产预览地址")
    expires_at: str | None = Field(default=None, description="资产地址过期时间")
    source_anchor: str | None = Field(default=None, description="原文锚点")
    summary: str | None = Field(default=None, description="资产摘要")
    ocr_text: str | None = Field(default=None, description="资产 OCR 文本")


class ChatMessageItem(BaseModel):
    """单条聊天消息。"""

    role: Literal["system", "user", "assistant"] = Field(description="消息角色")
    content: str = Field(min_length=1, description="消息内容")


class ChatCompletionRequest(BaseModel):
    """聊天补全请求体。"""

    session_id: str | None = Field(default=None, description="会话 ID；为空时按无状态模式处理")
    messages: list[ChatMessageItem] = Field(
        min_length=1,
        description="会话消息列表，至少包含一条 user 消息",
    )
    stream: bool = Field(default=False, description="是否启用 SSE 流式返回")
    retrieval_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="兼容字段：客户端可继续传入，但当前由后端策略决定实际检索数量",
    )
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description="兼容字段：客户端可继续传入，但当前由后端策略决定实际采样温度",
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
    file_url: str | None = Field(default=None, description="文件访问地址")
    file_content_type: str | None = Field(default=None, description="文件 MIME 类型")
    extension: NormalizedExtension = Field(
        description="标准化文件扩展名，不含点，例如 pdf、docx；无法识别时为 null",
        title="文件扩展名",
    )
    size: int | None = Field(default=None, description="文件大小（字节）")
    expires_at: str | None = Field(default=None, description="文件地址过期时间")
    image_assets: list[ChatImageAssetItem] = Field(
        default_factory=list,
        description="关联图片资产",
    )


class ChatContentBlockItem(BaseModel):
    """结构化正文块。"""

    block_id: str = Field(description="正文块唯一标识")
    block_type: Literal["markdown", "notice", "image_gallery", "source_file"] = Field(
        description="正文块类型"
    )
    text: str = Field(description="正文块文本")
    citation_ids: list[str] = Field(default_factory=list, description="关联引用标识列表")
    sequence: int = Field(description="正文块顺序")
    files_assets: list[ChatBlockAssetItem] = Field(
        default_factory=list,
        description="正文块内联资产",
    )


ChatStreamDeltaType = Literal["append", "insert", "replace", "complete", "content_block"]


class ChatStreamDeltaItem(BaseModel):
    """流式正文增量协议。"""

    message_id: str = Field(description="消息唯一标识")
    request_id: str = Field(description="请求链路编号")
    seq: int = Field(description="消息内全局递增事件序号")
    delta_type: ChatStreamDeltaType = Field(description="增量类型")
    offset: int | None = Field(default=None, description="文本增量追加前的偏移量")
    after_block_id: str | None = Field(default=None, description="新块插入的锚点块 ID")
    block: ChatContentBlockItem | None = Field(default=None, description="增量关联正文块")


class ChatAssistantMessage(BaseModel):
    """结构化助手消息。"""

    message_id: str = Field(description="消息唯一标识")
    role: Literal["assistant"] = Field(default="assistant", description="消息角色")
    session_id: str | None = Field(default=None, description="所属会话 ID")
    sequence_no: int | None = Field(default=None, description="会话内消息序号")
    plain_text: str = Field(description="助手回答纯文本")
    content_blocks: list[ChatContentBlockItem] = Field(description="结构化正文块")
    citations: list[ChatCitationItem] = Field(description="结构化引用列表")
    finish_reason: str = Field(description="消息完成原因")
    created_at: datetime | None = Field(default=None, description="消息创建时间")
    completed_at: datetime | None = Field(default=None, description="消息完成时间")


class ChatTraceItem(BaseModel):
    """聊天链路追踪信息。"""

    request_id: str = Field(description="请求链路编号")
    original_query: str = Field(description="用户原始问题")
    retrieval_query: str = Field(description="实际用于检索的查询文本")
    rewrite_applied: bool = Field(description="是否执行了查询改写")
    query_intent: str | None = Field(default=None, description="查询意图")
    retrieval_mode: str | None = Field(default=None, description="检索模式")
    lane_count: int | None = Field(default=None, description="检索 lane 数量")
    final_hit_count: int | None = Field(default=None, description="最终命中数量")
    evidence_sufficient: bool | None = Field(default=None, description="证据是否充分")
    evidence_reason: str | None = Field(default=None, description="证据判断原因")
    deep_rerank_triggered: bool | None = Field(default=None, description="是否触发深度重排")
    applied_retrieval_size: int | None = Field(
        default=None,
        description="本轮后端实际使用的检索结果数量",
    )
    applied_temperature: float | None = Field(
        default=None,
        description="本轮后端实际使用的采样温度",
    )
    lanes: list[dict[str, Any]] | None = Field(default=None, description="各检索 lane 摘要")
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
