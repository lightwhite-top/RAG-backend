"""数据库模型定义。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):
    """统一按 UTC 读写数据库中的时间字段。"""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        """入库前把时间规整为 UTC 无时区格式。"""
        del dialect
        if value is None:
            return None
        normalized = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return normalized.replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        """出库后补回 UTC 时区信息。"""
        del dialect
        if value is None:
            return None
        if value.tzinfo is not None:
            return value.astimezone(UTC)
        return value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    """ORM 基类。"""


def mysql_table_options(comment: str) -> dict[str, str]:
    """返回统一的 MySQL 建表选项。"""
    return {
        "mysql_engine": "InnoDB",
        "mysql_charset": "utf8mb4",
        "mysql_collate": "utf8mb4_unicode_ci",
        "comment": comment,
    }


class ChatSessionModel(Base):
    """聊天会话表。"""

    __tablename__ = "chat_sessions"
    __table_args__ = (
        Index("ix_chat_sessions_owner_updated_at", "owner_user_id", "updated_at"),
        Index(
            "ix_chat_sessions_owner_status_updated_at",
            "owner_user_id",
            "status",
            "updated_at",
        ),
        mysql_table_options("聊天会话表"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="会话ID")
    owner_user_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("users.id"),
        nullable=False,
        comment="所属用户ID",
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, comment="会话标题")
    status: Mapped[str] = mapped_column(String(16), nullable=False, comment="会话状态")
    message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="消息总数",
    )
    summary_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="摘要版本",
    )
    last_message_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        comment="最近消息时间",
    )
    last_user_message_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        comment="最近用户消息时间",
    )
    last_assistant_message_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        comment="最近助手消息时间",
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        comment="删除时间",
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="更新时间",
    )


class ChatMessageModel(Base):
    """聊天消息表。"""

    __tablename__ = "chat_messages"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "sequence_no",
            name="uq_chat_messages_session_sequence",
        ),
        Index("ix_chat_messages_session_created_at", "session_id", "created_at"),
        Index("ix_chat_messages_request_id", "request_id"),
        mysql_table_options("聊天消息表"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="消息ID")
    session_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("chat_sessions.id"),
        nullable=False,
        comment="会话ID",
    )
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False, comment="会话内序号")
    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="消息角色")
    status: Mapped[str] = mapped_column(String(16), nullable=False, comment="消息状态")
    plain_text: Mapped[str] = mapped_column(Text, nullable=False, comment="消息正文")
    content_blocks_json: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="结构化内容块",
    )
    request_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="请求ID",
    )
    model_name: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="模型名称",
    )
    original_query: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="原始问题",
    )
    retrieval_query: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="检索问题",
    )
    rewrite_applied: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="是否执行改写",
    )
    retrieval_size: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="召回数量",
    )
    temperature: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="采样温度",
    )
    finish_reason: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="完成原因",
    )
    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="耗时毫秒",
    )
    usage_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="用量信息",
    )
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="错误码",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="错误消息",
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="更新时间",
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        comment="完成时间",
    )


class ChatMessageCitationModel(Base):
    """聊天消息引用表。"""

    __tablename__ = "chat_message_citations"
    __table_args__ = (
        UniqueConstraint(
            "message_id",
            "citation_index",
            name="uq_chat_message_citations_message_index",
        ),
        Index("ix_chat_message_citations_file_id", "file_id"),
        Index("ix_chat_message_citations_chunk_id", "chunk_id"),
        mysql_table_options("聊天消息引用表"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="引用ID")
    message_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("chat_messages.id"),
        nullable=False,
        comment="消息ID",
    )
    citation_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="引用序号")
    chunk_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="Chunk ID")
    file_id: Mapped[str] = mapped_column(String(32), nullable=False, comment="文件ID")
    source_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="原文件名",
    )
    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="存储键",
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="Chunk序号")
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="字符数")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="引用正文")
    snippet: Mapped[str] = mapped_column(Text, nullable=False, comment="引用摘要")
    merged_terms_json: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="命中领域词",
    )
    score: Mapped[float | None] = mapped_column(Float, nullable=True, comment="检索得分")
    heading_path_json: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="标题路径",
    )
    section_title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="节标题",
    )
    content_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="内容类型",
    )
    source_anchor: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="原文锚点",
    )
    image_assets_json: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="图片资产投影",
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="创建时间",
    )


class ChatSessionMemorySnapshotModel(Base):
    """聊天会话摘要快照表。"""

    __tablename__ = "chat_session_memory_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "version",
            name="uq_chat_session_memory_snapshots_session_version",
        ),
        Index(
            "ix_chat_session_memory_snapshots_session_covered",
            "session_id",
            "covered_until_sequence_no",
        ),
        mysql_table_options("聊天会话摘要快照表"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="快照ID")
    session_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("chat_sessions.id"),
        nullable=False,
        comment="会话ID",
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, comment="摘要版本")
    covered_until_sequence_no: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="覆盖到的消息序号",
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False, comment="摘要正文")
    memory_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="结构化记忆",
    )
    token_estimate: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Token 估算",
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="创建时间",
    )


class UserModel(Base):
    """用户表。"""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
        UniqueConstraint("username", name="uq_users_username"),
        mysql_table_options("用户表"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="用户ID")
    email: Mapped[str] = mapped_column(String(255), nullable=False, comment="邮箱")
    username: Mapped[str] = mapped_column(String(64), nullable=False, comment="用户名")
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="密码哈希",
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="用户角色")
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="更新时间",
    )


class RegistrationVerificationCodeModel(Base):
    """注册邮箱验证码表。"""

    __tablename__ = "registration_verification_codes"
    __table_args__ = (
        Index(
            "ix_registration_verification_codes_email_sent_at",
            "email",
            "sent_at",
        ),
        mysql_table_options("注册邮箱验证码表"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="记录ID")
    email: Mapped[str] = mapped_column(String(255), nullable=False, comment="邮箱")
    code_digest: Mapped[str] = mapped_column(String(64), nullable=False, comment="验证码摘要")
    failed_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="失败校验次数",
    )
    sent_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="发送时间",
    )
    expires_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="过期时间",
    )
    used_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        comment="使用时间",
    )
    invalidated_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        comment="失效时间",
    )


class KnowledgeFileModel(Base):
    """知识文件元数据表。"""

    __tablename__ = "knowledge_files"
    __table_args__ = (
        UniqueConstraint(
            "uploader_user_id",
            "content_sha256",
            name="uq_knowledge_files_uploader_content_sha256",
        ),
        UniqueConstraint(
            "uploader_user_id",
            "text_sha256",
            name="uq_knowledge_files_uploader_text_sha256",
        ),
        Index(
            "ix_knowledge_files_uploader_filename",
            "uploader_user_id",
            "original_filename",
        ),
        Index(
            "ix_knowledge_files_uploader_raw_sha256",
            "uploader_user_id",
            "raw_sha256",
        ),
        Index(
            "ix_knowledge_files_uploader_content_sha256",
            "uploader_user_id",
            "content_sha256",
        ),
        Index(
            "ix_knowledge_files_uploader_text_sha256",
            "uploader_user_id",
            "text_sha256",
        ),
        mysql_table_options("知识文件元数据表"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="文件ID")
    uploader_user_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("users.id"),
        nullable=False,
        comment="上传用户ID",
    )
    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="原始文件名",
    )
    content_type: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="文件MIME类型",
    )
    size: Mapped[int] = mapped_column(Integer, nullable=False, comment="文件大小（字节）")
    raw_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="原始文件SHA256",
    )
    text_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="正文稳定SHA256",
    )
    content_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="规范化内容SHA256",
    )
    storage_provider: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="存储提供方",
    )
    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="存储对象键",
    )
    visibility_scope: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="可见范围",
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="切片数量",
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="上传时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="更新时间",
    )


class KnowledgeFileImageAssetModel(Base):
    """知识文件图片资产表。"""

    __tablename__ = "knowledge_file_image_assets"
    __table_args__ = (
        UniqueConstraint(
            "file_id",
            "source_anchor",
            name="uq_knowledge_file_image_assets_file_anchor",
        ),
        Index("ix_knowledge_file_image_assets_semantic_chunk_id", "semantic_chunk_id"),
        Index(
            "ix_knowledge_file_image_assets_file_segment",
            "file_id",
            "segment_id",
        ),
        Index(
            "ix_knowledge_file_image_assets_uploader_normalized_sha256",
            "uploader_user_id",
            "normalized_image_sha256",
        ),
        mysql_table_options("知识文件图片资产表"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="图片资产ID")
    file_id: Mapped[str] = mapped_column(String(32), nullable=False, comment="所属文件ID")
    segment_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="所属原始片段ID")
    semantic_chunk_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="对应图片语义Chunk ID",
    )
    asset_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="Chunk内图片序号")
    uploader_user_id: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="上传用户ID",
    )
    source_anchor: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="原文锚点",
    )
    image_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="原始图片SHA256",
    )
    normalized_image_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="归一化图片SHA256",
    )
    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="原图存储对象键",
    )
    thumbnail_storage_key: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="缩略图存储对象键",
    )
    content_type: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="图片MIME类型",
    )
    width: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="图片宽度")
    height: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="图片高度")
    ocr_text: Mapped[str] = mapped_column(Text, nullable=False, comment="图片OCR文本")
    summary: Mapped[str] = mapped_column(Text, nullable=False, comment="图片语义摘要")
    image_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="图片类型",
    )
    recognition_model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="图片识别模型名称",
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="更新时间",
    )


class KnowledgeFileBlobModel(Base):
    """原始上传文件 blob 表。"""

    __tablename__ = "knowledge_file_blobs"
    __table_args__ = (
        UniqueConstraint("raw_sha256", name="uq_knowledge_file_blobs_raw_sha256"),
        mysql_table_options("原始上传文件Blob表"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="Blob记录ID")
    raw_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="原始文件SHA256",
    )
    content_type: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="文件MIME类型",
    )
    size: Mapped[int] = mapped_column(Integer, nullable=False, comment="文件大小（字节）")
    storage_provider: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="存储提供方",
    )
    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="存储对象键",
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="更新时间",
    )


class KnowledgeUploadTaskModel(Base):
    """知识文件上传任务表。"""

    __tablename__ = "knowledge_upload_tasks"
    __table_args__ = (
        UniqueConstraint(
            "uploader_user_id",
            "raw_sha256",
            "ingest_version",
            name="uq_knowledge_upload_tasks_uploader_raw_ingest",
        ),
        Index(
            "ix_knowledge_upload_tasks_status_lease",
            "status",
            "lease_expires_at",
        ),
        Index(
            "ix_knowledge_upload_tasks_uploader_created_at",
            "uploader_user_id",
            "created_at",
        ),
        mysql_table_options("知识文件上传任务表"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="任务ID")
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="请求ID")
    uploader_user_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("users.id"),
        nullable=False,
        comment="上传用户ID",
    )
    uploader_role: Mapped[str] = mapped_column(String(16), nullable=False, comment="上传用户角色")
    raw_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="原始文件SHA256",
    )
    content_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="规范化内容SHA256",
    )
    # 继续复用历史列名 `blob_key`，避免现网库表在未执行迁移时无法启动。
    # 领域层已把该字段统一解释为本地源文件的 storage_key。
    source_storage_key: Mapped[str] = mapped_column(
        "blob_key",
        String(512),
        nullable=False,
        comment="源文件存储对象键",
    )
    requested_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="请求文件名",
    )
    content_type: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="文件MIME类型",
    )
    size: Mapped[int] = mapped_column(Integer, nullable=False, comment="文件大小（字节）")
    ingest_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="导入版本",
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, comment="任务状态")
    stage: Mapped[str] = mapped_column(String(32), nullable=False, comment="处理阶段")
    file_id: Mapped[str | None] = mapped_column(String(32), nullable=True, comment="生成文件ID")
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="切片数量",
    )
    deduplicated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="是否命中去重",
    )
    replaced: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="是否替换已有文件",
    )
    title_updated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="是否更新标题",
    )
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="错误码",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="错误信息",
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="重试次数",
    )
    worker_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="处理工作节点ID",
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        comment="租约过期时间",
    )
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        comment="最后心跳时间",
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        comment="更新时间",
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        comment="完成时间",
    )
