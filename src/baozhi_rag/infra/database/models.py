"""数据库模型定义。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):
    """统一把数据库中的时间按 UTC 读写。"""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        """入库前把时间规整到 UTC 无时区格式。"""
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


def mysql_table_options() -> dict[str, str]:
    """返回统一的 MySQL 建表选项。

    返回:
        固定为 `utf8mb4` 与 `utf8mb4_unicode_ci` 的建表参数，用于避免
        新表跟随库默认排序规则创建后，与既有表的字符串主键/外键不兼容。
    """
    return {
        "mysql_engine": "InnoDB",
        "mysql_charset": "utf8mb4",
        "mysql_collate": "utf8mb4_unicode_ci",
    }


class UserModel(Base):
    """用户表。"""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
        UniqueConstraint("username", name="uq_users_username"),
        mysql_table_options(),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class RegistrationVerificationCodeModel(Base):
    """注册邮箱验证码表。"""

    __tablename__ = "registration_verification_codes"
    __table_args__ = (
        Index(
            "ix_registration_verification_codes_email_sent_at",
            "email",
            "sent_at",
        ),
        mysql_table_options(),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    code_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    failed_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sent_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    invalidated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class KnowledgeFileModel(Base):
    """知识文件元数据表。"""

    __tablename__ = "knowledge_files"
    __table_args__ = (
        UniqueConstraint(
            "uploader_user_id",
            "content_sha256",
            name="uq_knowledge_files_uploader_content_sha256",
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
        mysql_table_options(),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    uploader_user_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("users.id"),
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_provider: Mapped[str] = mapped_column(String(32), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    visibility_scope: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    uploaded_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class KnowledgeFileBlobModel(Base):
    """原始上传文件 blob 表。"""

    __tablename__ = "knowledge_file_blobs"
    __table_args__ = (
        UniqueConstraint("raw_sha256", name="uq_knowledge_file_blobs_raw_sha256"),
        mysql_table_options(),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    raw_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_provider: Mapped[str] = mapped_column(String(32), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


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
        mysql_table_options(),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    uploader_user_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("users.id"),
        nullable=False,
    )
    uploader_role: Mapped[str] = mapped_column(String(16), nullable=False)
    raw_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 继续复用历史列名 `blob_key`，避免现网库表在未执行迁移时无法启动；
    # 领域层已把该字段统一解释为本地源文件的 storage_key。
    source_storage_key: Mapped[str] = mapped_column("blob_key", String(512), nullable=False)
    requested_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    ingest_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    file_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    deduplicated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    replaced: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    title_updated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
