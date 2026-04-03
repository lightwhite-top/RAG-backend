"""知识文件领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class FileVisibilityScope(StrEnum):
    """文件可见性范围。"""

    GLOBAL = "global"
    OWNER_ONLY = "owner_only"


class FileStorageProvider(StrEnum):
    """文件持久化存储提供商。"""

    ALIYUN_OSS = "aliyun_oss"


@dataclass(frozen=True, slots=True)
class KnowledgeFile:
    """文件元数据领域实体。"""

    id: str
    uploader_user_id: str
    original_filename: str
    content_type: str
    size: int
    sha256: str
    storage_provider: FileStorageProvider
    storage_key: str
    visibility_scope: FileVisibilityScope
    chunk_count: int
    uploaded_at: datetime
    updated_at: datetime
