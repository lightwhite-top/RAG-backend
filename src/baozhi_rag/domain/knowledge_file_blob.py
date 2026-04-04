"""原始上传文件 blob 领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from baozhi_rag.domain.knowledge_file import FileStorageProvider


@dataclass(frozen=True, slots=True)
class KnowledgeFileBlob:
    """按原始字节流去重的文件 blob 实体。"""

    id: str
    raw_sha256: str
    content_type: str
    size: int
    storage_provider: FileStorageProvider
    storage_key: str
    created_at: datetime
    updated_at: datetime
