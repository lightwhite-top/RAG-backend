"""知识文件异步上传任务领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class KnowledgeUploadTaskStatus(StrEnum):
    """上传任务状态。"""

    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class KnowledgeUploadTaskStage(StrEnum):
    """上传任务阶段。"""

    UPLOADED = "uploaded"
    PARSING = "parsing"
    INDEXING = "indexing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class KnowledgeUploadTask:
    """知识文件上传任务实体。"""

    id: str
    request_id: str
    uploader_user_id: str
    uploader_role: str
    raw_sha256: str
    blob_key: str
    requested_filename: str
    content_type: str
    size: int
    ingest_version: str
    status: KnowledgeUploadTaskStatus
    stage: KnowledgeUploadTaskStage
    content_sha256: str | None
    file_id: str | None
    chunk_count: int
    deduplicated: bool
    replaced: bool
    title_updated: bool
    error_code: str | None
    error_message: str | None
    attempt_count: int
    worker_id: str | None
    lease_expires_at: datetime | None
    last_heartbeat_at: datetime | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    @property
    def original_filename(self) -> str:
        """兼容接口层原始文件名语义。"""
        return self.requested_filename
