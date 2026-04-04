"""知识文件上传任务仓储抽象。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from baozhi_rag.domain.knowledge_upload_task import (
    KnowledgeUploadTask,
    KnowledgeUploadTaskStage,
    KnowledgeUploadTaskStatus,
)


class KnowledgeUploadTaskRepository(Protocol):
    """知识文件上传任务仓储协议。"""

    def create_task(self, task: KnowledgeUploadTask) -> KnowledgeUploadTask:
        """创建上传任务。"""

    def get_task_by_id(self, task_id: str) -> KnowledgeUploadTask | None:
        """按任务标识查询任务。"""

    def get_task_by_id_for_user(
        self,
        task_id: str,
        uploader_user_id: str,
    ) -> KnowledgeUploadTask | None:
        """按任务标识和上传用户查询任务。"""

    def get_task_by_user_and_raw_sha256(
        self,
        uploader_user_id: str,
        raw_sha256: str,
        ingest_version: str,
    ) -> KnowledgeUploadTask | None:
        """按用户、原始哈希和 ingest 版本查询任务。"""

    def list_tasks_by_user(
        self,
        uploader_user_id: str,
        *,
        limit: int,
    ) -> list[KnowledgeUploadTask]:
        """按用户倒序列出最近任务。"""

    def update_submission_context(
        self,
        task_id: str,
        *,
        requested_filename: str,
        source_storage_key: str | None = None,
    ) -> KnowledgeUploadTask | None:
        """更新任务最近一次提交使用的标题与源文件位置。"""

    def claim_next_task(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_expires_at: datetime,
    ) -> KnowledgeUploadTask | None:
        """抢占一条待处理或租约过期的任务。"""

    def refresh_lease(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime,
        heartbeat_at: datetime,
    ) -> bool:
        """刷新任务租约与心跳。"""

    def update_task_progress(
        self,
        task_id: str,
        *,
        worker_id: str,
        stage: KnowledgeUploadTaskStage,
        status: KnowledgeUploadTaskStatus = KnowledgeUploadTaskStatus.PROCESSING,
        content_sha256: str | None = None,
        file_id: str | None = None,
        chunk_count: int | None = None,
        deduplicated: bool | None = None,
        replaced: bool | None = None,
        title_updated: bool | None = None,
    ) -> KnowledgeUploadTask | None:
        """更新任务阶段性处理结果。"""

    def mark_succeeded(
        self,
        task_id: str,
        *,
        worker_id: str,
        stage: KnowledgeUploadTaskStage,
        content_sha256: str | None,
        file_id: str | None,
        chunk_count: int,
        deduplicated: bool,
        replaced: bool,
        title_updated: bool,
        completed_at: datetime,
    ) -> KnowledgeUploadTask | None:
        """把任务标记为成功。"""

    def mark_failed(
        self,
        task_id: str,
        *,
        worker_id: str,
        error_code: str,
        error_message: str,
        failed_at: datetime,
    ) -> KnowledgeUploadTask | None:
        """把任务标记为失败。"""

    def retry_task(
        self,
        task_id: str,
        *,
        uploader_user_id: str,
        queued_at: datetime,
    ) -> KnowledgeUploadTask | None:
        """将失败任务重新入队。"""
