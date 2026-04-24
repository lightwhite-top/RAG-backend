from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from baozhi_rag.domain.knowledge_file import (
    FileStorageProvider,
    FileVisibilityScope,
    KnowledgeFile,
)
from baozhi_rag.domain.knowledge_upload_task import (
    KnowledgeUploadTask,
    KnowledgeUploadTaskStage,
    KnowledgeUploadTaskStatus,
)
from baozhi_rag.domain.user import CurrentUser, UserRole
from baozhi_rag.services.file_upload import StagedUploadFileResult
from baozhi_rag.services.upload_tasks import KnowledgeUploadService


def _build_current_user() -> CurrentUser:
    now = datetime(2026, 4, 24, 10, 0, tzinfo=UTC)
    return CurrentUser(
        id="user-1",
        email="user@example.com",
        username="Light",
        role=UserRole.USER,
        created_at=now,
        updated_at=now,
    )


def _build_task(*, status: KnowledgeUploadTaskStatus, file_id: str | None) -> KnowledgeUploadTask:
    now = datetime(2026, 4, 24, 10, 0, tzinfo=UTC)
    return KnowledgeUploadTask(
        id="task-1",
        request_id="req-1",
        uploader_user_id="user-1",
        uploader_role=UserRole.USER.value,
        raw_sha256="raw-sha",
        source_storage_key="temp/old.pdf",
        requested_filename="policy.pdf",
        content_type="application/pdf",
        size=128,
        ingest_version="v1",
        status=status,
        stage=KnowledgeUploadTaskStage.COMPLETED
        if status is KnowledgeUploadTaskStatus.SUCCEEDED
        else KnowledgeUploadTaskStage.FAILED,
        content_sha256="content-sha" if file_id is not None else None,
        file_id=file_id,
        chunk_count=3 if file_id is not None else 0,
        deduplicated=False,
        replaced=False,
        title_updated=False,
        error_code=None,
        error_message=None,
        attempt_count=1,
        worker_id=None,
        lease_expires_at=None,
        last_heartbeat_at=None,
        created_at=now,
        updated_at=now,
        completed_at=now,
    )


@dataclass
class _StubFileUploadService:
    staged_result: StagedUploadFileResult

    async def stage_async_files(self, files: list[object]) -> list[StagedUploadFileResult]:
        del files
        return [self.staged_result]


@dataclass
class _StubTempFileStore:
    deleted_storage_keys: list[str]

    def delete(self, storage_key: str) -> None:
        self.deleted_storage_keys.append(storage_key)


@dataclass
class _StubTaskRepository:
    existing_task: KnowledgeUploadTask
    retried_task: KnowledgeUploadTask
    retried_task_id: str | None = None

    def get_task_by_user_and_raw_sha256(
        self,
        uploader_user_id: str,
        raw_sha256: str,
        ingest_version: str,
    ) -> KnowledgeUploadTask | None:
        del uploader_user_id, raw_sha256, ingest_version
        return self.existing_task

    def update_submission_context(
        self,
        task_id: str,
        *,
        requested_filename: str,
        source_storage_key: str | None = None,
    ) -> KnowledgeUploadTask | None:
        del task_id, requested_filename, source_storage_key
        return self.existing_task

    def retry_task(
        self,
        task_id: str,
        *,
        uploader_user_id: str,
        queued_at: datetime,
    ) -> KnowledgeUploadTask | None:
        del uploader_user_id, queued_at
        self.retried_task_id = task_id
        return self.retried_task

    def create_task(self, task: KnowledgeUploadTask) -> KnowledgeUploadTask:
        return task


@dataclass
class _StubKnowledgeFileRepository:
    file_to_return: KnowledgeFile | None

    def get_file_by_id(self, file_id: str) -> KnowledgeFile | None:
        del file_id
        return self.file_to_return


def test_submit_files_requeues_succeeded_task_when_linked_file_is_missing() -> None:
    staged_file = StagedUploadFileResult(
        stage_id="stage-1",
        original_filename="policy.pdf",
        safe_filename="policy.pdf",
        content_type="application/pdf",
        size=128,
        sha256="raw-sha",
        temp_storage_key="temp/new.pdf",
        staged_at=datetime(2026, 4, 24, 10, 0, tzinfo=UTC),
    )
    existing_task = _build_task(
        status=KnowledgeUploadTaskStatus.SUCCEEDED,
        file_id="missing-file",
    )
    retried_task = _build_task(
        status=KnowledgeUploadTaskStatus.QUEUED,
        file_id=None,
    )
    retried_task = replace(retried_task, source_storage_key="temp/new.pdf")
    temp_file_store = _StubTempFileStore(deleted_storage_keys=[])
    task_repository = _StubTaskRepository(
        existing_task=existing_task,
        retried_task=retried_task,
    )
    service = KnowledgeUploadService(
        file_upload_service=_StubFileUploadService(staged_result=staged_file),  # type: ignore[arg-type]
        temp_file_store=temp_file_store,  # type: ignore[arg-type]
        task_repository=task_repository,  # type: ignore[arg-type]
        knowledge_file_repository=_StubKnowledgeFileRepository(file_to_return=None),  # type: ignore[arg-type]
        ingest_version="v1",
    )

    results = asyncio.run(
        service.submit_files(
            files=[object()],
            current_user=_build_current_user(),
            request_id="req-1",
        )
    )

    assert task_repository.retried_task_id == "task-1"
    assert results[0].status is KnowledgeUploadTaskStatus.QUEUED
    assert temp_file_store.deleted_storage_keys == ["temp/old.pdf"]


def test_submit_files_reuses_succeeded_task_when_linked_file_still_exists() -> None:
    staged_file = StagedUploadFileResult(
        stage_id="stage-1",
        original_filename="policy.pdf",
        safe_filename="policy.pdf",
        content_type="application/pdf",
        size=128,
        sha256="raw-sha",
        temp_storage_key="temp/new.pdf",
        staged_at=datetime(2026, 4, 24, 10, 0, tzinfo=UTC),
    )
    existing_task = _build_task(
        status=KnowledgeUploadTaskStatus.SUCCEEDED,
        file_id="existing-file",
    )
    existing_file = KnowledgeFile(
        id="existing-file",
        uploader_user_id="user-1",
        original_filename="policy.pdf",
        content_type="application/pdf",
        size=128,
        storage_provider=FileStorageProvider.ALIYUN_OSS,
        storage_key="knowledge-files/user-1/existing-file/policy.pdf",
        visibility_scope=FileVisibilityScope.OWNER_ONLY,
        chunk_count=3,
        uploaded_at=datetime(2026, 4, 24, 10, 0, tzinfo=UTC),
        updated_at=datetime(2026, 4, 24, 10, 0, tzinfo=UTC),
        raw_sha256="raw-sha",
        text_sha256="text-sha",
        content_sha256="content-sha",
    )
    temp_file_store = _StubTempFileStore(deleted_storage_keys=[])
    task_repository = _StubTaskRepository(
        existing_task=existing_task,
        retried_task=_build_task(status=KnowledgeUploadTaskStatus.QUEUED, file_id=None),
    )
    service = KnowledgeUploadService(
        file_upload_service=_StubFileUploadService(staged_result=staged_file),  # type: ignore[arg-type]
        temp_file_store=temp_file_store,  # type: ignore[arg-type]
        task_repository=task_repository,  # type: ignore[arg-type]
        knowledge_file_repository=_StubKnowledgeFileRepository(file_to_return=existing_file),  # type: ignore[arg-type]
        ingest_version="v1",
    )

    results = asyncio.run(
        service.submit_files(
            files=[object()],
            current_user=_build_current_user(),
            request_id="req-1",
        )
    )

    assert task_repository.retried_task_id is None
    assert results[0].status is KnowledgeUploadTaskStatus.SUCCEEDED
    assert temp_file_store.deleted_storage_keys == ["temp/new.pdf"]
