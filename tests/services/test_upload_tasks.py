"""异步上传任务服务测试。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from docx import Document

from baozhi_rag.domain.knowledge_file import KnowledgeFile
from baozhi_rag.domain.knowledge_file_blob import KnowledgeFileBlob
from baozhi_rag.domain.knowledge_file_errors import (
    KnowledgeFileConflictError,
    KnowledgeUploadTaskRetryNotAllowedError,
)
from baozhi_rag.domain.knowledge_upload_task import (
    KnowledgeUploadTask,
    KnowledgeUploadTaskStage,
    KnowledgeUploadTaskStatus,
)
from baozhi_rag.domain.user import CurrentUser, UserRole
from baozhi_rag.infra.storage.local_file_store import LocalFileStore
from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest
from baozhi_rag.services.document_chunking import DocumentChunkService
from baozhi_rag.services.file_upload import AsyncFileUploadInput, FileUploadService
from baozhi_rag.services.upload_tasks import (
    KnowledgeUploadProcessor,
    KnowledgeUploadService,
    UploadTaskObjectStoreProtocol,
)


class AsyncBytesReader:
    """异步读取内存字节流的测试替身。"""

    def __init__(self, data: bytes) -> None:
        self._buffer = BytesIO(data)

    async def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


class InMemoryKnowledgeFileRepository:
    """知识文件仓储测试替身。"""

    def __init__(self) -> None:
        self._files: dict[str, KnowledgeFile] = {}

    def create_file(self, file: KnowledgeFile) -> KnowledgeFile:
        self._ensure_unique(file)
        self._files[file.id] = file
        return file

    def get_file_by_id(self, file_id: str) -> KnowledgeFile | None:
        return self._files.get(file_id)

    def get_file_by_user_and_filename(
        self,
        uploader_user_id: str,
        original_filename: str,
    ) -> KnowledgeFile | None:
        for file in self._files.values():
            if (
                file.uploader_user_id == uploader_user_id
                and file.original_filename == original_filename
            ):
                return file
        return None

    def get_file_by_user_and_sha256(
        self,
        uploader_user_id: str,
        sha256: str,
    ) -> KnowledgeFile | None:
        return self.get_file_by_user_and_content_sha256(uploader_user_id, sha256)

    def get_file_by_user_and_content_sha256(
        self,
        uploader_user_id: str,
        content_sha256: str,
    ) -> KnowledgeFile | None:
        for file in self._files.values():
            if file.uploader_user_id == uploader_user_id and file.content_sha256 == content_sha256:
                return file
        return None

    def get_files_by_ids(self, file_ids: list[str]) -> list[KnowledgeFile]:
        return [self._files[file_id] for file_id in file_ids if file_id in self._files]

    def update_file(self, file_id: str, **kwargs: object) -> KnowledgeFile | None:
        file = self._files.get(file_id)
        if file is None:
            return None
        updated_file = KnowledgeFile(
            id=file.id,
            uploader_user_id=file.uploader_user_id,
            original_filename=str(kwargs.get("original_filename", file.original_filename)),
            content_type=str(kwargs.get("content_type", file.content_type)),
            size=int(cast(int | str, kwargs.get("size", file.size))),
            storage_provider=cast(Any, kwargs.get("storage_provider", file.storage_provider)),
            storage_key=str(kwargs.get("storage_key", file.storage_key)),
            visibility_scope=cast(Any, kwargs.get("visibility_scope", file.visibility_scope)),
            chunk_count=int(cast(int | str, kwargs.get("chunk_count", file.chunk_count))),
            uploaded_at=file.uploaded_at,
            updated_at=file.updated_at,
            raw_sha256=str(kwargs.get("raw_sha256", file.raw_sha256)),
            content_sha256=str(
                kwargs.get(
                    "content_sha256",
                    kwargs.get("sha256", file.content_sha256 or file.sha256),
                )
            ),
        )
        self._ensure_unique(updated_file, ignored_file_id=file_id)
        self._files[file_id] = updated_file
        return updated_file

    def replace_file(self, existing_file_id: str, replacement_file: KnowledgeFile) -> KnowledgeFile:
        self._files.pop(existing_file_id, None)
        self._ensure_unique(replacement_file, ignored_file_id=existing_file_id)
        self._files[replacement_file.id] = replacement_file
        return replacement_file

    def delete_file(self, file_id: str) -> bool:
        return self._files.pop(file_id, None) is not None

    def _ensure_unique(self, candidate: KnowledgeFile, ignored_file_id: str | None = None) -> None:
        for file in self._files.values():
            if file.id == ignored_file_id:
                continue
            if (
                file.uploader_user_id == candidate.uploader_user_id
                and file.original_filename == candidate.original_filename
            ):
                raise KnowledgeFileConflictError("同一用户的同名文件记录冲突")
            if (
                file.uploader_user_id == candidate.uploader_user_id
                and file.content_sha256 == candidate.content_sha256
            ):
                raise KnowledgeFileConflictError("同一用户的同内容文件记录冲突")


class InMemoryBlobRepository:
    """blob 仓储测试替身。"""

    def __init__(self) -> None:
        self._blobs: dict[str, KnowledgeFileBlob] = {}

    def create_blob(self, blob: KnowledgeFileBlob) -> KnowledgeFileBlob:
        if blob.raw_sha256 in self._blobs:
            raise KnowledgeFileConflictError("原始文件 blob 记录冲突")
        self._blobs[blob.raw_sha256] = blob
        return blob

    def get_blob_by_raw_sha256(self, raw_sha256: str) -> KnowledgeFileBlob | None:
        return self._blobs.get(raw_sha256)


class InMemoryUploadTaskRepository:
    """上传任务仓储测试替身。"""

    def __init__(self) -> None:
        self._tasks: dict[str, KnowledgeUploadTask] = {}

    def create_task(self, task: KnowledgeUploadTask) -> KnowledgeUploadTask:
        existing = self.get_task_by_user_and_raw_sha256(
            task.uploader_user_id,
            task.raw_sha256,
            task.ingest_version,
        )
        if existing is not None:
            raise KnowledgeFileConflictError("上传任务冲突")
        self._tasks[task.id] = task
        return task

    def get_task_by_id(self, task_id: str) -> KnowledgeUploadTask | None:
        return self._tasks.get(task_id)

    def get_task_by_id_for_user(
        self,
        task_id: str,
        uploader_user_id: str,
    ) -> KnowledgeUploadTask | None:
        task = self._tasks.get(task_id)
        if task is None or task.uploader_user_id != uploader_user_id:
            return None
        return task

    def get_task_by_user_and_raw_sha256(
        self,
        uploader_user_id: str,
        raw_sha256: str,
        ingest_version: str,
    ) -> KnowledgeUploadTask | None:
        for task in self._tasks.values():
            if (
                task.uploader_user_id == uploader_user_id
                and task.raw_sha256 == raw_sha256
                and task.ingest_version == ingest_version
            ):
                return task
        return None

    def list_tasks_by_user(self, uploader_user_id: str, *, limit: int) -> list[KnowledgeUploadTask]:
        tasks = [task for task in self._tasks.values() if task.uploader_user_id == uploader_user_id]
        tasks.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        return tasks[:limit]

    def update_requested_filename(
        self,
        task_id: str,
        requested_filename: str,
    ) -> KnowledgeUploadTask | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        updated = replace(task, requested_filename=requested_filename, updated_at=datetime.now(UTC))
        self._tasks[task_id] = updated
        return updated

    def claim_next_task(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_expires_at: datetime,
    ) -> KnowledgeUploadTask | None:
        for task in sorted(self._tasks.values(), key=lambda item: (item.created_at, item.id)):
            if task.status is KnowledgeUploadTaskStatus.QUEUED or (
                task.status is KnowledgeUploadTaskStatus.PROCESSING
                and task.lease_expires_at is not None
                and task.lease_expires_at < now
            ):
                updated = replace(
                    task,
                    status=KnowledgeUploadTaskStatus.PROCESSING,
                    worker_id=worker_id,
                    lease_expires_at=lease_expires_at,
                    last_heartbeat_at=now,
                    attempt_count=task.attempt_count + 1,
                    error_code=None,
                    error_message=None,
                    updated_at=now,
                )
                self._tasks[task.id] = updated
                return updated
        return None

    def refresh_lease(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime,
        heartbeat_at: datetime,
    ) -> bool:
        task = self._tasks.get(task_id)
        if task is None or task.worker_id != worker_id:
            return False
        self._tasks[task_id] = replace(
            task,
            lease_expires_at=lease_expires_at,
            last_heartbeat_at=heartbeat_at,
            updated_at=heartbeat_at,
        )
        return True

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
        task = self._tasks.get(task_id)
        if task is None or task.worker_id != worker_id:
            return None
        updated = replace(
            task,
            status=status,
            stage=stage,
            content_sha256=content_sha256 if content_sha256 is not None else task.content_sha256,
            file_id=file_id if file_id is not None else task.file_id,
            chunk_count=chunk_count if chunk_count is not None else task.chunk_count,
            deduplicated=deduplicated if deduplicated is not None else task.deduplicated,
            replaced=replaced if replaced is not None else task.replaced,
            title_updated=title_updated if title_updated is not None else task.title_updated,
            updated_at=datetime.now(UTC),
        )
        self._tasks[task_id] = updated
        return updated

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
        task = self._tasks.get(task_id)
        if task is None or task.worker_id != worker_id:
            return None
        updated = replace(
            task,
            status=KnowledgeUploadTaskStatus.SUCCEEDED,
            stage=stage,
            content_sha256=content_sha256,
            file_id=file_id,
            chunk_count=chunk_count,
            deduplicated=deduplicated,
            replaced=replaced,
            title_updated=title_updated,
            worker_id=None,
            lease_expires_at=None,
            last_heartbeat_at=completed_at,
            updated_at=completed_at,
            completed_at=completed_at,
        )
        self._tasks[task_id] = updated
        return updated

    def mark_failed(
        self,
        task_id: str,
        *,
        worker_id: str,
        error_code: str,
        error_message: str,
        failed_at: datetime,
    ) -> KnowledgeUploadTask | None:
        task = self._tasks.get(task_id)
        if task is None or task.worker_id != worker_id:
            return None
        updated = replace(
            task,
            status=KnowledgeUploadTaskStatus.FAILED,
            stage=KnowledgeUploadTaskStage.FAILED,
            error_code=error_code,
            error_message=error_message,
            worker_id=None,
            lease_expires_at=None,
            last_heartbeat_at=failed_at,
            updated_at=failed_at,
            completed_at=failed_at,
        )
        self._tasks[task_id] = updated
        return updated

    def retry_task(
        self,
        task_id: str,
        *,
        uploader_user_id: str,
        queued_at: datetime,
    ) -> KnowledgeUploadTask | None:
        task = self._tasks.get(task_id)
        if task is None or task.uploader_user_id != uploader_user_id:
            return None
        if task.status is not KnowledgeUploadTaskStatus.FAILED:
            raise KnowledgeUploadTaskRetryNotAllowedError()
        updated = replace(
            task,
            status=KnowledgeUploadTaskStatus.QUEUED,
            stage=KnowledgeUploadTaskStage.UPLOADED,
            content_sha256=None,
            file_id=None,
            chunk_count=0,
            deduplicated=False,
            replaced=False,
            title_updated=False,
            error_code=None,
            error_message=None,
            worker_id=None,
            lease_expires_at=None,
            last_heartbeat_at=None,
            updated_at=queued_at,
            completed_at=None,
        )
        self._tasks[task_id] = updated
        return updated


class FakeObjectStore:
    """对象存储测试替身。"""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.uploaded_keys: list[str] = []
        self.deleted_keys: list[str] = []

    def upload_file(self, *, local_path: Path, storage_key: str) -> None:
        self.objects[storage_key] = local_path.read_bytes()
        self.uploaded_keys.append(storage_key)

    def delete(self, storage_key: str) -> None:
        self.objects.pop(storage_key, None)
        self.deleted_keys.append(storage_key)

    def download_file(self, *, storage_key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.objects[storage_key])


class RecordingChunkStore:
    """索引存储测试替身。"""

    def __init__(self, *, fail_on_index: bool = False) -> None:
        self.ensure_calls = 0
        self.indexed_chunk_batches: list[list[Any]] = []
        self.deleted_file_ids: list[str] = []
        self._fail_on_index = fail_on_index

    def ensure_index(self) -> None:
        self.ensure_calls += 1

    def index_chunks(self, chunks: list[Any]) -> int:
        if self._fail_on_index:
            raise RuntimeError("index failed")
        self.indexed_chunk_batches.append(chunks)
        return len(chunks)

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        self.deleted_file_ids.append(file_id)

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        del request
        return []


class FakeEmbeddingClient:
    """返回固定向量的测试替身。"""

    def ensure_ready(self) -> None:
        return

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


def test_submit_files_reuses_same_task_for_same_raw_hash(tmp_path: Path) -> None:
    """同一用户重复提交完全相同原始文件时应复用同一任务。"""
    service, _, _, blob_repository, task_repository, _ = _build_runtime(tmp_path)
    user = _build_user(UserRole.USER)
    content = _build_docx_bytes("完全相同内容", author="A")

    first_task = asyncio.run(
        service.submit_files(
            [_build_async_input("保险条款.docx", content)],
            current_user=user,
            request_id="request-1",
        )
    )[0]
    second_task = asyncio.run(
        service.submit_files(
            [_build_async_input("保险条款-重传.docx", content)],
            current_user=user,
            request_id="request-2",
        )
    )[0]

    assert first_task.id == second_task.id
    assert len(blob_repository._blobs) == 1
    assert task_repository.get_task_by_id(first_task.id) is not None
    persisted_task = task_repository.get_task_by_id(first_task.id)
    assert persisted_task is not None
    assert persisted_task.requested_filename == "保险条款-重传.docx"


def test_processor_uses_latest_filename_when_same_raw_file_resubmitted(tmp_path: Path) -> None:
    """处理中再次提交相同原始文件时，最终标题应以最新文件名为准。"""
    service, processor, object_store, _, task_repository, file_repository = _build_runtime(tmp_path)
    user = _build_user(UserRole.USER)
    content = _build_docx_bytes("原始内容", author="A")

    first_task = asyncio.run(
        service.submit_files(
            [_build_async_input("旧标题.docx", content)],
            current_user=user,
            request_id="request-1",
        )
    )[0]
    asyncio.run(
        service.submit_files(
            [_build_async_input("新标题.docx", content)],
            current_user=user,
            request_id="request-2",
        )
    )

    processed = processor.process_next_task("worker-1")
    final_task = task_repository.get_task_by_id(first_task.id)
    assert processed is True
    assert final_task is not None
    assert final_task.status is KnowledgeUploadTaskStatus.SUCCEEDED
    persisted_file = file_repository.get_file_by_id(str(final_task.file_id))
    assert persisted_file is not None
    assert persisted_file.original_filename == "新标题.docx"
    assert persisted_file.storage_key.startswith("knowledge-files/user-1/")
    assert persisted_file.storage_key in object_store.uploaded_keys
    assert first_task.blob_key != persisted_file.storage_key


def test_processor_reuses_existing_content_for_different_binary_same_text(tmp_path: Path) -> None:
    """同内容不同二进制上传应命中内容去重，不生成第二条文件记录。"""
    service, processor, _, _, task_repository, file_repository = _build_runtime(tmp_path)
    user = _build_user(UserRole.USER)

    first_task = asyncio.run(
        service.submit_files(
            [_build_async_input("旧标题.docx", _build_docx_bytes("同样正文", author="A"))],
            current_user=user,
            request_id="request-1",
        )
    )[0]
    assert processor.process_next_task("worker-1") is True

    second_task = asyncio.run(
        service.submit_files(
            [_build_async_input("新标题.docx", _build_docx_bytes("同样正文", author="B"))],
            current_user=user,
            request_id="request-2",
        )
    )[0]
    assert processor.process_next_task("worker-1") is True

    first_final = task_repository.get_task_by_id(first_task.id)
    second_final = task_repository.get_task_by_id(second_task.id)
    assert first_final is not None
    assert second_final is not None
    assert second_final.deduplicated is True
    assert second_final.title_updated is True
    assert len(file_repository._files) == 1
    assert first_final.file_id == second_final.file_id
    only_file = next(iter(file_repository._files.values()))
    assert only_file.original_filename == "新标题.docx"


def test_failed_replacement_keeps_old_file_unchanged(tmp_path: Path) -> None:
    """同名不同内容上传若新任务失败，旧文件记录应保持不变。"""
    service, processor, object_store, _, task_repository, file_repository = _build_runtime(tmp_path)
    user = _build_user(UserRole.USER)

    first_task = asyncio.run(
        service.submit_files(
            [_build_async_input("保险条款.docx", _build_docx_bytes("第一版内容", author="A"))],
            current_user=user,
            request_id="request-1",
        )
    )[0]
    assert processor.process_next_task("worker-1") is True
    first_final = task_repository.get_task_by_id(first_task.id)
    assert first_final is not None

    failing_processor = _build_processor(
        tmp_path,
        task_repository=task_repository,
        file_repository=file_repository,
        object_store=processor._object_store,  # type: ignore[attr-defined]
        chunk_store=RecordingChunkStore(fail_on_index=True),
    )

    second_task = asyncio.run(
        service.submit_files(
            [_build_async_input("保险条款.docx", _build_docx_bytes("第二版内容", author="B"))],
            current_user=user,
            request_id="request-2",
        )
    )[0]
    assert failing_processor.process_next_task("worker-2") is True

    second_final = task_repository.get_task_by_id(second_task.id)
    assert second_final is not None
    assert second_final.status is KnowledgeUploadTaskStatus.FAILED
    persisted_file = file_repository.get_file_by_id(str(first_final.file_id))
    assert persisted_file is not None
    assert persisted_file.original_filename == "保险条款.docx"
    assert persisted_file.content_sha256
    assert len(file_repository._files) == 1
    assert any(key.startswith("knowledge-files/user-1/") for key in object_store.deleted_keys)


def test_retry_task_requeues_failed_task(tmp_path: Path) -> None:
    """失败任务应允许重新入队。"""
    service, _, _, _, task_repository, _ = _build_runtime(tmp_path)
    user = _build_user(UserRole.USER)
    content = _build_docx_bytes("失败重试内容", author="A")
    task = asyncio.run(
        service.submit_files(
            [_build_async_input("保险条款.docx", content)],
            current_user=user,
            request_id="request-1",
        )
    )[0]
    claimed_task = task_repository.claim_next_task(
        worker_id="worker-1",
        now=datetime.now(UTC),
        lease_expires_at=datetime.now(UTC),
    )
    assert claimed_task is not None
    task_repository.mark_failed(
        task.id,
        worker_id="worker-1",
        error_code="index_failed",
        error_message="index failed",
        failed_at=datetime.now(UTC),
    )

    retried = service.retry_task(task_id=task.id, current_user=user)

    assert retried.status is KnowledgeUploadTaskStatus.QUEUED
    assert retried.stage is KnowledgeUploadTaskStage.UPLOADED
    assert retried.error_code is None


def _build_runtime(
    tmp_path: Path,
) -> tuple[
    KnowledgeUploadService,
    KnowledgeUploadProcessor,
    FakeObjectStore,
    InMemoryBlobRepository,
    InMemoryUploadTaskRepository,
    InMemoryKnowledgeFileRepository,
]:
    temp_file_store = LocalFileStore(tmp_path)
    object_store = FakeObjectStore()
    blob_repository = InMemoryBlobRepository()
    task_repository = InMemoryUploadTaskRepository()
    file_repository = InMemoryKnowledgeFileRepository()
    chunk_store = RecordingChunkStore()
    service = KnowledgeUploadService(
        file_upload_service=FileUploadService(temp_file_store),
        temp_file_store=temp_file_store,
        object_store=object_store,
        blob_repository=blob_repository,
        task_repository=task_repository,
        raw_object_prefix="knowledge-files/raw",
        ingest_version="v1",
    )
    processor = _build_processor(
        tmp_path,
        task_repository=task_repository,
        file_repository=file_repository,
        object_store=object_store,
        chunk_store=chunk_store,
    )
    return service, processor, object_store, blob_repository, task_repository, file_repository


def _build_processor(
    tmp_path: Path,
    *,
    task_repository: InMemoryUploadTaskRepository,
    file_repository: InMemoryKnowledgeFileRepository,
    object_store: UploadTaskObjectStoreProtocol,
    chunk_store: RecordingChunkStore,
) -> KnowledgeUploadProcessor:
    temp_file_store = LocalFileStore(tmp_path)
    return KnowledgeUploadProcessor(
        temp_file_store=temp_file_store,
        object_store=object_store,
        final_object_prefix="knowledge-files",
        task_repository=task_repository,
        knowledge_file_repository=file_repository,
        chunk_service=DocumentChunkService(
            chunk_size=200,
            chunk_overlap=20,
            convert_temp_dir=tmp_path / "tmp",
        ),
        chunk_store=chunk_store,
        chunk_embedding_service=ChunkEmbeddingService(FakeEmbeddingClient()),
        lease_seconds=180,
        heartbeat_interval_seconds=30.0,
    )


def _build_user(role: UserRole) -> CurrentUser:
    return CurrentUser(
        id="user-1",
        email="user@example.com",
        username="tester",
        role=role,
        created_at=datetime(2026, 4, 3, tzinfo=UTC),
        updated_at=datetime(2026, 4, 3, tzinfo=UTC),
    )


def _build_async_input(filename: str, data: bytes) -> AsyncFileUploadInput:
    return AsyncFileUploadInput(
        filename=filename,
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        stream=AsyncBytesReader(data),
    )


def _build_docx_bytes(text: str, *, author: str) -> bytes:
    """构造带可变元数据的 docx 二进制。"""
    document = Document()
    document.core_properties.author = author
    document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
