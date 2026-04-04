"""知识文件异步上传任务服务。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from baozhi_rag.domain.knowledge_file import FileStorageProvider, FileVisibilityScope, KnowledgeFile
from baozhi_rag.domain.knowledge_file_blob import KnowledgeFileBlob
from baozhi_rag.domain.knowledge_file_blob_repository import KnowledgeFileBlobRepository
from baozhi_rag.domain.knowledge_file_errors import (
    KnowledgeFileConflictError,
    KnowledgeUploadTaskNotFoundError,
)
from baozhi_rag.domain.knowledge_file_repository import KnowledgeFileRepository
from baozhi_rag.domain.knowledge_upload_task import (
    KnowledgeUploadTask,
    KnowledgeUploadTaskStage,
    KnowledgeUploadTaskStatus,
)
from baozhi_rag.domain.knowledge_upload_task_repository import KnowledgeUploadTaskRepository
from baozhi_rag.domain.user import CurrentUser, UserRole
from baozhi_rag.infra.storage.local_file_store import LocalFileStore
from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.chunk_search import ChunkSearchStore
from baozhi_rag.services.document_chunking import DocumentChunk, DocumentChunkService
from baozhi_rag.services.file_upload import (
    AsyncFileUploadInput,
    FileUploadService,
    StagedUploadFileResult,
)

LOGGER = logging.getLogger(__name__)


class UploadTaskObjectStoreProtocol(Protocol):
    """上传任务使用的对象存储协议。"""

    def upload_file(self, *, local_path: Path, storage_key: str) -> None:
        """上传本地文件到对象存储。"""

    def delete(self, storage_key: str) -> None:
        """删除对象存储中的文件。"""

    def download_file(self, *, storage_key: str, local_path: Path) -> None:
        """把对象存储文件下载到本地路径。"""


@dataclass(frozen=True, slots=True)
class UploadTaskProcessResult:
    """后台处理完成后的任务结果。"""

    file_id: str | None
    content_sha256: str | None
    chunk_count: int
    deduplicated: bool
    replaced: bool
    title_updated: bool
    cleanup_file_ids: list[str]
    cleanup_storage_keys: list[str]


class KnowledgeUploadService:
    """面向 API 的上传任务提交、查询与重试服务。"""

    def __init__(
        self,
        *,
        file_upload_service: FileUploadService,
        temp_file_store: LocalFileStore,
        object_store: UploadTaskObjectStoreProtocol,
        blob_repository: KnowledgeFileBlobRepository,
        task_repository: KnowledgeUploadTaskRepository,
        raw_object_prefix: str,
        ingest_version: str,
    ) -> None:
        """初始化上传任务服务。"""
        self._file_upload_service = file_upload_service
        self._temp_file_store = temp_file_store
        self._object_store = object_store
        self._blob_repository = blob_repository
        self._task_repository = task_repository
        self._raw_object_prefix = raw_object_prefix.strip().strip("/")
        self._ingest_version = ingest_version.strip() or "v1"

    async def submit_files(
        self,
        files: list[AsyncFileUploadInput],
        *,
        current_user: CurrentUser,
        request_id: str,
    ) -> list[KnowledgeUploadTask]:
        """接收文件并创建或复用上传任务。"""
        staged_files = await self._file_upload_service.stage_async_files(files)
        results: list[KnowledgeUploadTask] = []

        try:
            for staged_file in staged_files:
                existing_task = self._task_repository.get_task_by_user_and_raw_sha256(
                    current_user.id,
                    staged_file.sha256,
                    self._ingest_version,
                )
                if existing_task is not None:
                    updated_task = self._task_repository.update_requested_filename(
                        existing_task.id,
                        staged_file.original_filename,
                    )
                    results.append(updated_task or existing_task)
                    continue

                blob = self._ensure_blob(staged_file)
                task = self._build_task(
                    request_id=request_id,
                    current_user=current_user,
                    staged_file=staged_file,
                    blob=blob,
                )
                try:
                    persisted_task = self._task_repository.create_task(task)
                except Exception:
                    # 并发重复提交以数据库唯一键兜底，再回读已有任务收敛。
                    existing_task = self._task_repository.get_task_by_user_and_raw_sha256(
                        current_user.id,
                        staged_file.sha256,
                        self._ingest_version,
                    )
                    if existing_task is None:
                        raise
                    persisted_task = (
                        self._task_repository.update_requested_filename(
                            existing_task.id,
                            staged_file.original_filename,
                        )
                        or existing_task
                    )
                results.append(persisted_task)
        finally:
            self._cleanup_staged_files(staged_files)

        return results

    def get_task(self, *, task_id: str, current_user: CurrentUser) -> KnowledgeUploadTask:
        """查询当前用户的单条上传任务。"""
        task = self._task_repository.get_task_by_id_for_user(task_id, current_user.id)
        if task is None:
            raise KnowledgeUploadTaskNotFoundError()
        return task

    def list_tasks(
        self,
        *,
        current_user: CurrentUser,
        limit: int = 20,
    ) -> list[KnowledgeUploadTask]:
        """查询当前用户最近的上传任务。"""
        return self._task_repository.list_tasks_by_user(current_user.id, limit=limit)

    def retry_task(self, *, task_id: str, current_user: CurrentUser) -> KnowledgeUploadTask:
        """重试当前用户的失败任务。"""
        task = self._task_repository.retry_task(
            task_id,
            uploader_user_id=current_user.id,
            queued_at=datetime.now(UTC),
        )
        if task is None:
            raise KnowledgeUploadTaskNotFoundError()
        return task

    def _ensure_blob(self, staged_file: StagedUploadFileResult) -> KnowledgeFileBlob:
        """确保原始文件 blob 已存在。"""
        raw_sha256 = str(staged_file.sha256)
        existing_blob = self._blob_repository.get_blob_by_raw_sha256(raw_sha256)
        if existing_blob is not None:
            return existing_blob

        temp_storage_key = str(staged_file.temp_storage_key)
        temp_file_path = self._temp_file_store.resolve_path(temp_storage_key)
        blob = self._build_blob(staged_file)
        self._object_store.upload_file(local_path=temp_file_path, storage_key=blob.storage_key)

        try:
            return self._blob_repository.create_blob(blob)
        except KnowledgeFileConflictError:
            existing_blob = self._blob_repository.get_blob_by_raw_sha256(raw_sha256)
            if existing_blob is None:
                raise
            return existing_blob

    def _build_blob(self, staged_file: StagedUploadFileResult) -> KnowledgeFileBlob:
        """基于暂存文件构造 blob 记录。"""
        now = datetime.now(UTC)
        raw_sha256 = str(staged_file.sha256)
        original_filename = str(staged_file.original_filename)
        safe_filename = FileUploadService.sanitize_filename(original_filename)
        suffix = Path(safe_filename).suffix.lower()
        storage_key = self._build_blob_storage_key(raw_sha256=raw_sha256, suffix=suffix)
        return KnowledgeFileBlob(
            id=uuid4().hex,
            raw_sha256=raw_sha256,
            content_type=str(staged_file.content_type),
            size=int(staged_file.size),
            storage_provider=FileStorageProvider.ALIYUN_OSS,
            storage_key=storage_key,
            created_at=now,
            updated_at=now,
        )

    def _build_task(
        self,
        *,
        request_id: str,
        current_user: CurrentUser,
        staged_file: StagedUploadFileResult,
        blob: KnowledgeFileBlob,
    ) -> KnowledgeUploadTask:
        """构造新上传任务。"""
        now = datetime.now(UTC)
        return KnowledgeUploadTask(
            id=uuid4().hex,
            request_id=request_id,
            uploader_user_id=current_user.id,
            uploader_role=current_user.role.value,
            raw_sha256=blob.raw_sha256,
            blob_key=blob.storage_key,
            requested_filename=str(staged_file.original_filename),
            content_type=str(staged_file.content_type),
            size=int(staged_file.size),
            ingest_version=self._ingest_version,
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
            attempt_count=0,
            worker_id=None,
            lease_expires_at=None,
            last_heartbeat_at=None,
            created_at=now,
            updated_at=now,
            completed_at=None,
        )

    def _build_blob_storage_key(self, *, raw_sha256: str, suffix: str) -> str:
        """构造原始 blob 的对象键。"""
        extension = suffix if suffix.startswith(".") else ""
        parts = [
            part
            for part in [self._raw_object_prefix, raw_sha256[:2], f"{raw_sha256}{extension}"]
            if part
        ]
        return "/".join(parts)

    def _cleanup_staged_files(self, staged_files: Sequence[StagedUploadFileResult]) -> None:
        """清理接口请求生成的临时文件。"""
        for staged_file in reversed(staged_files):
            with suppress(Exception):
                self._temp_file_store.delete(str(staged_file.temp_storage_key))


class KnowledgeUploadProcessor:
    """后台处理上传任务并完成解析、去重、向量化与入库。"""

    def __init__(
        self,
        *,
        temp_file_store: LocalFileStore,
        object_store: UploadTaskObjectStoreProtocol,
        final_object_prefix: str,
        task_repository: KnowledgeUploadTaskRepository,
        knowledge_file_repository: KnowledgeFileRepository,
        chunk_service: DocumentChunkService,
        chunk_store: ChunkSearchStore,
        chunk_embedding_service: ChunkEmbeddingService,
        lease_seconds: int,
        heartbeat_interval_seconds: float,
    ) -> None:
        """初始化后台处理器。"""
        self._temp_file_store = temp_file_store
        self._object_store = object_store
        self._final_object_prefix = final_object_prefix.strip().strip("/")
        self._task_repository = task_repository
        self._knowledge_file_repository = knowledge_file_repository
        self._chunk_service = chunk_service
        self._chunk_store = chunk_store
        self._chunk_embedding_service = chunk_embedding_service
        self._lease_seconds = lease_seconds
        self._heartbeat_interval_seconds = heartbeat_interval_seconds

    def process_next_task(self, worker_id: str) -> bool:
        """抢占并处理下一条可执行任务。"""
        now = datetime.now(UTC)
        task = self._task_repository.claim_next_task(
            worker_id=worker_id,
            now=now,
            lease_expires_at=now + timedelta(seconds=self._lease_seconds),
        )
        if task is None:
            return False

        self._process_claimed_task(task=task, worker_id=worker_id)
        return True

    def _process_claimed_task(self, *, task: KnowledgeUploadTask, worker_id: str) -> None:
        """处理已抢占的任务。"""
        stop_event = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(task.id, worker_id, stop_event),
            daemon=True,
        )
        heartbeat_thread.start()

        local_storage_key = self._build_worker_storage_key(task)
        local_file_path = self._temp_file_store.resolve_path(local_storage_key)

        try:
            self._task_repository.update_task_progress(
                task.id,
                worker_id=worker_id,
                stage=KnowledgeUploadTaskStage.PARSING,
            )
            self._object_store.download_file(storage_key=task.blob_key, local_path=local_file_path)
            preview_chunks = self._chunk_service.chunk_document(
                file_path=local_file_path,
                source_filename=task.requested_filename,
                storage_key=task.blob_key,
                file_id=task.id,
            )
            content_sha256 = self._build_content_sha256(preview_chunks)
            task = self._task_repository.get_task_by_id(task.id) or task
            self._task_repository.update_task_progress(
                task.id,
                worker_id=worker_id,
                stage=KnowledgeUploadTaskStage.PARSING,
                content_sha256=content_sha256,
            )
            process_result = self._resolve_task_result(
                task=task,
                worker_id=worker_id,
                content_sha256=content_sha256,
                preview_chunks=preview_chunks,
                local_file_path=local_file_path,
            )
            self._task_repository.mark_succeeded(
                task.id,
                worker_id=worker_id,
                stage=KnowledgeUploadTaskStage.COMPLETED,
                content_sha256=process_result.content_sha256,
                file_id=process_result.file_id,
                chunk_count=process_result.chunk_count,
                deduplicated=process_result.deduplicated,
                replaced=process_result.replaced,
                title_updated=process_result.title_updated,
                completed_at=datetime.now(UTC),
            )
            for cleanup_file_id in process_result.cleanup_file_ids:
                with suppress(Exception):
                    self._chunk_store.delete_chunks_by_file_id(cleanup_file_id)
            for cleanup_storage_key in process_result.cleanup_storage_keys:
                with suppress(Exception):
                    self._object_store.delete(cleanup_storage_key)
        except Exception as exc:
            error_code = getattr(exc, "error_code", "knowledge_upload_task_failed")
            error_message = getattr(exc, "message", str(exc)).strip() or "上传任务处理失败"
            self._task_repository.mark_failed(
                task.id,
                worker_id=worker_id,
                error_code=str(error_code),
                error_message=error_message,
                failed_at=datetime.now(UTC),
            )
            LOGGER.exception(
                "knowledge_upload_task_failed task_id=%s worker_id=%s error_code=%s",
                task.id,
                worker_id,
                error_code,
            )
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=max(self._heartbeat_interval_seconds, 1.0))
            with suppress(Exception):
                self._temp_file_store.delete(local_storage_key)

    def _resolve_task_result(
        self,
        *,
        task: KnowledgeUploadTask,
        worker_id: str,
        content_sha256: str,
        preview_chunks: list[DocumentChunk],
        local_file_path: Path,
    ) -> UploadTaskProcessResult:
        """根据内容哈希与现有文件状态决定最终处理结果。"""
        latest_task = self._task_repository.get_task_by_id(task.id) or task
        requested_filename = latest_task.requested_filename
        existing_same_name = self._knowledge_file_repository.get_file_by_user_and_filename(
            latest_task.uploader_user_id,
            requested_filename,
        )
        existing_same_content = self._knowledge_file_repository.get_file_by_user_and_content_sha256(
            latest_task.uploader_user_id,
            content_sha256,
        )

        if existing_same_name is not None and existing_same_name.content_sha256 == content_sha256:
            return UploadTaskProcessResult(
                file_id=existing_same_name.id,
                content_sha256=content_sha256,
                chunk_count=existing_same_name.chunk_count,
                deduplicated=True,
                replaced=False,
                title_updated=False,
                cleanup_file_ids=[],
                cleanup_storage_keys=[],
            )

        if existing_same_content is not None and (
            existing_same_name is None or existing_same_name.id == existing_same_content.id
        ):
            title_updated = existing_same_content.original_filename != requested_filename
            if title_updated:
                updated_file = self._knowledge_file_repository.update_file(
                    existing_same_content.id,
                    original_filename=requested_filename,
                )
                existing_same_content = updated_file or existing_same_content
            return UploadTaskProcessResult(
                file_id=existing_same_content.id,
                content_sha256=content_sha256,
                chunk_count=existing_same_content.chunk_count,
                deduplicated=True,
                replaced=False,
                title_updated=title_updated,
                cleanup_file_ids=[],
                cleanup_storage_keys=[],
            )

        if (
            existing_same_name is not None
            and existing_same_content is not None
            and existing_same_name.id != existing_same_content.id
        ):
            self._knowledge_file_repository.delete_file(existing_same_name.id)
            updated_file = self._knowledge_file_repository.update_file(
                existing_same_content.id,
                original_filename=requested_filename,
            )
            resolved_file = updated_file or existing_same_content
            return UploadTaskProcessResult(
                file_id=resolved_file.id,
                content_sha256=content_sha256,
                chunk_count=resolved_file.chunk_count,
                deduplicated=True,
                replaced=False,
                title_updated=True,
                cleanup_file_ids=[existing_same_name.id],
                cleanup_storage_keys=[existing_same_name.storage_key],
            )

        candidate_file = self._build_knowledge_file(
            task=latest_task,
            original_filename=requested_filename,
            content_sha256=content_sha256,
        )
        uploaded_final_object = False
        indexed = False
        try:
            # 原始 blob 继续承担去重与重试下载职责，最终知识文件则固定写入用户目录，
            # 这样检索返回与审计链路都能稳定落在 `knowledge-files/<user_id>/...` 下。
            self._object_store.upload_file(
                local_path=local_file_path,
                storage_key=candidate_file.storage_key,
            )
            uploaded_final_object = True
            chunks = self._materialize_chunks(
                preview_chunks=preview_chunks,
                knowledge_file=candidate_file,
            )
            self._task_repository.update_task_progress(
                latest_task.id,
                worker_id=worker_id,
                stage=KnowledgeUploadTaskStage.INDEXING,
                content_sha256=content_sha256,
                file_id=candidate_file.id,
                chunk_count=len(chunks),
            )
            self._chunk_store.ensure_index()
            self._chunk_store.index_chunks(chunks)
            indexed = True
            if existing_same_name is not None:
                persisted_file = self._knowledge_file_repository.replace_file(
                    existing_same_name.id,
                    replace(candidate_file, chunk_count=len(chunks)),
                )
                return UploadTaskProcessResult(
                    file_id=persisted_file.id,
                    content_sha256=content_sha256,
                    chunk_count=len(chunks),
                    deduplicated=False,
                    replaced=True,
                    title_updated=False,
                    cleanup_file_ids=[existing_same_name.id],
                    cleanup_storage_keys=[existing_same_name.storage_key],
                )

            persisted_file = self._knowledge_file_repository.create_file(
                replace(candidate_file, chunk_count=len(chunks))
            )
            return UploadTaskProcessResult(
                file_id=persisted_file.id,
                content_sha256=content_sha256,
                chunk_count=len(chunks),
                deduplicated=False,
                replaced=False,
                title_updated=False,
                cleanup_file_ids=[],
                cleanup_storage_keys=[],
            )
        except KnowledgeFileConflictError:
            if indexed:
                with suppress(Exception):
                    self._chunk_store.delete_chunks_by_file_id(candidate_file.id)
            if uploaded_final_object:
                with suppress(Exception):
                    self._object_store.delete(candidate_file.storage_key)
            return self._resolve_conflict_after_index(
                latest_task=latest_task,
                requested_filename=requested_filename,
                content_sha256=content_sha256,
            )
        except Exception:
            if indexed:
                with suppress(Exception):
                    self._chunk_store.delete_chunks_by_file_id(candidate_file.id)
            if uploaded_final_object:
                with suppress(Exception):
                    self._object_store.delete(candidate_file.storage_key)
            raise

    def _resolve_conflict_after_index(
        self,
        *,
        latest_task: KnowledgeUploadTask,
        requested_filename: str,
        content_sha256: str,
    ) -> UploadTaskProcessResult:
        """在数据库唯一键冲突后回读现状，收敛为最终任务结果。"""
        existing_same_name = self._knowledge_file_repository.get_file_by_user_and_filename(
            latest_task.uploader_user_id,
            requested_filename,
        )
        existing_same_content = self._knowledge_file_repository.get_file_by_user_and_content_sha256(
            latest_task.uploader_user_id,
            content_sha256,
        )

        if existing_same_name is not None and existing_same_name.content_sha256 == content_sha256:
            return UploadTaskProcessResult(
                file_id=existing_same_name.id,
                content_sha256=content_sha256,
                chunk_count=existing_same_name.chunk_count,
                deduplicated=True,
                replaced=False,
                title_updated=False,
                cleanup_file_ids=[],
                cleanup_storage_keys=[],
            )

        if existing_same_content is not None:
            title_updated = existing_same_content.original_filename != requested_filename
            if title_updated:
                updated_file = self._knowledge_file_repository.update_file(
                    existing_same_content.id,
                    original_filename=requested_filename,
                )
                existing_same_content = updated_file or existing_same_content
            cleanup_file_ids: list[str] = []
            if existing_same_name is not None and existing_same_name.id != existing_same_content.id:
                self._knowledge_file_repository.delete_file(existing_same_name.id)
                cleanup_file_ids.append(existing_same_name.id)
            return UploadTaskProcessResult(
                file_id=existing_same_content.id,
                content_sha256=content_sha256,
                chunk_count=existing_same_content.chunk_count,
                deduplicated=True,
                replaced=False,
                title_updated=title_updated,
                cleanup_file_ids=cleanup_file_ids,
                cleanup_storage_keys=[existing_same_name.storage_key]
                if cleanup_file_ids and existing_same_name is not None
                else [],
            )

        raise KnowledgeFileConflictError("上传任务收敛失败，请稍后重试")

    def _build_content_sha256(self, preview_chunks: list[DocumentChunk]) -> str:
        """基于解析后的文档内容计算稳定哈希。"""
        payload = [
            {
                "chunk_index": chunk.chunk_index,
                "char_count": chunk.char_count,
                "content": chunk.content,
                "merged_terms": chunk.merged_terms,
            }
            for chunk in preview_chunks
        ]
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _materialize_chunks(
        self,
        *,
        preview_chunks: list[DocumentChunk],
        knowledge_file: KnowledgeFile,
    ) -> list[DocumentChunk]:
        """把预览 chunk 转换为最终入库 chunk。"""
        scoped_chunks = [
            replace(
                chunk,
                file_id=knowledge_file.id,
                chunk_id=f"{knowledge_file.id}-chunk-{chunk.chunk_index}",
                source_filename=knowledge_file.original_filename,
                storage_key=knowledge_file.storage_key,
                uploader_user_id=knowledge_file.uploader_user_id,
                visibility_scope=knowledge_file.visibility_scope.value,
            )
            for chunk in preview_chunks
        ]
        return self._chunk_embedding_service.embed_chunks(scoped_chunks)

    def _build_knowledge_file(
        self,
        *,
        task: KnowledgeUploadTask,
        original_filename: str,
        content_sha256: str,
    ) -> KnowledgeFile:
        """基于任务信息构造待落库文件元数据。"""
        uploaded_at = datetime.now(UTC)
        file_id = uuid4().hex
        return KnowledgeFile(
            id=file_id,
            uploader_user_id=task.uploader_user_id,
            original_filename=original_filename,
            content_type=task.content_type,
            size=task.size,
            storage_provider=FileStorageProvider.ALIYUN_OSS,
            storage_key=self._build_storage_key(
                uploader_user_id=task.uploader_user_id,
                file_id=file_id,
                safe_filename=FileUploadService.sanitize_filename(original_filename),
            ),
            visibility_scope=self._resolve_visibility_scope(task),
            chunk_count=0,
            uploaded_at=uploaded_at,
            updated_at=uploaded_at,
            raw_sha256=task.raw_sha256,
            content_sha256=content_sha256,
        )

    def _build_storage_key(
        self,
        *,
        uploader_user_id: str,
        file_id: str,
        safe_filename: str,
    ) -> str:
        """构造最终知识文件的 OSS 对象键。"""
        parts = [
            part
            for part in [self._final_object_prefix, uploader_user_id, file_id, safe_filename]
            if part
        ]
        return "/".join(parts)

    def _resolve_visibility_scope(self, task: KnowledgeUploadTask) -> FileVisibilityScope:
        """根据上传者角色决定文件可见性。"""
        if task.uploader_role == UserRole.ADMIN.value:
            return FileVisibilityScope.GLOBAL
        return FileVisibilityScope.OWNER_ONLY

    def _build_worker_storage_key(self, task: KnowledgeUploadTask) -> str:
        """构造 worker 下载原始文件时使用的临时路径。"""
        safe_filename = FileUploadService.sanitize_filename(task.requested_filename)
        date_path = datetime.now(UTC).strftime("%Y/%m/%d")
        return f"_worker/{date_path}/{task.id}_{safe_filename}"

    def _heartbeat_loop(
        self,
        task_id: str,
        worker_id: str,
        stop_event: threading.Event,
    ) -> None:
        """后台刷新任务租约，防止长任务因租约过期被重复消费。"""
        interval = max(self._heartbeat_interval_seconds, 1.0)
        while not stop_event.wait(interval):
            now = datetime.now(UTC)
            self._task_repository.refresh_lease(
                task_id=task_id,
                worker_id=worker_id,
                lease_expires_at=now + timedelta(seconds=self._lease_seconds),
                heartbeat_at=now,
            )


class KnowledgeUploadWorker:
    """异步轮询 MySQL 任务表的后台 worker。"""

    def __init__(
        self,
        *,
        processor: KnowledgeUploadProcessor,
        worker_id: str,
        poll_interval_seconds: float,
    ) -> None:
        """初始化后台 worker。"""
        self._processor = processor
        self._worker_id = worker_id
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        """持续轮询并处理上传任务。"""
        while not self._stop_event.is_set():
            processed = await asyncio.to_thread(
                self._processor.process_next_task,
                self._worker_id,
            )
            if processed:
                continue
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=max(self._poll_interval_seconds, 0.1),
                )
            except TimeoutError:
                continue

    def stop(self) -> None:
        """请求后台 worker 停止。"""
        self._stop_event.set()
