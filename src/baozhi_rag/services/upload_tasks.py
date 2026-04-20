"""知识文件异步上传任务服务。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tempfile
import threading
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from baozhi_rag.domain.knowledge_file import FileStorageProvider, FileVisibilityScope, KnowledgeFile
from baozhi_rag.domain.knowledge_file_errors import (
    KnowledgeFileConflictError,
    KnowledgeUploadTaskNotFoundError,
    KnowledgeUploadTaskSourceMissingError,
)
from baozhi_rag.domain.knowledge_file_image_asset import KnowledgeFileImageAsset
from baozhi_rag.domain.knowledge_file_image_asset_repository import (
    KnowledgeFileImageAssetRepository,
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
from baozhi_rag.services.document_chunking import (
    ChunkImageAsset,
    ChunkImageAssetRef,
    ChunkType,
    DocumentChunk,
    DocumentChunkService,
    UnsupportedDocumentTypeError,
)
from baozhi_rag.services.document_image_understanding import (
    DocumentImageUnderstandingResult,
    DocumentImageUnderstandingService,
)
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


@dataclass(frozen=True, slots=True)
class PreparedChunkImageAsset:
    """图片识别与上传前的预处理资产。"""

    preview_asset: ChunkImageAsset
    analysis: DocumentImageUnderstandingResult


class KnowledgeUploadService:
    """面向 API 的上传任务提交、查询与重试服务。"""

    _SUPPORTED_SUFFIXES = {".docx", ".doc", ".pdf"}

    def __init__(
        self,
        *,
        file_upload_service: FileUploadService,
        temp_file_store: LocalFileStore,
        task_repository: KnowledgeUploadTaskRepository,
        ingest_version: str,
    ) -> None:
        """初始化上传任务服务。"""
        self._file_upload_service = file_upload_service
        self._temp_file_store = temp_file_store
        self._task_repository = task_repository
        self._ingest_version = ingest_version.strip() or "v1"

    async def submit_files(
        self,
        files: list[AsyncFileUploadInput],
        *,
        current_user: CurrentUser,
        request_id: str,
    ) -> list[KnowledgeUploadTask]:
        """接收文件并创建或复用上传任务。"""
        self._validate_supported_files(files)
        staged_files = await self._file_upload_service.stage_async_files(files)
        results: list[KnowledgeUploadTask] = []
        retained_storage_keys: set[str] = set()
        superseded_storage_keys: list[str] = []

        try:
            for staged_file in staged_files:
                existing_task = self._task_repository.get_task_by_user_and_raw_sha256(
                    current_user.id,
                    staged_file.sha256,
                    self._ingest_version,
                )
                if existing_task is not None:
                    reused_task = self._reuse_existing_task(
                        existing_task=existing_task,
                        staged_file=staged_file,
                        uploader_user_id=current_user.id,
                        retained_storage_keys=retained_storage_keys,
                        superseded_storage_keys=superseded_storage_keys,
                    )
                    results.append(reused_task)
                    continue

                task = self._build_task(
                    request_id=request_id,
                    current_user=current_user,
                    staged_file=staged_file,
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
                    persisted_task = self._reuse_existing_task(
                        existing_task=existing_task,
                        staged_file=staged_file,
                        uploader_user_id=current_user.id,
                        retained_storage_keys=retained_storage_keys,
                        superseded_storage_keys=superseded_storage_keys,
                    )
                else:
                    retained_storage_keys.add(task.source_storage_key)
                results.append(persisted_task)
        finally:
            self._cleanup_redundant_staged_files(staged_files, retained_storage_keys)
            self._cleanup_storage_keys(superseded_storage_keys)

        return results

    def _validate_supported_files(self, files: Sequence[AsyncFileUploadInput]) -> None:
        """在进入异步任务链路前拦截不支持的文档格式。"""
        for file_input in files:
            suffix = Path(file_input.filename or "").suffix.lower()
            if suffix not in self._SUPPORTED_SUFFIXES:
                msg = f"暂不支持的文件格式: {suffix or 'unknown'}"
                raise UnsupportedDocumentTypeError(msg)

    def _reuse_existing_task(
        self,
        *,
        existing_task: KnowledgeUploadTask,
        staged_file: StagedUploadFileResult,
        uploader_user_id: str,
        retained_storage_keys: set[str],
        superseded_storage_keys: list[str],
    ) -> KnowledgeUploadTask:
        """复用同原始文件哈希的已有任务，并在失败时重新入队。"""
        next_source_storage_key: str | None = None
        if existing_task.status in {
            KnowledgeUploadTaskStatus.QUEUED,
            KnowledgeUploadTaskStatus.FAILED,
        }:
            next_source_storage_key = staged_file.temp_storage_key
            retained_storage_keys.add(next_source_storage_key)
            superseded_storage_keys.append(existing_task.source_storage_key)

        updated_task = (
            self._task_repository.update_submission_context(
                existing_task.id,
                requested_filename=staged_file.original_filename,
                source_storage_key=next_source_storage_key,
            )
            or existing_task
        )

        if existing_task.status is KnowledgeUploadTaskStatus.FAILED:
            retried_task = self._task_repository.retry_task(
                existing_task.id,
                uploader_user_id=uploader_user_id,
                queued_at=datetime.now(UTC),
            )
            return retried_task or updated_task

        return updated_task

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

    def _build_task(
        self,
        *,
        request_id: str,
        current_user: CurrentUser,
        staged_file: StagedUploadFileResult,
    ) -> KnowledgeUploadTask:
        """构造新上传任务。"""
        now = datetime.now(UTC)
        return KnowledgeUploadTask(
            id=uuid4().hex,
            request_id=request_id,
            uploader_user_id=current_user.id,
            uploader_role=current_user.role.value,
            raw_sha256=staged_file.sha256,
            source_storage_key=staged_file.temp_storage_key,
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

    def _cleanup_redundant_staged_files(
        self,
        staged_files: Sequence[StagedUploadFileResult],
        retained_storage_keys: set[str],
    ) -> None:
        """清理本次提交中未被任务接管的冗余临时文件。"""
        for staged_file in reversed(staged_files):
            if staged_file.temp_storage_key in retained_storage_keys:
                continue
            with suppress(Exception):
                self._temp_file_store.delete(str(staged_file.temp_storage_key))

    def _cleanup_storage_keys(self, storage_keys: Sequence[str]) -> None:
        """清理已被新提交通知替换的旧源文件。"""
        for storage_key in reversed(storage_keys):
            with suppress(Exception):
                self._temp_file_store.delete(storage_key)


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
        knowledge_file_image_asset_repository: KnowledgeFileImageAssetRepository,
        chunk_service: DocumentChunkService,
        chunk_store: ChunkSearchStore,
        chunk_embedding_service: ChunkEmbeddingService,
        image_understanding_service: DocumentImageUnderstandingService,
        lease_seconds: int,
        heartbeat_interval_seconds: float,
    ) -> None:
        """初始化后台处理器。"""
        self._temp_file_store = temp_file_store
        self._object_store = object_store
        self._final_object_prefix = final_object_prefix.strip().strip("/")
        self._task_repository = task_repository
        self._knowledge_file_repository = knowledge_file_repository
        self._knowledge_file_image_asset_repository = knowledge_file_image_asset_repository
        self._chunk_service = chunk_service
        self._chunk_store = chunk_store
        self._chunk_embedding_service = chunk_embedding_service
        self._image_understanding_service = image_understanding_service
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

        local_file_path = self._temp_file_store.resolve_path(task.source_storage_key)
        should_cleanup_source_file = False

        try:
            self._task_repository.update_task_progress(
                task.id,
                worker_id=worker_id,
                stage=KnowledgeUploadTaskStage.PARSING,
            )
            if not local_file_path.exists():
                raise KnowledgeUploadTaskSourceMissingError(
                    f"上传任务源文件不存在: {task.source_storage_key}"
                )
            preview_chunks = self._chunk_service.chunk_document(
                file_path=local_file_path,
                source_filename=task.requested_filename,
                storage_key=task.source_storage_key,
                file_id=task.id,
            )
            prepared_segment_image_assets = self._prepare_segment_image_assets(preview_chunks)
            text_sha256 = self._build_text_sha256(preview_chunks)
            content_sha256 = self._build_content_sha256(
                preview_chunks,
                prepared_segment_image_assets,
            )
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
                text_sha256=text_sha256,
                content_sha256=content_sha256,
                preview_chunks=preview_chunks,
                prepared_segment_image_assets=prepared_segment_image_assets,
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
            should_cleanup_source_file = True
            for cleanup_file_id in process_result.cleanup_file_ids:
                with suppress(Exception):
                    self._chunk_store.delete_chunks_by_file_id(cleanup_file_id)
                with suppress(Exception):
                    self._knowledge_file_image_asset_repository.delete_assets_by_file_id(
                        cleanup_file_id
                    )
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
            if should_cleanup_source_file:
                with suppress(Exception):
                    self._temp_file_store.delete(task.source_storage_key)

    def _resolve_task_result(
        self,
        *,
        task: KnowledgeUploadTask,
        worker_id: str,
        text_sha256: str,
        content_sha256: str,
        preview_chunks: list[DocumentChunk],
        prepared_segment_image_assets: dict[str, list[PreparedChunkImageAsset]],
        local_file_path: Path,
    ) -> UploadTaskProcessResult:
        """根据正文哈希、图片稳定语义和文件名综合决定最终处理结果。"""
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
        existing_same_text = self._knowledge_file_repository.get_file_by_user_and_text_sha256(
            latest_task.uploader_user_id,
            text_sha256,
        )
        existing_same_name_assets = (
            self._knowledge_file_image_asset_repository.list_assets_by_file_id(
                existing_same_name.id
            )
            if existing_same_name is not None
            else []
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
            cleanup_file_ids: list[str] = []
            cleanup_storage_keys: list[str] = []
            if existing_same_name is not None and existing_same_name.id != existing_same_content.id:
                self._knowledge_file_repository.delete_file(existing_same_name.id)
                self._knowledge_file_image_asset_repository.delete_assets_by_file_id(
                    existing_same_name.id
                )
                cleanup_file_ids.append(existing_same_name.id)
                cleanup_storage_keys.extend(
                    [
                        existing_same_name.storage_key,
                        *self._collect_asset_storage_keys(existing_same_name_assets),
                    ]
                )
                title_updated = True
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
                cleanup_file_ids=cleanup_file_ids,
                cleanup_storage_keys=cleanup_storage_keys,
            )

        replacement_target = existing_same_name
        replacement_target_assets = existing_same_name_assets
        if replacement_target is None and existing_same_text is not None:
            replacement_target = existing_same_text
            replacement_target_assets = (
                self._knowledge_file_image_asset_repository.list_assets_by_file_id(
                    existing_same_text.id
                )
            )

        candidate_file = self._build_knowledge_file(
            task=latest_task,
            original_filename=requested_filename,
            text_sha256=text_sha256,
            content_sha256=content_sha256,
        )
        uploaded_final_object = False
        indexed = False
        uploaded_image_storage_keys: list[str] = []
        persisted_assets_by_segment: dict[str, list[KnowledgeFileImageAsset]] = {}
        try:
            # 原始上传文件直接在本地源目录完成解析，OSS 只承接最终知识文件对象，
            # 这样既避免“先上传再回下载”的带宽往返，也能保持检索与审计对象键稳定。
            self._object_store.upload_file(
                local_path=local_file_path,
                storage_key=candidate_file.storage_key,
            )
            uploaded_final_object = True
            persisted_assets_by_segment, uploaded_image_storage_keys = self._persist_image_assets(
                knowledge_file=candidate_file,
                preview_chunks=preview_chunks,
                prepared_assets_by_segment=prepared_segment_image_assets,
            )
            chunks = self._materialize_chunks(
                preview_chunks=preview_chunks,
                knowledge_file=candidate_file,
                persisted_assets_by_segment=persisted_assets_by_segment,
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
            if replacement_target is not None:
                persisted_file = self._knowledge_file_repository.replace_file(
                    replacement_target.id,
                    replace(candidate_file, chunk_count=len(chunks)),
                )
                self._knowledge_file_image_asset_repository.delete_assets_by_file_id(
                    replacement_target.id
                )
                return UploadTaskProcessResult(
                    file_id=persisted_file.id,
                    content_sha256=content_sha256,
                    chunk_count=len(chunks),
                    deduplicated=False,
                    replaced=True,
                    title_updated=False,
                    cleanup_file_ids=[replacement_target.id],
                    cleanup_storage_keys=[
                        replacement_target.storage_key,
                        *self._collect_asset_storage_keys(replacement_target_assets),
                    ],
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
            for storage_key in uploaded_image_storage_keys:
                with suppress(Exception):
                    self._object_store.delete(storage_key)
            with suppress(Exception):
                self._knowledge_file_image_asset_repository.delete_assets_by_file_id(
                    candidate_file.id
                )
            return self._resolve_conflict_after_index(
                latest_task=latest_task,
                requested_filename=requested_filename,
                text_sha256=text_sha256,
                content_sha256=content_sha256,
            )
        except Exception:
            if indexed:
                with suppress(Exception):
                    self._chunk_store.delete_chunks_by_file_id(candidate_file.id)
            if uploaded_final_object:
                with suppress(Exception):
                    self._object_store.delete(candidate_file.storage_key)
            for storage_key in uploaded_image_storage_keys:
                with suppress(Exception):
                    self._object_store.delete(storage_key)
            with suppress(Exception):
                self._knowledge_file_image_asset_repository.delete_assets_by_file_id(
                    candidate_file.id
                )
            raise

    def _resolve_conflict_after_index(
        self,
        *,
        latest_task: KnowledgeUploadTask,
        requested_filename: str,
        text_sha256: str,
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
        existing_same_text = self._knowledge_file_repository.get_file_by_user_and_text_sha256(
            latest_task.uploader_user_id,
            text_sha256,
        )
        existing_same_name_assets = (
            self._knowledge_file_image_asset_repository.list_assets_by_file_id(
                existing_same_name.id
            )
            if existing_same_name is not None
            else []
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
                self._knowledge_file_image_asset_repository.delete_assets_by_file_id(
                    existing_same_name.id
                )
                cleanup_file_ids.append(existing_same_name.id)
            return UploadTaskProcessResult(
                file_id=existing_same_content.id,
                content_sha256=content_sha256,
                chunk_count=existing_same_content.chunk_count,
                deduplicated=True,
                replaced=False,
                title_updated=title_updated,
                cleanup_file_ids=cleanup_file_ids,
                cleanup_storage_keys=(
                    [
                        existing_same_name.storage_key,
                        *self._collect_asset_storage_keys(existing_same_name_assets),
                    ]
                    if existing_same_name is not None
                    and existing_same_name.id != existing_same_content.id
                    else []
                ),
            )

        if existing_same_text is not None:
            return UploadTaskProcessResult(
                file_id=existing_same_text.id,
                content_sha256=content_sha256,
                chunk_count=existing_same_text.chunk_count,
                deduplicated=False,
                replaced=True,
                title_updated=False,
                cleanup_file_ids=[],
                cleanup_storage_keys=[],
            )

        raise KnowledgeFileConflictError("上传任务收敛失败，请稍后重试")

    def _build_text_sha256(
        self,
        preview_chunks: list[DocumentChunk],
    ) -> str:
        """基于正文 chunk 计算稳定文本哈希。"""
        payload = [
            {
                "chunk_index": chunk.chunk_index,
                "segment_id": chunk.segment_id,
                "content": chunk.content,
            }
            for chunk in preview_chunks
        ]
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _build_content_sha256(
        self,
        preview_chunks: list[DocumentChunk],
        prepared_assets_by_segment: dict[str, list[PreparedChunkImageAsset]],
    ) -> str:
        """基于正文与图片稳定语义构造文件级哈希。"""
        payload = {
            "text_chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "segment_id": chunk.segment_id,
                    "content": chunk.content,
                }
                for chunk in preview_chunks
            ],
            "images": [
                {
                    "segment_id": segment_id,
                    "asset_index": prepared_asset.preview_asset.asset_index,
                    "source_anchor": prepared_asset.preview_asset.source_anchor,
                    "normalized_image_sha256": prepared_asset.analysis.normalized_image_sha256,
                    "image_type": prepared_asset.analysis.image_type,
                    "normalized_ocr_text": prepared_asset.analysis.normalized_ocr_text,
                }
                for segment_id, segment_assets in sorted(prepared_assets_by_segment.items())
                for prepared_asset in sorted(
                    segment_assets,
                    key=lambda item: item.preview_asset.asset_index,
                )
            ],
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _prepare_segment_image_assets(
        self,
        preview_chunks: list[DocumentChunk],
    ) -> dict[str, list[PreparedChunkImageAsset]]:
        """按 segment_id 收集图片并完成规范化识别。"""
        prepared_assets_by_segment: dict[str, list[PreparedChunkImageAsset]] = {}
        seen_source_anchors: set[str] = set()

        for chunk in preview_chunks:
            for asset in chunk.image_assets:
                if asset.image_bytes is None or asset.source_anchor in seen_source_anchors:
                    continue
                analysis = self._image_understanding_service.analyze_image(
                    image_bytes=asset.image_bytes,
                    content_type=asset.content_type,
                )
                prepared_assets_by_segment.setdefault(asset.segment_id, []).append(
                    PreparedChunkImageAsset(
                        preview_asset=asset,
                        analysis=analysis,
                    )
                )
                seen_source_anchors.add(asset.source_anchor)

        return prepared_assets_by_segment

    def _collect_asset_storage_keys(
        self,
        assets: list[KnowledgeFileImageAsset],
    ) -> list[str]:
        """收集图片资产关联的存储对象键。"""
        storage_keys: list[str] = []
        for asset in assets:
            for storage_key in (asset.storage_key, asset.thumbnail_storage_key):
                if not storage_key or storage_key in storage_keys:
                    continue
                storage_keys.append(storage_key)
        return storage_keys

    def _materialize_chunks(
        self,
        *,
        preview_chunks: list[DocumentChunk],
        knowledge_file: KnowledgeFile,
        persisted_assets_by_segment: dict[str, list[KnowledgeFileImageAsset]],
    ) -> list[DocumentChunk]:
        """把预览文本 chunk 和图片语义 chunk 一起物化为最终入库结果。"""
        text_chunks = [
            replace(
                chunk,
                file_id=knowledge_file.id,
                chunk_id=f"{knowledge_file.id}-chunk-{chunk.chunk_index}",
                source_filename=knowledge_file.original_filename,
                storage_key=knowledge_file.storage_key,
                uploader_user_id=knowledge_file.uploader_user_id,
                visibility_scope=knowledge_file.visibility_scope.value,
                image_asset_refs=self._build_materialized_image_asset_refs(
                    persisted_assets_by_segment.get(chunk.segment_id, []),
                ),
                image_assets=self._build_materialized_chunk_image_assets(
                    persisted_assets_by_segment.get(chunk.segment_id, []),
                ),
            )
            for chunk in preview_chunks
        ]
        semantic_chunks = self._build_image_semantic_chunks(
            knowledge_file=knowledge_file,
            persisted_assets_by_segment=persisted_assets_by_segment,
        )
        return self._chunk_embedding_service.embed_chunks(text_chunks + semantic_chunks)

    def _build_materialized_chunk_image_assets(
        self,
        assets: list[KnowledgeFileImageAsset],
    ) -> list[ChunkImageAsset]:
        """把已落库图片资产转换为可写入检索侧的 chunk 图片资产。"""
        return [
            ChunkImageAsset(
                segment_id=asset.segment_id,
                asset_id=asset.id,
                asset_index=asset.asset_index,
                source_anchor=asset.source_anchor,
                content_type=asset.content_type,
                extension=Path(asset.storage_key).suffix or ".bin",
                image_bytes=None,
                image_sha256=asset.image_sha256,
                normalized_image_sha256=asset.normalized_image_sha256,
                storage_key=asset.storage_key,
                thumbnail_storage_key=asset.thumbnail_storage_key,
                width=asset.width,
                height=asset.height,
                ocr_text=asset.ocr_text,
                summary=asset.summary,
                image_type=asset.image_type,
                recognition_model=asset.recognition_model,
            )
            for asset in assets
        ]

    def _build_materialized_image_asset_refs(
        self,
        assets: list[KnowledgeFileImageAsset],
    ) -> list[ChunkImageAssetRef]:
        """把已落库图片资产转换为 chunk 引用键列表。"""
        return [
            ChunkImageAssetRef(
                segment_id=asset.segment_id,
                asset_id=asset.id,
            )
            for asset in assets
        ]

    def _build_image_semantic_chunks(
        self,
        *,
        knowledge_file: KnowledgeFile,
        persisted_assets_by_segment: dict[str, list[KnowledgeFileImageAsset]],
    ) -> list[DocumentChunk]:
        """基于图片资产构造独立的图片语义 chunk。"""
        semantic_chunks: list[DocumentChunk] = []
        for segment_id, assets in sorted(persisted_assets_by_segment.items()):
            for asset in sorted(assets, key=lambda item: item.asset_index):
                semantic_content = self._build_image_semantic_content(asset)
                chunk_index = self._parse_chunk_index_from_chunk_id(asset.semantic_chunk_id)
                semantic_chunks.append(
                    DocumentChunk(
                        file_id=knowledge_file.id,
                        chunk_id=asset.semantic_chunk_id,
                        chunk_index=chunk_index,
                        chunk_type=ChunkType.IMAGE_SEMANTIC.value,
                        segment_id=segment_id,
                        content=semantic_content,
                        char_count=len(semantic_content),
                        source_filename=knowledge_file.original_filename,
                        storage_key=knowledge_file.storage_key,
                        uploader_user_id=knowledge_file.uploader_user_id,
                        visibility_scope=knowledge_file.visibility_scope.value,
                        merged_terms=[],
                        image_asset_refs=[
                            ChunkImageAssetRef(
                                segment_id=segment_id,
                                asset_id=asset.id,
                            )
                        ],
                        image_assets=self._build_materialized_chunk_image_assets([asset]),
                    )
                )
        return semantic_chunks

    def _build_image_semantic_content(
        self,
        asset: KnowledgeFileImageAsset,
    ) -> str:
        """为单张图片构造独立图片语义 chunk 正文。"""
        semantic_lines = [f"图片类型：{asset.image_type.strip() or 'unknown'}"]
        if asset.summary.strip():
            semantic_lines.append(f"图片摘要：{asset.summary.strip()}")
        if asset.ocr_text.strip():
            semantic_lines.append(f"图片文本：{asset.ocr_text.strip()}")
        return "\n".join(semantic_lines)

    def _parse_chunk_index_from_chunk_id(self, chunk_id: str) -> int:
        """从标准 chunk_id 中解析 chunk_index。"""
        try:
            return int(chunk_id.rsplit("-", maxsplit=1)[1])
        except (IndexError, ValueError):
            msg = f"非法 chunk_id，无法解析 chunk_index: {chunk_id}"
            raise KnowledgeFileConflictError(msg) from None

    def _build_knowledge_file(
        self,
        *,
        task: KnowledgeUploadTask,
        original_filename: str,
        text_sha256: str,
        content_sha256: str,
        file_id: str | None = None,
    ) -> KnowledgeFile:
        """基于任务信息构造待落库文件元数据。"""
        uploaded_at = datetime.now(UTC)
        resolved_file_id = file_id or uuid4().hex
        return KnowledgeFile(
            id=resolved_file_id,
            uploader_user_id=task.uploader_user_id,
            original_filename=original_filename,
            content_type=task.content_type,
            size=task.size,
            storage_provider=FileStorageProvider.ALIYUN_OSS,
            storage_key=self._build_storage_key(
                uploader_user_id=task.uploader_user_id,
                file_id=resolved_file_id,
                safe_filename=FileUploadService.sanitize_filename(original_filename),
            ),
            visibility_scope=self._resolve_visibility_scope(task),
            chunk_count=0,
            uploaded_at=uploaded_at,
            updated_at=uploaded_at,
            raw_sha256=task.raw_sha256,
            text_sha256=text_sha256,
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

    def _persist_image_assets(
        self,
        *,
        knowledge_file: KnowledgeFile,
        preview_chunks: list[DocumentChunk],
        prepared_assets_by_segment: dict[str, list[PreparedChunkImageAsset]],
    ) -> tuple[dict[str, list[KnowledgeFileImageAsset]], list[str]]:
        """上传图片资源并批量持久化图片资产。"""
        now = datetime.now(UTC)
        persisted_assets: list[KnowledgeFileImageAsset] = []
        uploaded_storage_keys: list[str] = []
        segment_order = list(
            dict.fromkeys(
                chunk.segment_id
                for chunk in preview_chunks
                if chunk.segment_id in prepared_assets_by_segment
            )
        )
        next_semantic_chunk_index = len(preview_chunks)

        for segment_id in segment_order:
            for prepared_asset in sorted(
                prepared_assets_by_segment.get(segment_id, []),
                key=lambda item: item.preview_asset.asset_index,
            ):
                asset_id = uuid4().hex
                semantic_chunk_id = f"{knowledge_file.id}-chunk-{next_semantic_chunk_index}"
                next_semantic_chunk_index += 1
                original_storage_key = self._build_image_storage_key(
                    uploader_user_id=knowledge_file.uploader_user_id,
                    file_id=knowledge_file.id,
                    asset_id=asset_id,
                    folder_name="images",
                    extension=prepared_asset.preview_asset.extension,
                )
                thumbnail_storage_key = self._build_image_storage_key(
                    uploader_user_id=knowledge_file.uploader_user_id,
                    file_id=knowledge_file.id,
                    asset_id=asset_id,
                    folder_name="thumbnails",
                    extension=".png",
                )
                self._upload_bytes_to_object_store(
                    payload=prepared_asset.preview_asset.image_bytes or b"",
                    storage_key=original_storage_key,
                    suffix=prepared_asset.preview_asset.extension,
                )
                uploaded_storage_keys.append(original_storage_key)
                self._upload_bytes_to_object_store(
                    payload=prepared_asset.analysis.normalized_image_bytes,
                    storage_key=thumbnail_storage_key,
                    suffix=".png",
                )
                uploaded_storage_keys.append(thumbnail_storage_key)
                persisted_assets.append(
                    KnowledgeFileImageAsset(
                        id=asset_id,
                        file_id=knowledge_file.id,
                        segment_id=segment_id,
                        semantic_chunk_id=semantic_chunk_id,
                        asset_index=prepared_asset.preview_asset.asset_index,
                        uploader_user_id=knowledge_file.uploader_user_id,
                        source_anchor=prepared_asset.preview_asset.source_anchor,
                        image_sha256=prepared_asset.analysis.image_sha256,
                        normalized_image_sha256=prepared_asset.analysis.normalized_image_sha256,
                        storage_key=original_storage_key,
                        thumbnail_storage_key=thumbnail_storage_key,
                        content_type=prepared_asset.preview_asset.content_type,
                        width=prepared_asset.analysis.width,
                        height=prepared_asset.analysis.height,
                        ocr_text=prepared_asset.analysis.ocr_text,
                        summary=prepared_asset.analysis.summary,
                        image_type=prepared_asset.analysis.image_type,
                        recognition_model=prepared_asset.analysis.recognition_model,
                        created_at=now,
                        updated_at=now,
                    )
                )

        created_assets = self._knowledge_file_image_asset_repository.create_assets(persisted_assets)
        assets_by_segment: dict[str, list[KnowledgeFileImageAsset]] = {}
        for asset in created_assets:
            assets_by_segment.setdefault(asset.segment_id, []).append(asset)
        return assets_by_segment, uploaded_storage_keys

    def _build_image_storage_key(
        self,
        *,
        uploader_user_id: str,
        file_id: str,
        asset_id: str,
        folder_name: str,
        extension: str,
    ) -> str:
        """构造图片资源的 OSS 对象键。"""
        normalized_extension = extension if extension.startswith(".") else f".{extension}"
        parts = [
            part
            for part in [
                self._final_object_prefix,
                uploader_user_id,
                file_id,
                "assets",
                folder_name,
                f"{asset_id}{normalized_extension}",
            ]
            if part
        ]
        return "/".join(parts)

    def _upload_bytes_to_object_store(
        self,
        *,
        payload: bytes,
        storage_key: str,
        suffix: str,
    ) -> None:
        """把内存中的图片字节临时落盘后上传到对象存储。"""
        if not payload:
            msg = f"图片资源内容为空，无法上传: {storage_key}"
            raise KnowledgeFileConflictError(msg)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(payload)
            temp_path = Path(temp_file.name)
        try:
            self._object_store.upload_file(local_path=temp_path, storage_key=storage_key)
        finally:
            with suppress(OSError):
                temp_path.unlink()

    def _resolve_visibility_scope(self, task: KnowledgeUploadTask) -> FileVisibilityScope:
        """根据上传者角色决定文件可见性。"""
        if task.uploader_role == UserRole.ADMIN.value:
            return FileVisibilityScope.GLOBAL
        return FileVisibilityScope.OWNER_ONLY

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
