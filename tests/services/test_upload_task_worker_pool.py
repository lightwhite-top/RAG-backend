from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from baozhi_rag.domain.knowledge_upload_task import (
    KnowledgeUploadTask,
    KnowledgeUploadTaskStage,
    KnowledgeUploadTaskStatus,
)
from baozhi_rag.services.document_chunking import ChunkImageAsset, DocumentChunk
from baozhi_rag.services.document_image_understanding import DocumentImageUnderstandingResult
from baozhi_rag.services.document_ocr.ocr_capacity_limiter import (
    LocalOcrTaskLimiter,
    OcrCapacityPendingError,
)
from baozhi_rag.services.upload_tasks import (
    KnowledgeUploadProcessor,
    KnowledgeUploadWorker,
    UploadTaskProcessResult,
)


class _BlockingProcessor:
    """用于测试线程池并行度的假处理器。"""

    def __init__(self, *, task_count: int, task_duration_seconds: float) -> None:
        self._remaining = task_count
        self._task_duration_seconds = task_duration_seconds
        self._lock = threading.Lock()
        self.active_count = 0
        self.max_active_count = 0
        self.completed_count = 0

    def process_next_task(self, worker_id: str) -> bool:
        """模拟可并行执行的任务处理。"""
        del worker_id
        with self._lock:
            if self._remaining <= 0:
                return False
            self._remaining -= 1
            self.active_count += 1
            self.max_active_count = max(self.max_active_count, self.active_count)

        try:
            time.sleep(self._task_duration_seconds)
        finally:
            with self._lock:
                self.active_count -= 1
                self.completed_count += 1
        return True


class _FakeTempFileStore:
    """用于测试处理器回队逻辑的假文件存储。"""

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path

    def resolve_path(self, storage_key: str) -> Path:
        """返回固定的本地文件路径。"""
        del storage_key
        return self._file_path

    def delete(self, storage_key: str) -> None:
        """兼容处理器调用的删除接口。"""
        del storage_key


class _FakeTaskRepository:
    """用于测试 OCR 回队分支的假任务仓储。"""

    def __init__(self) -> None:
        self.requeued_retry_at: datetime | None = None
        self.failed_error_code: str | None = None
        self.succeeded_task_id: str | None = None
        self.progress_updates: list[KnowledgeUploadTaskStage] = []

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
    ) -> None:
        """记录阶段推进。"""
        del task_id, worker_id, status, content_sha256, file_id, chunk_count
        del deduplicated, replaced, title_updated
        self.progress_updates.append(stage)
        return None

    def requeue_waiting_for_ocr_capacity(
        self,
        task_id: str,
        *,
        worker_id: str,
        retry_at: datetime,
    ) -> None:
        """记录 OCR 容量不足时的回队动作。"""
        del task_id, worker_id
        self.requeued_retry_at = retry_at
        return None

    def mark_failed(
        self,
        task_id: str,
        *,
        worker_id: str,
        error_code: str,
        error_message: str,
        failed_at: datetime,
    ) -> None:
        """记录失败标记。"""
        del task_id, worker_id, error_message, failed_at
        self.failed_error_code = error_code
        return None

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
    ) -> None:
        """记录成功标记，便于断言是否出现假成功。"""
        del worker_id, stage, content_sha256, file_id, chunk_count
        del deduplicated, replaced, title_updated, completed_at
        self.succeeded_task_id = task_id
        return None

    def refresh_lease(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime,
        heartbeat_at: datetime,
    ) -> bool:
        """兼容心跳线程调用。"""
        del task_id, worker_id, lease_expires_at, heartbeat_at
        return True

    def get_task_by_id(self, task_id: str) -> None:
        """兼容处理器调用。"""
        del task_id
        return None


class _RaisingChunkService:
    """用于模拟 OCR 容量不足的假切块服务。"""

    def chunk_document(
        self,
        *,
        file_path: Path,
        source_filename: str,
        storage_key: str,
        file_id: str,
    ) -> list[object]:
        """始终抛出 OCR 容量不足异常。"""
        del file_path, source_filename, storage_key, file_id
        raise OcrCapacityPendingError("OCR 槽位等待超时", retry_after_seconds=0.2)


class _EmptyChunkService:
    """用于测试最小成功路径的空切块服务。"""

    def chunk_document(
        self,
        *,
        file_path: Path,
        source_filename: str,
        storage_key: str,
        file_id: str,
    ) -> list[DocumentChunk]:
        """返回空切块列表，让测试聚焦于收口逻辑。"""
        del file_path, source_filename, storage_key, file_id
        return []


class _NoOpImageAssetRepository:
    """用于测试替换清理路径的最小图片资产仓储。"""

    def __init__(self) -> None:
        self.deleted_file_ids: list[str] = []

    def delete_assets_by_file_id(self, file_id: str) -> int:
        """兼容处理器在旧文件清理后删除图片资产。"""
        self.deleted_file_ids.append(file_id)
        return 0


class _FailingCleanupChunkStore:
    """用于模拟旧索引清理失败，验证任务不会假成功。"""

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """删除旧文件索引时始终失败。"""
        raise RuntimeError(f"cleanup failed for {file_id}")


class _SuccessfulCleanupChunkStore:
    """用于验证旧索引清理成功后的收口顺序。"""

    def __init__(self) -> None:
        self.deleted_file_ids: list[str] = []

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """记录旧索引清理动作。"""
        self.deleted_file_ids.append(file_id)


class _TrackingKnowledgeFileRepository:
    """用于验证旧文件元数据在成功收口前被清理。"""

    def __init__(self) -> None:
        self.deleted_file_ids: list[str] = []

    def delete_file(self, file_id: str) -> bool:
        """记录旧文件元数据删除动作。"""
        self.deleted_file_ids.append(file_id)
        return True


class _UnusedDependency:
    """占位依赖，避免为本次测试构造无关对象。"""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"不应访问无关依赖: {name}")


class _FallbackImageUnderstandingService:
    """用于测试图片理解失败后的降级分支。"""

    def analyze_image(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
    ) -> DocumentImageUnderstandingResult:
        del image_bytes, content_type
        raise RuntimeError("image model timeout")

    def build_fallback_result(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        fallback_text: str = "",
    ) -> DocumentImageUnderstandingResult:
        del image_bytes, content_type
        return DocumentImageUnderstandingResult(
            image_sha256="raw-sha",
            normalized_image_sha256="normalized-sha",
            normalized_image_bytes=b"normalized-image",
            width=10,
            height=10,
            ocr_text=fallback_text,
            normalized_ocr_text=fallback_text,
            summary="",
            image_type="unknown",
            content_type="image/png",
            recognition_model="fallback",
        )


def _build_task() -> KnowledgeUploadTask:
    """构造一条已被 claim 的上传任务。"""
    now = datetime.now(UTC)
    return KnowledgeUploadTask(
        id="task-1",
        request_id="req-1",
        uploader_user_id="user-1",
        uploader_role="admin",
        raw_sha256="sha-raw",
        source_storage_key="tmp/source.pdf",
        requested_filename="source.pdf",
        content_type="application/pdf",
        size=123,
        ingest_version="v1",
        status=KnowledgeUploadTaskStatus.PROCESSING,
        stage=KnowledgeUploadTaskStage.UPLOADED,
        content_sha256=None,
        file_id=None,
        chunk_count=0,
        deduplicated=False,
        replaced=False,
        title_updated=False,
        error_code=None,
        error_message=None,
        attempt_count=1,
        worker_id="worker-1",
        lease_expires_at=now,
        last_heartbeat_at=now,
        created_at=now,
        updated_at=now,
        completed_at=None,
    )


def test_local_ocr_task_limiter_rejects_when_wait_queue_full() -> None:
    """当 OCR 等待队列已满时，应立即拒绝新的等待者。"""
    limiter = LocalOcrTaskLimiter(max_concurrent=1, max_waiters=1, wait_timeout_seconds=0.2)
    holder_started = threading.Event()
    release_holder = threading.Event()
    waiter_started = threading.Event()

    def hold_slot() -> None:
        with limiter.reserve_slot(page_number=1):
            holder_started.set()
            release_holder.wait(timeout=2.0)

    def wait_slot() -> None:
        waiter_started.set()
        try:
            with limiter.reserve_slot(page_number=2):
                return
        except OcrCapacityPendingError:
            return

    holder_thread = threading.Thread(target=hold_slot)
    waiter_thread = threading.Thread(target=wait_slot)
    holder_thread.start()
    assert holder_started.wait(timeout=1.0)
    waiter_thread.start()
    assert waiter_started.wait(timeout=1.0)
    time.sleep(0.05)

    try:
        with limiter.reserve_slot(page_number=3):
            raise AssertionError("等待队列已满时不应成功获取 OCR 槽位")
    except OcrCapacityPendingError as exc:
        assert "等待队列已满" in exc.message
    finally:
        release_holder.set()
        holder_thread.join(timeout=1.0)
        waiter_thread.join(timeout=1.0)


def test_upload_worker_uses_thread_pool_max_parallelism() -> None:
    """线程池在有持续工作时应扩展到最大并行度。"""
    processor = _BlockingProcessor(task_count=8, task_duration_seconds=0.25)

    async def _run_worker() -> None:
        worker = KnowledgeUploadWorker(
            processor=processor,
            worker_id="pool-worker",
            poll_interval_seconds=0.02,
            thread_pool_core_size=2,
            thread_pool_max_size=4,
        )

        async def _stop_when_done() -> None:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if processor.completed_count >= 8 and processor.active_count == 0:
                    worker.stop()
                    return
                await asyncio.sleep(0.02)
            raise AssertionError("线程池测试超时，任务未按预期完成")

        await asyncio.gather(worker.run(), _stop_when_done())

    asyncio.run(_run_worker())
    assert processor.max_active_count == 4


def test_process_claimed_task_requeues_when_ocr_capacity_pending(tmp_path: Path) -> None:
    """OCR 容量不足时，任务应回队而不是标记失败。"""
    source_file = tmp_path / "source.pdf"
    source_file.write_bytes(b"fake-pdf")
    repository = _FakeTaskRepository()
    processor = KnowledgeUploadProcessor(
        temp_file_store=_FakeTempFileStore(source_file),
        object_store=_UnusedDependency(),
        final_object_prefix="knowledge-files",
        task_repository=repository,
        knowledge_file_repository=_UnusedDependency(),
        knowledge_file_image_asset_repository=_UnusedDependency(),
        chunk_service=_RaisingChunkService(),
        chunk_store=_UnusedDependency(),
        chunk_embedding_service=_UnusedDependency(),
        image_understanding_service=_UnusedDependency(),
        lease_seconds=30,
        heartbeat_interval_seconds=5.0,
        ocr_capacity_retry_delay_seconds=0.2,
    )

    processor._process_claimed_task(task=_build_task(), worker_id="worker-1")

    assert repository.progress_updates == [KnowledgeUploadTaskStage.PARSING]
    assert repository.requeued_retry_at is not None
    assert repository.failed_error_code is None


def test_prepare_segment_image_assets_falls_back_when_image_model_fails() -> None:
    """图片理解模型失败时，应保留图片资产并回退到段落文本。"""
    processor = KnowledgeUploadProcessor(
        temp_file_store=_UnusedDependency(),
        object_store=_UnusedDependency(),
        final_object_prefix="knowledge-files",
        task_repository=_UnusedDependency(),
        knowledge_file_repository=_UnusedDependency(),
        knowledge_file_image_asset_repository=_UnusedDependency(),
        chunk_service=_UnusedDependency(),
        chunk_store=_UnusedDependency(),
        chunk_embedding_service=_UnusedDependency(),
        image_understanding_service=_FallbackImageUnderstandingService(),  # type: ignore[arg-type]
        lease_seconds=30,
        heartbeat_interval_seconds=5.0,
        ocr_capacity_retry_delay_seconds=0.2,
    )
    preview_chunks = [
        DocumentChunk(
            file_id="file-1",
            chunk_id="file-1-chunk-0",
            chunk_index=0,
            chunk_type="text",
            segment_id="seg-1",
            content="影像补传截止时点：报销提单后24小时。",
            char_count=20,
            source_filename="image.docx",
            storage_key="stage/image.docx",
            image_assets=[
                ChunkImageAsset(
                    segment_id="seg-1",
                    asset_id="img-1",
                    asset_index=1,
                    source_anchor="seg-1:image:1",
                    content_type="image/png",
                    extension=".png",
                    image_bytes=b"fake-image",
                )
            ],
        )
    ]

    prepared_assets = processor._prepare_segment_image_assets(preview_chunks)  # pyright: ignore[reportPrivateUsage]

    assert "seg-1" in prepared_assets
    assert prepared_assets["seg-1"][0].analysis.ocr_text == "影像补传截止时点：报销提单后24小时。"
    assert prepared_assets["seg-1"][0].analysis.recognition_model == "fallback"


def test_process_claimed_task_does_not_mark_success_when_superseded_chunk_cleanup_fails(
    tmp_path: Path,
) -> None:
    """旧索引清理失败时，任务不能先被标记为成功。"""

    class _CleanupFailureProcessor(KnowledgeUploadProcessor):
        def _resolve_task_result(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            *,
            task: KnowledgeUploadTask,
            worker_id: str,
            text_sha256: str,
            content_sha256: str,
            preview_chunks: list[DocumentChunk],
            prepared_segment_image_assets: dict[str, list[object]],
            local_file_path: Path,
        ) -> UploadTaskProcessResult:
            del task, worker_id, text_sha256, content_sha256
            del preview_chunks, prepared_segment_image_assets, local_file_path
            return UploadTaskProcessResult(
                file_id="new-file",
                content_sha256="content-sha",
                chunk_count=1,
                deduplicated=False,
                replaced=True,
                title_updated=False,
                cleanup_file_ids=["old-file"],
                cleanup_storage_keys=[],
            )

    source_file = tmp_path / "source.pdf"
    source_file.write_bytes(b"fake-pdf")
    repository = _FakeTaskRepository()
    processor = _CleanupFailureProcessor(
        temp_file_store=_FakeTempFileStore(source_file),
        object_store=_UnusedDependency(),
        final_object_prefix="knowledge-files",
        task_repository=repository,
        knowledge_file_repository=_UnusedDependency(),
        knowledge_file_image_asset_repository=_NoOpImageAssetRepository(),  # type: ignore[arg-type]
        chunk_service=_EmptyChunkService(),  # type: ignore[arg-type]
        chunk_store=_FailingCleanupChunkStore(),  # type: ignore[arg-type]
        chunk_embedding_service=_UnusedDependency(),
        image_understanding_service=_UnusedDependency(),
        lease_seconds=30,
        heartbeat_interval_seconds=5.0,
        ocr_capacity_retry_delay_seconds=0.2,
    )

    processor._process_claimed_task(task=_build_task(), worker_id="worker-1")

    assert repository.succeeded_task_id is None
    assert repository.failed_error_code == "knowledge_upload_task_failed"


def test_process_claimed_task_cleans_superseded_metadata_before_marking_success(
    tmp_path: Path,
) -> None:
    """旧索引清理成功后，应继续清掉旧元数据再标记成功。"""

    class _CleanupSuccessProcessor(KnowledgeUploadProcessor):
        def _resolve_task_result(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            *,
            task: KnowledgeUploadTask,
            worker_id: str,
            text_sha256: str,
            content_sha256: str,
            preview_chunks: list[DocumentChunk],
            prepared_segment_image_assets: dict[str, list[object]],
            local_file_path: Path,
        ) -> UploadTaskProcessResult:
            del task, worker_id, text_sha256, content_sha256
            del preview_chunks, prepared_segment_image_assets, local_file_path
            return UploadTaskProcessResult(
                file_id="new-file",
                content_sha256="content-sha",
                chunk_count=1,
                deduplicated=False,
                replaced=True,
                title_updated=False,
                cleanup_file_ids=["old-file"],
                cleanup_storage_keys=[],
            )

    source_file = tmp_path / "source.pdf"
    source_file.write_bytes(b"fake-pdf")
    repository = _FakeTaskRepository()
    knowledge_file_repository = _TrackingKnowledgeFileRepository()
    image_asset_repository = _NoOpImageAssetRepository()
    chunk_store = _SuccessfulCleanupChunkStore()
    processor = _CleanupSuccessProcessor(
        temp_file_store=_FakeTempFileStore(source_file),
        object_store=_UnusedDependency(),
        final_object_prefix="knowledge-files",
        task_repository=repository,
        knowledge_file_repository=knowledge_file_repository,  # type: ignore[arg-type]
        knowledge_file_image_asset_repository=image_asset_repository,  # type: ignore[arg-type]
        chunk_service=_EmptyChunkService(),  # type: ignore[arg-type]
        chunk_store=chunk_store,  # type: ignore[arg-type]
        chunk_embedding_service=_UnusedDependency(),
        image_understanding_service=_UnusedDependency(),
        lease_seconds=30,
        heartbeat_interval_seconds=5.0,
        ocr_capacity_retry_delay_seconds=0.2,
    )

    processor._process_claimed_task(task=_build_task(), worker_id="worker-1")

    assert chunk_store.deleted_file_ids == ["old-file"]
    assert image_asset_repository.deleted_file_ids == ["old-file"]
    assert knowledge_file_repository.deleted_file_ids == ["old-file"]
    assert repository.failed_error_code is None
    assert repository.succeeded_task_id == "task-1"
