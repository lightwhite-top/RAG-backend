from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from baozhi_rag.domain.knowledge_file import (
    FileStorageProvider,
    FileVisibilityScope,
    KnowledgeFile,
    KnowledgeFileListPage,
)
from baozhi_rag.domain.knowledge_file_image_asset import KnowledgeFileImageAsset
from baozhi_rag.domain.knowledge_upload_task import (
    KnowledgeUploadTask,
    KnowledgeUploadTaskStage,
    KnowledgeUploadTaskStatus,
)
from baozhi_rag.domain.user import CurrentUser, UserRole
from baozhi_rag.services.knowledge_file_delete import (
    ChatRecordPurgeResult,
    KnowledgeFileDeleteCleanupError,
    KnowledgeFileDeleteService,
)


def _build_current_user() -> CurrentUser:
    now = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
    return CurrentUser(
        id="user-1",
        email="user@example.com",
        username="Light",
        role=UserRole.USER,
        created_at=now,
        updated_at=now,
    )


def _build_file(*, file_id: str, storage_key: str) -> KnowledgeFile:
    now = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
    return KnowledgeFile(
        id=file_id,
        uploader_user_id="user-1",
        original_filename=f"{file_id}.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        size=128,
        storage_provider=FileStorageProvider.ALIYUN_OSS,
        storage_key=storage_key,
        visibility_scope=FileVisibilityScope.OWNER_ONLY,
        chunk_count=2,
        uploaded_at=now,
        updated_at=now,
        raw_sha256=f"raw-{file_id}",
        text_sha256=f"text-{file_id}",
        content_sha256=f"content-{file_id}",
    )


def _build_asset(*, file_id: str, asset_id: str) -> KnowledgeFileImageAsset:
    now = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
    return KnowledgeFileImageAsset(
        id=asset_id,
        file_id=file_id,
        segment_id="seg-1",
        semantic_chunk_id=f"{file_id}-chunk-9",
        asset_index=1,
        uploader_user_id="user-1",
        source_anchor="p:1:image:1",
        image_sha256=f"sha-{asset_id}",
        normalized_image_sha256=f"norm-{asset_id}",
        storage_key=f"knowledge-files/user-1/{file_id}/assets/images/{asset_id}.png",
        thumbnail_storage_key=f"knowledge-files/user-1/{file_id}/assets/thumbnails/{asset_id}.png",
        content_type="image/png",
        width=320,
        height=240,
        ocr_text="ocr",
        summary="summary",
        image_type="diagram",
        recognition_model="test-model",
        created_at=now,
        updated_at=now,
    )


def _build_task(*, task_id: str, storage_key: str) -> KnowledgeUploadTask:
    now = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
    return KnowledgeUploadTask(
        id=task_id,
        request_id=f"req-{task_id}",
        uploader_user_id="user-1",
        uploader_role=UserRole.USER.value,
        raw_sha256=f"raw-{task_id}",
        source_storage_key=storage_key,
        requested_filename=f"{task_id}.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        size=64,
        ingest_version="v1",
        status=KnowledgeUploadTaskStatus.SUCCEEDED,
        stage=KnowledgeUploadTaskStage.COMPLETED,
        content_sha256=f"content-{task_id}",
        file_id=None,
        chunk_count=0,
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
class _StubKnowledgeFileRepository:
    files: list[KnowledgeFile]
    deleted_file_ids: list[str] = field(default_factory=list)

    def get_file_by_id(self, file_id: str) -> KnowledgeFile | None:
        for item in self.files:
            if item.id == file_id:
                return item
        return None

    def list_user_files(
        self,
        *,
        uploader_user_id: str,
        page: int,
        page_size: int,
    ) -> KnowledgeFileListPage:
        items = [item for item in self.files if item.uploader_user_id == uploader_user_id]
        start = (page - 1) * page_size
        end = start + page_size
        return KnowledgeFileListPage(
            items=items[start:end],
            total=len(items),
            page=page,
            page_size=page_size,
        )

    def list_all_files(
        self,
        *,
        page: int,
        page_size: int,
    ) -> KnowledgeFileListPage:
        start = (page - 1) * page_size
        end = start + page_size
        return KnowledgeFileListPage(
            items=self.files[start:end],
            total=len(self.files),
            page=page,
            page_size=page_size,
        )

    def delete_file(self, file_id: str) -> bool:
        for index, item in enumerate(self.files):
            if item.id == file_id:
                self.deleted_file_ids.append(file_id)
                self.files.pop(index)
                return True
        return False


@dataclass
class _StubImageAssetRepository:
    assets_by_file_id: dict[str, list[KnowledgeFileImageAsset]]
    deleted_file_ids: list[str] = field(default_factory=list)

    def list_assets_by_file_id(self, file_id: str) -> list[KnowledgeFileImageAsset]:
        return list(self.assets_by_file_id.get(file_id, []))

    def delete_assets_by_file_id(self, file_id: str) -> int:
        self.deleted_file_ids.append(file_id)
        return len(self.assets_by_file_id.pop(file_id, []))


@dataclass
class _StubTaskRepository:
    tasks: list[KnowledgeUploadTask]
    deleted_user_ids: list[str] = field(default_factory=list)
    delete_all_called: bool = False

    def delete_tasks_by_user(self, uploader_user_id: str) -> list[KnowledgeUploadTask]:
        self.deleted_user_ids.append(uploader_user_id)
        matched = [task for task in self.tasks if task.uploader_user_id == uploader_user_id]
        self.tasks = [task for task in self.tasks if task.uploader_user_id != uploader_user_id]
        return matched

    def delete_all_tasks(self) -> list[KnowledgeUploadTask]:
        self.delete_all_called = True
        matched = list(self.tasks)
        self.tasks = []
        return matched


@dataclass
class _StubChatCleanupRepository:
    delete_all_called: bool = False
    deleted_user_ids: list[str] = field(default_factory=list)

    def delete_chat_records_by_user(self, owner_user_id: str) -> ChatRecordPurgeResult:
        self.deleted_user_ids.append(owner_user_id)
        return ChatRecordPurgeResult(
            deleted_session_count=1,
            deleted_message_count=2,
            deleted_snapshot_count=3,
        )

    def delete_all_chat_records(self) -> ChatRecordPurgeResult:
        self.delete_all_called = True
        return ChatRecordPurgeResult(
            deleted_session_count=4,
            deleted_message_count=5,
            deleted_snapshot_count=6,
        )


@dataclass
class _StubChunkStore:
    deleted_file_ids: list[str] = field(default_factory=list)
    failing_file_ids: set[str] = field(default_factory=set)

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        if file_id in self.failing_file_ids:
            raise RuntimeError(f"chunk cleanup failed: {file_id}")
        self.deleted_file_ids.append(file_id)


@dataclass
class _StubObjectStore:
    deleted_storage_keys: list[str] = field(default_factory=list)

    def delete(self, storage_key: str) -> None:
        self.deleted_storage_keys.append(storage_key)


@dataclass
class _StubTempFileStore:
    deleted_storage_keys: list[str] = field(default_factory=list)

    def delete(self, storage_key: str) -> None:
        self.deleted_storage_keys.append(storage_key)


def test_delete_all_files_removes_user_files_and_upload_tasks() -> None:
    file_one = _build_file(file_id="file-1", storage_key="knowledge-files/user-1/file-1/a.docx")
    file_two = _build_file(file_id="file-2", storage_key="knowledge-files/user-1/file-2/b.docx")
    asset_one = _build_asset(file_id="file-1", asset_id="img-1")
    task_one = _build_task(task_id="task-1", storage_key="temp/user-1/task-1.docx")
    task_two = _build_task(task_id="task-2", storage_key="temp/user-1/task-2.docx")

    knowledge_file_repository = _StubKnowledgeFileRepository(files=[file_one, file_two])
    image_asset_repository = _StubImageAssetRepository(
        assets_by_file_id={"file-1": [asset_one], "file-2": []}
    )
    task_repository = _StubTaskRepository(tasks=[task_one, task_two])
    chat_cleanup_repository = _StubChatCleanupRepository()
    chunk_store = _StubChunkStore()
    object_store = _StubObjectStore()
    temp_file_store = _StubTempFileStore()

    service = KnowledgeFileDeleteService(
        knowledge_file_repository=knowledge_file_repository,
        knowledge_file_image_asset_repository=image_asset_repository,
        chat_cleanup_repository=chat_cleanup_repository,
        task_repository=task_repository,
        chunk_store=chunk_store,
        object_store=object_store,
        temp_file_store=temp_file_store,
    )

    result = service.delete_all_files(current_user=_build_current_user())

    assert result.deleted_file_count == 2
    assert result.deleted_task_count == 2
    assert knowledge_file_repository.deleted_file_ids == ["file-1", "file-2"]
    assert image_asset_repository.deleted_file_ids == ["file-1", "file-2"]
    assert chat_cleanup_repository.delete_all_called is False
    assert chat_cleanup_repository.deleted_user_ids == ["user-1"]
    assert task_repository.deleted_user_ids == ["user-1"]
    assert temp_file_store.deleted_storage_keys == [
        "temp/user-1/task-1.docx",
        "temp/user-1/task-2.docx",
    ]
    assert chunk_store.deleted_file_ids == ["file-1", "file-2"]
    assert "knowledge-files/user-1/file-1/a.docx" in object_store.deleted_storage_keys
    assert asset_one.storage_key in object_store.deleted_storage_keys
    assert asset_one.thumbnail_storage_key in object_store.deleted_storage_keys


def test_delete_all_files_globally_removes_all_users_data() -> None:
    file_one = _build_file(file_id="file-1", storage_key="knowledge-files/user-1/file-1/a.docx")
    file_two = replace(
        _build_file(file_id="file-2", storage_key="knowledge-files/user-2/file-2/b.docx"),
        uploader_user_id="user-2",
    )
    task_one = _build_task(task_id="task-1", storage_key="temp/user-1/task-1.docx")
    task_two = replace(
        _build_task(task_id="task-2", storage_key="temp/user-2/task-2.docx"),
        uploader_user_id="user-2",
    )

    knowledge_file_repository = _StubKnowledgeFileRepository(files=[file_one, file_two])
    image_asset_repository = _StubImageAssetRepository(
        assets_by_file_id={"file-1": [], "file-2": []}
    )
    task_repository = _StubTaskRepository(tasks=[task_one, task_two])
    chat_cleanup_repository = _StubChatCleanupRepository()
    chunk_store = _StubChunkStore()
    object_store = _StubObjectStore()
    temp_file_store = _StubTempFileStore()

    service = KnowledgeFileDeleteService(
        knowledge_file_repository=knowledge_file_repository,
        knowledge_file_image_asset_repository=image_asset_repository,
        chat_cleanup_repository=chat_cleanup_repository,
        task_repository=task_repository,
        chunk_store=chunk_store,
        object_store=object_store,
        temp_file_store=temp_file_store,
    )

    result = service.delete_all_files_globally(current_user=_build_current_user())

    assert result.deleted_file_count == 2
    assert result.deleted_task_count == 2
    assert chat_cleanup_repository.delete_all_called is True
    assert chat_cleanup_repository.deleted_user_ids == []
    assert task_repository.delete_all_called is True
    assert sorted(knowledge_file_repository.deleted_file_ids) == ["file-1", "file-2"]
    assert sorted(chunk_store.deleted_file_ids) == ["file-1", "file-2"]


def test_delete_file_raises_when_chunk_cleanup_fails_before_metadata_delete() -> None:
    file_one = _build_file(file_id="file-1", storage_key="knowledge-files/user-1/file-1/a.docx")
    asset_one = _build_asset(file_id="file-1", asset_id="img-1")

    knowledge_file_repository = _StubKnowledgeFileRepository(files=[file_one])
    image_asset_repository = _StubImageAssetRepository(assets_by_file_id={"file-1": [asset_one]})
    chunk_store = _StubChunkStore(failing_file_ids={"file-1"})
    service = KnowledgeFileDeleteService(
        knowledge_file_repository=knowledge_file_repository,
        knowledge_file_image_asset_repository=image_asset_repository,
        chat_cleanup_repository=_StubChatCleanupRepository(),
        task_repository=_StubTaskRepository(tasks=[]),
        chunk_store=chunk_store,
        object_store=_StubObjectStore(),
        temp_file_store=_StubTempFileStore(),
    )

    try:
        service.delete_file(file_id="file-1", current_user=_build_current_user())
        raise AssertionError("索引删除失败时不应继续返回删除成功")
    except KnowledgeFileDeleteCleanupError:
        pass

    assert knowledge_file_repository.deleted_file_ids == []
    assert image_asset_repository.deleted_file_ids == []


def test_delete_all_files_stops_before_chat_and_task_cleanup_when_chunk_cleanup_fails() -> None:
    file_one = _build_file(file_id="file-1", storage_key="knowledge-files/user-1/file-1/a.docx")
    file_two = _build_file(file_id="file-2", storage_key="knowledge-files/user-1/file-2/b.docx")
    task_one = _build_task(task_id="task-1", storage_key="temp/user-1/task-1.docx")

    knowledge_file_repository = _StubKnowledgeFileRepository(files=[file_one, file_two])
    image_asset_repository = _StubImageAssetRepository(
        assets_by_file_id={"file-1": [], "file-2": []}
    )
    task_repository = _StubTaskRepository(tasks=[task_one])
    chat_cleanup_repository = _StubChatCleanupRepository()
    chunk_store = _StubChunkStore(failing_file_ids={"file-2"})
    service = KnowledgeFileDeleteService(
        knowledge_file_repository=knowledge_file_repository,
        knowledge_file_image_asset_repository=image_asset_repository,
        chat_cleanup_repository=chat_cleanup_repository,
        task_repository=task_repository,
        chunk_store=chunk_store,
        object_store=_StubObjectStore(),
        temp_file_store=_StubTempFileStore(),
    )

    try:
        service.delete_all_files(current_user=_build_current_user())
        raise AssertionError("批量删除遇到索引清理失败时不应继续返回成功")
    except KnowledgeFileDeleteCleanupError:
        pass

    assert knowledge_file_repository.deleted_file_ids == []
    assert image_asset_repository.deleted_file_ids == []
    assert chat_cleanup_repository.deleted_user_ids == []
    assert task_repository.deleted_user_ids == []
