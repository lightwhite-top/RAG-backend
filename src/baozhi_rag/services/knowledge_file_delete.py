"""知识文件删除服务。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from typing import Protocol

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.domain.knowledge_file import KnowledgeFile, KnowledgeFileListPage
from baozhi_rag.domain.knowledge_file_errors import KnowledgeFileNotFoundError
from baozhi_rag.domain.knowledge_file_image_asset import KnowledgeFileImageAsset
from baozhi_rag.domain.knowledge_upload_task import KnowledgeUploadTask
from baozhi_rag.domain.user import CurrentUser

LOGGER = logging.getLogger(__name__)


class KnowledgeFileDeleteRepository(Protocol):
    """知识文件删除所需的最小仓储协议。"""

    def get_file_by_id(self, file_id: str) -> KnowledgeFile | None:
        """按文件 ID 查询文件元数据。

        参数:
            file_id: 需要查询的文件 ID。

        返回:
            找到时返回知识文件实体，否则返回 `None`。
        """
        ...

    def delete_file(self, file_id: str) -> bool:
        """删除文件记录。

        参数:
            file_id: 需要删除的文件 ID。

        返回:
            删除成功返回 `True`，否则返回 `False`。
        """
        ...

    def list_user_files(
        self,
        *,
        uploader_user_id: str,
        page: int,
        page_size: int,
    ) -> KnowledgeFileListPage:
        """按用户分页列出文件。"""
        ...

    def list_all_files(
        self,
        *,
        page: int,
        page_size: int,
    ) -> KnowledgeFileListPage:
        """分页列出全部文件。"""
        ...


class KnowledgeFileDeleteChunkStore(Protocol):
    """知识文件删除所需的最小检索存储协议。"""

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """删除文件关联的全部 chunk。

        参数:
            file_id: 需要删除索引的文件 ID。

        返回:
            None。
        """
        ...


class KnowledgeFileObjectStore(Protocol):
    """知识文件删除使用的对象存储协议。"""

    def delete(self, storage_key: str) -> None:
        """删除对象存储中的最终知识文件。

        参数:
            storage_key: 需要删除的对象键。

        返回:
            None。
        """
        ...


class KnowledgeUploadTaskCleanupRepository(Protocol):
    """批量清理文件数据时使用的上传任务仓储协议。"""

    def delete_tasks_by_user(
        self,
        uploader_user_id: str,
    ) -> list[KnowledgeUploadTask]:
        """删除指定用户的全部上传任务。"""
        ...

    def delete_all_tasks(self) -> list[KnowledgeUploadTask]:
        """删除全部上传任务。"""
        ...


class KnowledgeFileTempStore(Protocol):
    """批量清理时使用的临时文件存储协议。"""

    def delete(self, storage_key: str) -> None:
        """删除暂存文件。"""
        ...


@dataclass(frozen=True, slots=True)
class KnowledgeFilePurgeResult:
    """批量清理当前用户知识库数据后的结果摘要。"""

    deleted_file_count: int
    deleted_task_count: int


@dataclass(frozen=True, slots=True)
class ChatRecordPurgeResult:
    """聊天记录批量清理结果摘要。"""

    deleted_session_count: int
    deleted_message_count: int
    deleted_snapshot_count: int


class KnowledgeFileImageAssetDeleteRepository(Protocol):
    """知识文件删除使用的图片资产仓储协议。"""

    def list_assets_by_file_id(self, file_id: str) -> list[KnowledgeFileImageAsset]:
        """按文件 ID 查询图片资产。"""
        ...

    def delete_assets_by_file_id(self, file_id: str) -> int:
        """按文件 ID 删除图片资产。"""
        ...


class ChatRecordCleanupRepository(Protocol):
    """批量清理聊天记录所需的最小仓储协议。"""

    def delete_chat_records_by_user(self, owner_user_id: str) -> ChatRecordPurgeResult:
        """删除指定用户的全部聊天记录。"""
        ...

    def delete_all_chat_records(self) -> ChatRecordPurgeResult:
        """删除全站全部聊天记录。"""
        ...


class KnowledgeFileDeleteCleanupError(AppError):
    """知识文件删除时的检索或外部资源清理失败。"""

    default_message = "删除文件关联索引失败，请稍后重试"
    default_error_code = "knowledge_file_delete_cleanup_failed"
    default_status_code = int(HTTPStatus.BAD_GATEWAY)


@dataclass(frozen=True, slots=True)
class _KnowledgeFileCleanupPlan:
    """描述单个知识文件删除后的后续清理计划。"""

    file_id: str
    uploader_user_id: str
    storage_key: str
    image_assets: list[KnowledgeFileImageAsset]


@dataclass(frozen=True, slots=True)
class _ObjectStorageCleanupTask:
    """描述单个对象存储删除任务。"""

    file_id: str
    uploader_user_id: str
    cleanup_target: str
    storage_key: str


class KnowledgeFileDeleteService:
    """编排知识文件删除与关联资源清理。"""

    _BULK_OBJECT_STORAGE_DELETE_MAX_WORKERS = 8

    def __init__(
        self,
        *,
        knowledge_file_repository: KnowledgeFileDeleteRepository,
        knowledge_file_image_asset_repository: KnowledgeFileImageAssetDeleteRepository,
        chat_cleanup_repository: ChatRecordCleanupRepository,
        task_repository: KnowledgeUploadTaskCleanupRepository,
        chunk_store: KnowledgeFileDeleteChunkStore,
        object_store: KnowledgeFileObjectStore,
        temp_file_store: KnowledgeFileTempStore,
    ) -> None:
        """初始化知识文件删除服务。

        参数:
            knowledge_file_repository: 知识文件元数据仓储。
            chunk_store: 检索 chunk 存储，用于删除文件关联索引。
            object_store: 对象存储客户端，用于删除最终知识文件对象。

        返回:
            None。
        """
        self._knowledge_file_repository = knowledge_file_repository
        self._knowledge_file_image_asset_repository = knowledge_file_image_asset_repository
        self._chat_cleanup_repository = chat_cleanup_repository
        self._task_repository = task_repository
        self._chunk_store = chunk_store
        self._object_store = object_store
        self._temp_file_store = temp_file_store

    def delete_file(self, *, file_id: str, current_user: CurrentUser) -> None:
        """删除当前用户自己上传的知识文件。

        参数:
            file_id: 需要删除的知识文件 ID。
            current_user: 当前登录用户，用于校验文件归属。

        返回:
            None。

        异常:
            KnowledgeFileNotFoundError: 文件不存在，或不属于当前用户。
        """
        knowledge_file = self._knowledge_file_repository.get_file_by_id(file_id)
        if knowledge_file is None or knowledge_file.uploader_user_id != current_user.id:
            raise KnowledgeFileNotFoundError()

        image_assets = self._knowledge_file_image_asset_repository.list_assets_by_file_id(file_id)
        self._delete_chunk_index_or_raise(
            file_id=knowledge_file.id,
            uploader_user_id=knowledge_file.uploader_user_id,
        )
        if not self._knowledge_file_repository.delete_file(file_id):
            raise KnowledgeFileNotFoundError()
        self._knowledge_file_image_asset_repository.delete_assets_by_file_id(file_id)

        # 单文件删除优先保证检索残留不会继续暴露；对象存储失败只记日志，
        # 但不会再影响搜索或同源下载鉴权结果。
        self._run_cleanup(
            file_id=knowledge_file.id,
            uploader_user_id=knowledge_file.uploader_user_id,
            storage_key=knowledge_file.storage_key,
            image_assets=image_assets,
            skip_chunk_cleanup=True,
        )

    def _run_cleanup(
        self,
        *,
        file_id: str,
        uploader_user_id: str,
        storage_key: str,
        image_assets: list[KnowledgeFileImageAsset],
        skip_chunk_cleanup: bool = False,
    ) -> None:
        """执行删除后的索引与对象存储清理。

        参数:
            file_id: 已删除文件的 ID。
            uploader_user_id: 上传者用户 ID，用于日志审计。
            storage_key: 需要删除的最终知识文件对象键。

        返回:
            None。
        """
        cleanup_operations: tuple[tuple[str, Callable[[], None]], ...]
        if skip_chunk_cleanup:
            cleanup_operations = (
                ("object_storage", lambda: self._object_store.delete(storage_key)),
            )
        else:
            cleanup_operations = (
                ("chunk_index", lambda: self._chunk_store.delete_chunks_by_file_id(file_id)),
                ("object_storage", lambda: self._object_store.delete(storage_key)),
            )
        for cleanup_target, cleanup_operation in cleanup_operations:
            try:
                cleanup_operation()
            except Exception:
                LOGGER.warning(
                    (
                        "knowledge_file_delete_cleanup_failed "
                        "file_id=%s uploader_user_id=%s cleanup_target=%s"
                    ),
                    file_id,
                    uploader_user_id,
                    cleanup_target,
                    exc_info=True,
                )
        for image_asset in image_assets:
            for cleanup_target, cleanup_storage_key in (
                ("image_object_storage", image_asset.storage_key),
                ("image_thumbnail_storage", image_asset.thumbnail_storage_key),
            ):
                if not cleanup_storage_key:
                    continue
                try:
                    self._object_store.delete(cleanup_storage_key)
                except Exception:
                    LOGGER.warning(
                        (
                            "knowledge_file_delete_cleanup_failed "
                            "file_id=%s uploader_user_id=%s cleanup_target=%s"
                        ),
                        file_id,
                        uploader_user_id,
                        cleanup_target,
                        exc_info=True,
                    )

    def _run_bulk_cleanup(
        self,
        cleanup_plans: list[_KnowledgeFileCleanupPlan],
    ) -> None:
        """批量执行索引与对象存储清理。

        参数:
            cleanup_plans: 已完成元数据删除的文件清理计划列表。

        返回:
            None。
        """
        if not cleanup_plans:
            return

        object_cleanup_tasks = self._build_object_storage_cleanup_tasks(cleanup_plans)
        if not object_cleanup_tasks:
            return

        max_workers = min(
            self._BULK_OBJECT_STORAGE_DELETE_MAX_WORKERS,
            len(object_cleanup_tasks),
        )
        # OSS 删除属于高延迟 I/O；仅在批量清理场景下并发删除对象，缩短管理员全量清理耗时。
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="knowledge-file-delete",
        ) as executor:
            futures = [
                executor.submit(self._delete_object_storage_task, cleanup_task)
                for cleanup_task in object_cleanup_tasks
            ]
            for future in futures:
                future.result()

    def _build_object_storage_cleanup_tasks(
        self,
        cleanup_plans: list[_KnowledgeFileCleanupPlan],
    ) -> list[_ObjectStorageCleanupTask]:
        """把文件清理计划转换为对象存储删除任务。"""
        cleanup_tasks: list[_ObjectStorageCleanupTask] = []
        seen_storage_keys: set[str] = set()

        for cleanup_plan in cleanup_plans:
            self._append_cleanup_task_if_needed(
                cleanup_tasks=cleanup_tasks,
                seen_storage_keys=seen_storage_keys,
                cleanup_target="object_storage",
                storage_key=cleanup_plan.storage_key,
                file_id=cleanup_plan.file_id,
                uploader_user_id=cleanup_plan.uploader_user_id,
            )
            for image_asset in cleanup_plan.image_assets:
                self._append_cleanup_task_if_needed(
                    cleanup_tasks=cleanup_tasks,
                    seen_storage_keys=seen_storage_keys,
                    cleanup_target="image_object_storage",
                    storage_key=image_asset.storage_key,
                    file_id=cleanup_plan.file_id,
                    uploader_user_id=cleanup_plan.uploader_user_id,
                )
                self._append_cleanup_task_if_needed(
                    cleanup_tasks=cleanup_tasks,
                    seen_storage_keys=seen_storage_keys,
                    cleanup_target="image_thumbnail_storage",
                    storage_key=image_asset.thumbnail_storage_key,
                    file_id=cleanup_plan.file_id,
                    uploader_user_id=cleanup_plan.uploader_user_id,
                )
        return cleanup_tasks

    def _append_cleanup_task_if_needed(
        self,
        *,
        cleanup_tasks: list[_ObjectStorageCleanupTask],
        seen_storage_keys: set[str],
        cleanup_target: str,
        storage_key: str | None,
        file_id: str,
        uploader_user_id: str,
    ) -> None:
        """按需追加对象存储删除任务，避免重复删除同一对象键。"""
        normalized_storage_key = (storage_key or "").strip()
        if not normalized_storage_key or normalized_storage_key in seen_storage_keys:
            return

        seen_storage_keys.add(normalized_storage_key)
        cleanup_tasks.append(
            _ObjectStorageCleanupTask(
                file_id=file_id,
                uploader_user_id=uploader_user_id,
                cleanup_target=cleanup_target,
                storage_key=normalized_storage_key,
            )
        )

    def _delete_object_storage_task(self, cleanup_task: _ObjectStorageCleanupTask) -> None:
        """执行单个对象存储删除任务。"""
        try:
            self._object_store.delete(cleanup_task.storage_key)
        except Exception:
            LOGGER.warning(
                (
                    "knowledge_file_delete_cleanup_failed "
                    "file_id=%s uploader_user_id=%s cleanup_target=%s"
                ),
                cleanup_task.file_id,
                cleanup_task.uploader_user_id,
                cleanup_task.cleanup_target,
                exc_info=True,
            )

    def delete_all_files(self, *, current_user: CurrentUser) -> KnowledgeFilePurgeResult:
        """删除当前用户上传的全部知识文件与上传任务。

        参数:
            current_user: 当前登录用户，用于限定只清理自己的知识库数据。

        返回:
            批量删除结果摘要。
        """
        files = self._list_all_user_files(current_user.id)
        image_assets_by_file_id = {
            knowledge_file.id: self._knowledge_file_image_asset_repository.list_assets_by_file_id(
                knowledge_file.id
            )
            for knowledge_file in files
        }
        for knowledge_file in files:
            self._delete_chunk_index_or_raise(
                file_id=knowledge_file.id,
                uploader_user_id=knowledge_file.uploader_user_id,
            )

        deleted_file_count = 0
        cleanup_plans: list[_KnowledgeFileCleanupPlan] = []
        for knowledge_file in files:
            image_assets = image_assets_by_file_id.get(knowledge_file.id, [])
            if not self._knowledge_file_repository.delete_file(knowledge_file.id):
                continue
            self._knowledge_file_image_asset_repository.delete_assets_by_file_id(knowledge_file.id)
            cleanup_plans.append(
                _KnowledgeFileCleanupPlan(
                    file_id=knowledge_file.id,
                    uploader_user_id=knowledge_file.uploader_user_id,
                    storage_key=knowledge_file.storage_key,
                    image_assets=image_assets,
                )
            )
            deleted_file_count += 1

        chat_purge_result = self._chat_cleanup_repository.delete_chat_records_by_user(
            current_user.id
        )
        LOGGER.info(
            (
                "knowledge_file_user_chat_records_deleted user_id=%s "
                "deleted_session_count=%s deleted_message_count=%s deleted_snapshot_count=%s"
            ),
            current_user.id,
            chat_purge_result.deleted_session_count,
            chat_purge_result.deleted_message_count,
            chat_purge_result.deleted_snapshot_count,
        )
        tasks = self._task_repository.delete_tasks_by_user(current_user.id)

        for task in tasks:
            with suppress(Exception):
                self._temp_file_store.delete(task.source_storage_key)

        self._run_bulk_cleanup(cleanup_plans)

        return KnowledgeFilePurgeResult(
            deleted_file_count=deleted_file_count,
            deleted_task_count=len(tasks),
        )

    def delete_all_files_globally(
        self,
        *,
        current_user: CurrentUser,
    ) -> KnowledgeFilePurgeResult:
        """删除全站知识文件与上传任务。

        参数:
            current_user: 当前管理员用户，用于审计与日志上下文。

        返回:
            全站批量删除结果摘要。
        """
        files = self._list_all_files()
        image_assets_by_file_id = {
            knowledge_file.id: self._knowledge_file_image_asset_repository.list_assets_by_file_id(
                knowledge_file.id
            )
            for knowledge_file in files
        }
        for knowledge_file in files:
            self._delete_chunk_index_or_raise(
                file_id=knowledge_file.id,
                uploader_user_id=knowledge_file.uploader_user_id,
            )

        deleted_file_count = 0
        cleanup_plans: list[_KnowledgeFileCleanupPlan] = []
        for knowledge_file in files:
            image_assets = image_assets_by_file_id.get(knowledge_file.id, [])
            if not self._knowledge_file_repository.delete_file(knowledge_file.id):
                continue
            self._knowledge_file_image_asset_repository.delete_assets_by_file_id(knowledge_file.id)
            cleanup_plans.append(
                _KnowledgeFileCleanupPlan(
                    file_id=knowledge_file.id,
                    uploader_user_id=knowledge_file.uploader_user_id,
                    storage_key=knowledge_file.storage_key,
                    image_assets=image_assets,
                )
            )
            deleted_file_count += 1

        chat_purge_result = self._chat_cleanup_repository.delete_all_chat_records()
        LOGGER.info(
            (
                "knowledge_file_global_chat_records_deleted admin_user_id=%s "
                "deleted_session_count=%s deleted_message_count=%s deleted_snapshot_count=%s"
            ),
            current_user.id,
            chat_purge_result.deleted_session_count,
            chat_purge_result.deleted_message_count,
            chat_purge_result.deleted_snapshot_count,
        )
        tasks = self._task_repository.delete_all_tasks()

        for task in tasks:
            with suppress(Exception):
                self._temp_file_store.delete(task.source_storage_key)

        self._run_bulk_cleanup(cleanup_plans)

        return KnowledgeFilePurgeResult(
            deleted_file_count=deleted_file_count,
            deleted_task_count=len(tasks),
        )

    def _delete_chunk_index_or_raise(
        self,
        *,
        file_id: str,
        uploader_user_id: str,
    ) -> None:
        """删除检索索引，失败时中断删除流程。

        参数:
            file_id: 待删除文件 ID。
            uploader_user_id: 上传者用户 ID，用于日志审计。

        返回:
            None。

        异常:
            KnowledgeFileDeleteCleanupError: 当 ES/Milvus 清理失败时抛出。
        """
        try:
            self._chunk_store.delete_chunks_by_file_id(file_id)
        except Exception as exc:
            LOGGER.warning(
                (
                    "knowledge_file_delete_cleanup_failed "
                    "file_id=%s uploader_user_id=%s cleanup_target=%s"
                ),
                file_id,
                uploader_user_id,
                "chunk_index",
                exc_info=True,
            )
            raise KnowledgeFileDeleteCleanupError() from exc

    def _list_all_user_files(self, uploader_user_id: str) -> list[KnowledgeFile]:
        """按分页方式收集指定用户的全部文件快照。"""
        page = 1
        page_size = 100
        files: list[KnowledgeFile] = []

        while True:
            result = self._knowledge_file_repository.list_user_files(
                uploader_user_id=uploader_user_id,
                page=page,
                page_size=page_size,
            )
            current_items = list(getattr(result, "items", []))
            if not current_items:
                break
            files.extend(current_items)
            if len(current_items) < page_size:
                break
            page += 1

        return files

    def _list_all_files(self) -> list[KnowledgeFile]:
        """按分页方式收集全站全部文件快照。"""
        page = 1
        page_size = 100
        files: list[KnowledgeFile] = []

        while True:
            result = self._knowledge_file_repository.list_all_files(
                page=page,
                page_size=page_size,
            )
            current_items = list(result.items)
            if not current_items:
                break
            files.extend(current_items)
            if len(current_items) < page_size:
                break
            page += 1

        return files
