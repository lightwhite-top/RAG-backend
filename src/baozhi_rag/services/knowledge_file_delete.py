"""知识文件删除服务。"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

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


class KnowledgeFileImageAssetDeleteRepository(Protocol):
    """知识文件删除使用的图片资产仓储协议。"""

    def list_assets_by_file_id(self, file_id: str) -> list[KnowledgeFileImageAsset]:
        """按文件 ID 查询图片资产。"""
        ...

    def delete_assets_by_file_id(self, file_id: str) -> int:
        """按文件 ID 删除图片资产。"""
        ...


class KnowledgeFileDeleteService:
    """编排知识文件删除与关联资源清理。"""

    def __init__(
        self,
        *,
        knowledge_file_repository: KnowledgeFileDeleteRepository,
        knowledge_file_image_asset_repository: KnowledgeFileImageAssetDeleteRepository,
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
        if not self._knowledge_file_repository.delete_file(file_id):
            raise KnowledgeFileNotFoundError()
        self._knowledge_file_image_asset_repository.delete_assets_by_file_id(file_id)

        # 先删除数据库记录，把文件从列表和检索元数据补齐链路中移除；随后再尽力清理
        # 检索索引与对象存储，避免调用方在外部依赖短暂抖动时继续看到“已删除文件”。
        self._run_cleanup(
            file_id=knowledge_file.id,
            uploader_user_id=knowledge_file.uploader_user_id,
            storage_key=knowledge_file.storage_key,
            image_assets=image_assets,
        )

    def _run_cleanup(
        self,
        *,
        file_id: str,
        uploader_user_id: str,
        storage_key: str,
        image_assets: list[KnowledgeFileImageAsset],
    ) -> None:
        """执行删除后的索引与对象存储清理。

        参数:
            file_id: 已删除文件的 ID。
            uploader_user_id: 上传者用户 ID，用于日志审计。
            storage_key: 需要删除的最终知识文件对象键。

        返回:
            None。
        """
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

    def delete_all_files(self, *, current_user: CurrentUser) -> KnowledgeFilePurgeResult:
        """删除当前用户上传的全部知识文件与上传任务。

        参数:
            current_user: 当前登录用户，用于限定只清理自己的知识库数据。

        返回:
            批量删除结果摘要。
        """
        files = self._list_all_user_files(current_user.id)
        tasks = self._task_repository.delete_tasks_by_user(current_user.id)

        for task in tasks:
            with suppress(Exception):
                self._temp_file_store.delete(task.source_storage_key)

        deleted_file_count = 0
        for knowledge_file in files:
            image_assets = self._knowledge_file_image_asset_repository.list_assets_by_file_id(
                knowledge_file.id
            )
            if not self._knowledge_file_repository.delete_file(knowledge_file.id):
                continue
            self._knowledge_file_image_asset_repository.delete_assets_by_file_id(knowledge_file.id)
            self._run_cleanup(
                file_id=knowledge_file.id,
                uploader_user_id=knowledge_file.uploader_user_id,
                storage_key=knowledge_file.storage_key,
                image_assets=image_assets,
            )
            deleted_file_count += 1

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
        del current_user
        files = self._list_all_files()
        tasks = self._task_repository.delete_all_tasks()

        for task in tasks:
            with suppress(Exception):
                self._temp_file_store.delete(task.source_storage_key)

        deleted_file_count = 0
        for knowledge_file in files:
            image_assets = self._knowledge_file_image_asset_repository.list_assets_by_file_id(
                knowledge_file.id
            )
            if not self._knowledge_file_repository.delete_file(knowledge_file.id):
                continue
            self._knowledge_file_image_asset_repository.delete_assets_by_file_id(knowledge_file.id)
            self._run_cleanup(
                file_id=knowledge_file.id,
                uploader_user_id=knowledge_file.uploader_user_id,
                storage_key=knowledge_file.storage_key,
                image_assets=image_assets,
            )
            deleted_file_count += 1

        return KnowledgeFilePurgeResult(
            deleted_file_count=deleted_file_count,
            deleted_task_count=len(tasks),
        )

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
