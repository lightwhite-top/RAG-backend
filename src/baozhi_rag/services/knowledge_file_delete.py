"""知识文件删除服务。"""

from __future__ import annotations

import logging
from typing import Protocol

from baozhi_rag.domain.knowledge_file import KnowledgeFile
from baozhi_rag.domain.knowledge_file_errors import KnowledgeFileNotFoundError
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

    def delete_file(self, file_id: str) -> bool:
        """删除文件记录。

        参数:
            file_id: 需要删除的文件 ID。

        返回:
            删除成功返回 `True`，否则返回 `False`。
        """


class KnowledgeFileDeleteChunkStore(Protocol):
    """知识文件删除所需的最小检索存储协议。"""

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """删除文件关联的全部 chunk。

        参数:
            file_id: 需要删除索引的文件 ID。

        返回:
            None。
        """


class KnowledgeFileObjectStore(Protocol):
    """知识文件删除使用的对象存储协议。"""

    def delete(self, storage_key: str) -> None:
        """删除对象存储中的最终知识文件。

        参数:
            storage_key: 需要删除的对象键。

        返回:
            None。
        """


class KnowledgeFileDeleteService:
    """编排知识文件删除与关联资源清理。"""

    def __init__(
        self,
        *,
        knowledge_file_repository: KnowledgeFileDeleteRepository,
        chunk_store: KnowledgeFileDeleteChunkStore,
        object_store: KnowledgeFileObjectStore,
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
        self._chunk_store = chunk_store
        self._object_store = object_store

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

        if not self._knowledge_file_repository.delete_file(file_id):
            raise KnowledgeFileNotFoundError()

        # 先删除数据库记录，把文件从列表和检索元数据补齐链路中移除；随后再尽力清理
        # 检索索引与对象存储，避免调用方在外部依赖短暂抖动时继续看到“已删除文件”。
        self._run_cleanup(
            file_id=knowledge_file.id,
            uploader_user_id=knowledge_file.uploader_user_id,
            storage_key=knowledge_file.storage_key,
        )

    def _run_cleanup(
        self,
        *,
        file_id: str,
        uploader_user_id: str,
        storage_key: str,
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
