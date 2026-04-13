"""知识文件与图片资产访问服务。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from baozhi_rag.domain.knowledge_file import FileVisibilityScope, KnowledgeFile
from baozhi_rag.domain.knowledge_file_errors import KnowledgeFileNotFoundError
from baozhi_rag.domain.knowledge_file_image_asset import KnowledgeFileImageAsset
from baozhi_rag.domain.user import CurrentUser


class KnowledgeFileAccessRepository(Protocol):
    """知识文件访问所需的最小元数据仓储协议。"""

    def get_file_by_id(self, file_id: str) -> KnowledgeFile | None:
        """按文件 ID 查询知识文件元数据。"""
        ...


class KnowledgeFileImageAssetAccessRepository(Protocol):
    """图片资产访问所需的最小仓储协议。"""

    def get_asset_by_id(self, asset_id: str) -> KnowledgeFileImageAsset | None:
        """按资产 ID 查询图片资产。"""
        ...


class KnowledgeFileAccessObjectStore(Protocol):
    """知识文件访问所需的对象存储协议。"""

    def iter_download_file(self, *, storage_key: str) -> Iterator[bytes]:
        """按块迭代文件内容。"""
        ...


@dataclass(frozen=True, slots=True)
class KnowledgeFileAccessResult:
    """知识文件访问结果。"""

    knowledge_file: KnowledgeFile
    content_iter: Iterator[bytes]


@dataclass(frozen=True, slots=True)
class KnowledgeBinaryAccessResult:
    """同源二进制内容访问结果。"""

    content_type: str
    filename: str
    content_iter: Iterator[bytes]


class KnowledgeFileAccessService:
    """负责校验文件与图片资产访问权限并提供内容流。"""

    def __init__(
        self,
        *,
        knowledge_file_repository: KnowledgeFileAccessRepository,
        knowledge_file_image_asset_repository: KnowledgeFileImageAssetAccessRepository,
        object_store: KnowledgeFileAccessObjectStore,
    ) -> None:
        """初始化访问服务。"""
        self._knowledge_file_repository = knowledge_file_repository
        self._knowledge_file_image_asset_repository = knowledge_file_image_asset_repository
        self._object_store = object_store

    def open_file(
        self,
        *,
        file_id: str,
        current_user: CurrentUser,
    ) -> KnowledgeFileAccessResult:
        """打开当前用户可访问的知识文件。"""
        knowledge_file = self._knowledge_file_repository.get_file_by_id(file_id)
        if knowledge_file is None or not self._can_access(knowledge_file, current_user):
            raise KnowledgeFileNotFoundError()

        return KnowledgeFileAccessResult(
            knowledge_file=knowledge_file,
            content_iter=self._object_store.iter_download_file(
                storage_key=knowledge_file.storage_key
            ),
        )

    def open_image_asset(
        self,
        *,
        asset_id: str,
        current_user: CurrentUser,
        use_preview: bool = False,
    ) -> KnowledgeBinaryAccessResult:
        """打开当前用户可访问的图片资产或缩略图。"""
        image_asset = self._knowledge_file_image_asset_repository.get_asset_by_id(asset_id)
        if image_asset is None:
            raise KnowledgeFileNotFoundError()

        knowledge_file = self._knowledge_file_repository.get_file_by_id(image_asset.file_id)
        if knowledge_file is None or not self._can_access(knowledge_file, current_user):
            raise KnowledgeFileNotFoundError()

        resolved_storage_key = (
            image_asset.thumbnail_storage_key
            if use_preview and image_asset.thumbnail_storage_key
            else image_asset.storage_key
        )
        return KnowledgeBinaryAccessResult(
            content_type=image_asset.content_type.strip() or "application/octet-stream",
            filename=Path(resolved_storage_key).name or f"{asset_id}.bin",
            content_iter=self._object_store.iter_download_file(storage_key=resolved_storage_key),
        )

    def _can_access(
        self,
        knowledge_file: KnowledgeFile,
        current_user: CurrentUser,
    ) -> bool:
        """判断当前用户是否可以访问该知识文件。"""
        if knowledge_file.visibility_scope is FileVisibilityScope.GLOBAL:
            return True
        return knowledge_file.uploader_user_id == current_user.id
