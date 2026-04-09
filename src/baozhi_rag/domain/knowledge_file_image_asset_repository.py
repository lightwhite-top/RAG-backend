"""知识文件图片资产仓储抽象。"""

from __future__ import annotations

from typing import Protocol

from baozhi_rag.domain.knowledge_file_image_asset import KnowledgeFileImageAsset


class KnowledgeFileImageAssetRepository(Protocol):
    """知识文件图片资产仓储协议。"""

    def create_assets(
        self,
        assets: list[KnowledgeFileImageAsset],
    ) -> list[KnowledgeFileImageAsset]:
        """批量创建图片资产。"""
        ...

    def list_assets_by_file_id(self, file_id: str) -> list[KnowledgeFileImageAsset]:
        """按文件 ID 查询图片资产。"""
        ...

    def list_assets_by_chunk_ids(self, chunk_ids: list[str]) -> list[KnowledgeFileImageAsset]:
        """按 chunk ID 列表查询图片资产。"""
        ...

    def get_asset_by_uploader_and_normalized_sha256(
        self,
        uploader_user_id: str,
        normalized_image_sha256: str,
    ) -> KnowledgeFileImageAsset | None:
        """按上传者和归一化图片哈希查询可复用图片资产。"""
        ...

    def delete_assets_by_file_id(self, file_id: str) -> int:
        """按文件 ID 删除图片资产。"""
        ...
