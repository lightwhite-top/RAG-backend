"""知识文件仓储抽象。"""

from __future__ import annotations

from typing import Protocol

from baozhi_rag.domain.knowledge_file import (
    FileStorageProvider,
    FileVisibilityScope,
    KnowledgeFile,
    KnowledgeFileListPage,
)


class KnowledgeFileRepository(Protocol):
    """知识文件仓储协议。"""

    def create_file(self, file: KnowledgeFile) -> KnowledgeFile:
        """创建文件记录。"""

    def get_file_by_id(self, file_id: str) -> KnowledgeFile | None:
        """按文件 ID 查询文件。"""

    def get_file_by_user_and_filename(
        self,
        uploader_user_id: str,
        original_filename: str,
    ) -> KnowledgeFile | None:
        """按上传者和文件名查询文件。"""

    def get_file_by_user_and_sha256(
        self,
        uploader_user_id: str,
        sha256: str,
    ) -> KnowledgeFile | None:
        """按上传者和内容哈希查询文件。"""

    def get_file_by_user_and_content_sha256(
        self,
        uploader_user_id: str,
        content_sha256: str,
    ) -> KnowledgeFile | None:
        """按上传者和内容哈希查询文件。"""

    def get_files_by_ids(self, file_ids: list[str]) -> list[KnowledgeFile]:
        """批量查询文件元数据。"""

    def list_global_files(
        self,
        *,
        page: int,
        page_size: int,
    ) -> KnowledgeFileListPage:
        """分页查询全局可见文件。"""

    def list_user_files(
        self,
        *,
        uploader_user_id: str,
        page: int,
        page_size: int,
    ) -> KnowledgeFileListPage:
        """分页查询指定用户上传的文件。"""

    def update_file(
        self,
        file_id: str,
        *,
        original_filename: str | None = None,
        content_type: str | None = None,
        size: int | None = None,
        sha256: str | None = None,
        raw_sha256: str | None = None,
        content_sha256: str | None = None,
        storage_provider: FileStorageProvider | None = None,
        storage_key: str | None = None,
        visibility_scope: FileVisibilityScope | None = None,
        chunk_count: int | None = None,
    ) -> KnowledgeFile | None:
        """更新文件记录。"""

    def replace_file(
        self,
        existing_file_id: str,
        replacement_file: KnowledgeFile,
    ) -> KnowledgeFile:
        """以新文件记录替换旧文件记录。"""

    def delete_file(self, file_id: str) -> bool:
        """删除文件记录。"""
