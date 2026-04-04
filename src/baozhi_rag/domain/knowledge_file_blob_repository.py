"""知识文件原始 blob 仓储抽象。"""

from __future__ import annotations

from typing import Protocol

from baozhi_rag.domain.knowledge_file_blob import KnowledgeFileBlob


class KnowledgeFileBlobRepository(Protocol):
    """原始文件 blob 仓储协议。"""

    def create_blob(self, blob: KnowledgeFileBlob) -> KnowledgeFileBlob:
        """创建 blob 记录。"""

    def get_blob_by_raw_sha256(self, raw_sha256: str) -> KnowledgeFileBlob | None:
        """按原始哈希查询 blob。"""
