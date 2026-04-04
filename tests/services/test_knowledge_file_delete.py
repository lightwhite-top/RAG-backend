"""知识文件删除服务测试。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from baozhi_rag.domain.knowledge_file import (
    FileStorageProvider,
    FileVisibilityScope,
    KnowledgeFile,
)
from baozhi_rag.domain.knowledge_file_errors import KnowledgeFileNotFoundError
from baozhi_rag.domain.user import CurrentUser, UserRole
from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest
from baozhi_rag.services.knowledge_file_delete import KnowledgeFileDeleteService


class InMemoryKnowledgeFileRepository:
    """知识文件仓储测试替身。"""

    def __init__(self, *files: KnowledgeFile) -> None:
        self._files = {file.id: file for file in files}

    def get_file_by_id(self, file_id: str) -> KnowledgeFile | None:
        return self._files.get(file_id)

    def delete_file(self, file_id: str) -> bool:
        return self._files.pop(file_id, None) is not None


class RecordingChunkStore:
    """记录删除动作的 chunk 存储替身。"""

    def __init__(self) -> None:
        self.deleted_file_ids: list[str] = []

    def ensure_index(self) -> None:
        return

    def index_chunks(self, chunks: list[object]) -> int:
        return len(chunks)

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        self.deleted_file_ids.append(file_id)

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        del request
        return []


class RecordingObjectStore:
    """记录对象删除动作的对象存储替身。"""

    def __init__(self) -> None:
        self.deleted_keys: list[str] = []

    def delete(self, storage_key: str) -> None:
        self.deleted_keys.append(storage_key)


def test_delete_file_removes_metadata_index_and_object() -> None:
    """删除自己的文件时应清理元数据、索引和对象。"""
    knowledge_file = _build_file(file_id="file-1", uploader_user_id="user-1")
    repository = InMemoryKnowledgeFileRepository(knowledge_file)
    chunk_store = RecordingChunkStore()
    object_store = RecordingObjectStore()
    service = KnowledgeFileDeleteService(
        knowledge_file_repository=repository,
        chunk_store=chunk_store,
        object_store=object_store,
    )

    service.delete_file(file_id="file-1", current_user=_build_user(user_id="user-1"))

    assert repository.get_file_by_id("file-1") is None
    assert chunk_store.deleted_file_ids == ["file-1"]
    assert object_store.deleted_keys == [knowledge_file.storage_key]


def test_delete_file_raises_not_found_when_file_missing() -> None:
    """删除不存在的文件时应返回 not found。"""
    service = KnowledgeFileDeleteService(
        knowledge_file_repository=InMemoryKnowledgeFileRepository(),
        chunk_store=RecordingChunkStore(),
        object_store=RecordingObjectStore(),
    )

    with pytest.raises(KnowledgeFileNotFoundError):
        service.delete_file(file_id="missing-file", current_user=_build_user(user_id="user-1"))


def test_delete_file_hides_other_users_file_existence() -> None:
    """删除他人文件时应按不存在处理，避免泄露文件存在性。"""
    service = KnowledgeFileDeleteService(
        knowledge_file_repository=InMemoryKnowledgeFileRepository(
            _build_file(file_id="file-1", uploader_user_id="user-2")
        ),
        chunk_store=RecordingChunkStore(),
        object_store=RecordingObjectStore(),
    )

    with pytest.raises(KnowledgeFileNotFoundError):
        service.delete_file(file_id="file-1", current_user=_build_user(user_id="user-1"))


def _build_file(*, file_id: str, uploader_user_id: str) -> KnowledgeFile:
    """构造测试使用的知识文件实体。"""
    now = datetime(2026, 4, 4, tzinfo=UTC)
    return KnowledgeFile(
        id=file_id,
        uploader_user_id=uploader_user_id,
        original_filename="保险条款.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        size=1024,
        storage_provider=FileStorageProvider.ALIYUN_OSS,
        storage_key=f"knowledge-files/{uploader_user_id}/{file_id}/保险条款.docx",
        visibility_scope=FileVisibilityScope.OWNER_ONLY,
        chunk_count=3,
        uploaded_at=now,
        updated_at=now,
        raw_sha256="raw-sha256",
        content_sha256="content-sha256",
    )


def _build_user(*, user_id: str) -> CurrentUser:
    """构造测试使用的当前用户。"""
    now = datetime(2026, 4, 4, tzinfo=UTC)
    return CurrentUser(
        id=user_id,
        email=f"{user_id}@example.com",
        username=f"user-{user_id}",
        role=UserRole.USER,
        created_at=now,
        updated_at=now,
    )
