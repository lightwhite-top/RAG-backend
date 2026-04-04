"""上传后文档切块预览编排服务。"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from baozhi_rag.domain.knowledge_file import (
    FileStorageProvider,
    FileVisibilityScope,
    KnowledgeFile,
)
from baozhi_rag.domain.knowledge_file_repository import KnowledgeFileRepository
from baozhi_rag.domain.user import CurrentUser, UserRole
from baozhi_rag.infra.storage.local_file_store import LocalFileStore
from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.chunk_search import ChunkSearchStore
from baozhi_rag.services.document_chunking import DocumentChunk, DocumentChunkService
from baozhi_rag.services.file_upload import (
    FileUploadInput,
    FileUploadService,
    StagedUploadFileResult,
    UploadedFileResult,
)

LOGGER = logging.getLogger(__name__)


class ObjectFileStore(Protocol):
    """对象存储协议。"""

    def upload_file(self, *, local_path: Path, storage_key: str) -> None:
        """上传本地文件到对象存储。"""

    def delete(self, storage_key: str) -> None:
        """删除对象存储中的文件。"""


@dataclass(frozen=True, slots=True)
class ChunkedFileResult:
    """已上传并完成切块预览的文件结果。"""

    upload: UploadedFileResult
    chunks: list[DocumentChunk]


@dataclass(frozen=True, slots=True)
class _NewUploadAction:
    """新上传文件的回滚信息。"""

    knowledge_file: KnowledgeFile


@dataclass(frozen=True, slots=True)
class _TitleUpdateAction:
    """仅标题更新的回滚信息。"""

    file_id: str
    previous_filename: str


@dataclass(frozen=True, slots=True)
class _ReplacementAction:
    """覆盖更新的回滚与收尾信息。"""

    previous_file: KnowledgeFile
    new_file: KnowledgeFile


class DocumentPreviewService:
    """编排文件上传、切块、向量化与入库。"""

    def __init__(
        self,
        file_upload_service: FileUploadService,
        chunk_service: DocumentChunkService,
        temp_file_store: LocalFileStore,
        object_store: ObjectFileStore,
        knowledge_file_repository: KnowledgeFileRepository,
        chunk_store: ChunkSearchStore,
        chunk_embedding_service: ChunkEmbeddingService,
        oss_object_prefix: str,
    ) -> None:
        """初始化上传预览服务。"""
        self._file_upload_service = file_upload_service
        self._chunk_service = chunk_service
        self._temp_file_store = temp_file_store
        self._object_store = object_store
        self._knowledge_file_repository = knowledge_file_repository
        self._chunk_store = chunk_store
        self._chunk_embedding_service = chunk_embedding_service
        self._oss_object_prefix = oss_object_prefix.strip().strip("/")

    def upload_and_chunk_files(
        self,
        files: list[FileUploadInput],
        *,
        current_user: CurrentUser,
    ) -> list[ChunkedFileResult]:
        """上传文件并完成去重、切块、向量化与入库。"""
        staged_files = self._file_upload_service.stage_files(files)
        completed_actions: list[object] = []
        completed_results: list[ChunkedFileResult] = []

        try:
            for staged_file in staged_files:
                result, action = self._process_staged_file(
                    staged_file=staged_file,
                    current_user=current_user,
                )
                completed_results.append(result)
                if action is not None:
                    completed_actions.append(action)
        except Exception:
            self._rollback_actions(completed_actions)
            self._cleanup_staged_files(staged_files)
            raise

        self._cleanup_staged_files(staged_files)
        self._finalize_actions(completed_actions)
        return completed_results

    def _process_staged_file(
        self,
        *,
        staged_file: StagedUploadFileResult,
        current_user: CurrentUser,
    ) -> tuple[ChunkedFileResult, object | None]:
        """根据去重规则处理单个暂存文件。"""
        preview_chunks = self._build_preview_chunks(staged_file)
        content_sha256 = self._build_content_sha256(preview_chunks)
        existing_same_name = self._knowledge_file_repository.get_file_by_user_and_filename(
            current_user.id,
            staged_file.original_filename,
        )
        existing_same_content = self._knowledge_file_repository.get_file_by_user_and_sha256(
            current_user.id,
            content_sha256,
        )

        if existing_same_name is not None and existing_same_name.sha256 == content_sha256:
            return self._build_duplicate_result(existing_same_name), None

        if existing_same_name is not None:
            return self._replace_existing_file(
                existing_file=existing_same_name,
                staged_file=staged_file,
                preview_chunks=preview_chunks,
                content_sha256=content_sha256,
                current_user=current_user,
            )

        if existing_same_content is not None:
            return self._update_existing_title(
                existing_file=existing_same_content,
                staged_file=staged_file,
            )

        return self._create_new_file(
            staged_file=staged_file,
            preview_chunks=preview_chunks,
            content_sha256=content_sha256,
            current_user=current_user,
        )

    def _create_new_file(
        self,
        *,
        staged_file: StagedUploadFileResult,
        preview_chunks: list[DocumentChunk],
        content_sha256: str,
        current_user: CurrentUser,
    ) -> tuple[ChunkedFileResult, _NewUploadAction]:
        """处理全新文件上传。"""
        knowledge_file = self._build_knowledge_file(
            staged_file=staged_file,
            current_user=current_user,
            original_filename=staged_file.original_filename,
            content_sha256=content_sha256,
        )
        chunks = self._upload_and_index_new_file(
            staged_file=staged_file,
            knowledge_file=knowledge_file,
            preview_chunks=preview_chunks,
        )
        try:
            persisted_file = self._knowledge_file_repository.create_file(
                replace(knowledge_file, chunk_count=len(chunks))
            )
        except Exception:
            with suppress(Exception):
                self._chunk_store.delete_chunks_by_file_id(knowledge_file.id)
            with suppress(Exception):
                self._object_store.delete(knowledge_file.storage_key)
            raise
        return (
            ChunkedFileResult(
                upload=self._build_uploaded_result(persisted_file),
                chunks=chunks,
            ),
            _NewUploadAction(knowledge_file=persisted_file),
        )

    def _replace_existing_file(
        self,
        *,
        existing_file: KnowledgeFile,
        staged_file: StagedUploadFileResult,
        preview_chunks: list[DocumentChunk],
        content_sha256: str,
        current_user: CurrentUser,
    ) -> tuple[ChunkedFileResult, _ReplacementAction]:
        """处理同名不同内容的覆盖更新。"""
        replacement_file = self._build_knowledge_file(
            staged_file=staged_file,
            current_user=current_user,
            original_filename=staged_file.original_filename,
            content_sha256=content_sha256,
        )
        chunks = self._upload_and_index_new_file(
            staged_file=staged_file,
            knowledge_file=replacement_file,
            preview_chunks=preview_chunks,
        )
        try:
            persisted_file = self._knowledge_file_repository.replace_file(
                existing_file.id,
                replace(replacement_file, chunk_count=len(chunks)),
            )
        except Exception:
            with suppress(Exception):
                self._chunk_store.delete_chunks_by_file_id(replacement_file.id)
            with suppress(Exception):
                self._object_store.delete(replacement_file.storage_key)
            raise
        return (
            ChunkedFileResult(
                upload=self._build_uploaded_result(persisted_file, replaced=True),
                chunks=chunks,
            ),
            _ReplacementAction(previous_file=existing_file, new_file=persisted_file),
        )

    def _update_existing_title(
        self,
        *,
        existing_file: KnowledgeFile,
        staged_file: StagedUploadFileResult,
    ) -> tuple[ChunkedFileResult, _TitleUpdateAction]:
        """处理同内容不同标题的去重更新。"""
        updated_file = self._knowledge_file_repository.update_file(
            existing_file.id,
            original_filename=staged_file.original_filename,
        )
        if updated_file is None:  # pragma: no cover - 数据库记录在并发下丢失
            updated_file = existing_file
        return (
            ChunkedFileResult(
                upload=self._build_uploaded_result(
                    updated_file,
                    deduplicated=True,
                    title_updated=True,
                ),
                chunks=[],
            ),
            _TitleUpdateAction(
                file_id=existing_file.id,
                previous_filename=existing_file.original_filename,
            ),
        )

    def _build_duplicate_result(self, existing_file: KnowledgeFile) -> ChunkedFileResult:
        """构造完全重复文件的返回结果。"""
        return ChunkedFileResult(
            upload=self._build_uploaded_result(existing_file, deduplicated=True),
            chunks=[],
        )

    def _upload_and_index_new_file(
        self,
        *,
        staged_file: StagedUploadFileResult,
        knowledge_file: KnowledgeFile,
        preview_chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """执行 OSS 上传、向量化与索引写入。"""
        temp_file_path = self._temp_file_store.resolve_path(staged_file.temp_storage_key)
        uploaded_to_oss = False
        indexed = False

        try:
            self._object_store.upload_file(
                local_path=temp_file_path,
                storage_key=knowledge_file.storage_key,
            )
            uploaded_to_oss = True
            chunks = self._materialize_chunks(
                preview_chunks=preview_chunks,
                knowledge_file=knowledge_file,
            )
            self._chunk_store.ensure_index()
            self._chunk_store.index_chunks(chunks)
            indexed = True
            return chunks
        except Exception:
            if indexed:
                with suppress(Exception):
                    self._chunk_store.delete_chunks_by_file_id(knowledge_file.id)
            if uploaded_to_oss:
                with suppress(Exception):
                    self._object_store.delete(knowledge_file.storage_key)
            raise

    def _build_preview_chunks(self, staged_file: StagedUploadFileResult) -> list[DocumentChunk]:
        """基于暂存文件生成用于去重判定的预览 chunk。"""
        temp_file_path = self._temp_file_store.resolve_path(staged_file.temp_storage_key)
        return self._chunk_service.chunk_document(
            file_path=temp_file_path,
            source_filename=staged_file.original_filename,
            storage_key=staged_file.temp_storage_key,
            file_id=staged_file.stage_id,
        )

    def _build_content_sha256(self, preview_chunks: list[DocumentChunk]) -> str:
        """基于解析后的文档内容计算稳定哈希。"""
        payload = [
            {
                "chunk_index": chunk.chunk_index,
                "char_count": chunk.char_count,
                "content": chunk.content,
                "merged_terms": chunk.merged_terms,
            }
            for chunk in preview_chunks
        ]
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _materialize_chunks(
        self,
        *,
        preview_chunks: list[DocumentChunk],
        knowledge_file: KnowledgeFile,
    ) -> list[DocumentChunk]:
        """把预览 chunk 转换为最终入库 chunk。"""
        scoped_chunks = [
            replace(
                chunk,
                file_id=knowledge_file.id,
                chunk_id=f"{knowledge_file.id}-chunk-{chunk.chunk_index}",
                source_filename=knowledge_file.original_filename,
                storage_key=knowledge_file.storage_key,
                uploader_user_id=knowledge_file.uploader_user_id,
                visibility_scope=knowledge_file.visibility_scope.value,
            )
            for chunk in preview_chunks
        ]
        return self._chunk_embedding_service.embed_chunks(scoped_chunks)

    def _build_knowledge_file(
        self,
        *,
        staged_file: StagedUploadFileResult,
        current_user: CurrentUser,
        original_filename: str,
        content_sha256: str,
    ) -> KnowledgeFile:
        """基于暂存结果构造新的文件元数据对象。"""
        uploaded_at = datetime.now(UTC)
        file_id = uuid4().hex
        return KnowledgeFile(
            id=file_id,
            uploader_user_id=current_user.id,
            original_filename=original_filename,
            content_type=staged_file.content_type,
            size=staged_file.size,
            sha256=content_sha256,
            storage_provider=FileStorageProvider.ALIYUN_OSS,
            storage_key=self._build_storage_key(
                uploader_user_id=current_user.id,
                file_id=file_id,
                safe_filename=staged_file.safe_filename,
            ),
            visibility_scope=self._resolve_visibility_scope(current_user),
            chunk_count=0,
            uploaded_at=uploaded_at,
            updated_at=uploaded_at,
        )

    def _build_storage_key(
        self,
        *,
        uploader_user_id: str,
        file_id: str,
        safe_filename: str,
    ) -> str:
        """构造 OSS 对象键。"""
        parts = [
            part
            for part in [self._oss_object_prefix, uploader_user_id, file_id, safe_filename]
            if part
        ]
        return "/".join(parts)

    def _resolve_visibility_scope(self, current_user: CurrentUser) -> FileVisibilityScope:
        """根据上传者角色决定文件可见性。"""
        if current_user.role is UserRole.ADMIN:
            return FileVisibilityScope.GLOBAL
        return FileVisibilityScope.OWNER_ONLY

    def _build_uploaded_result(
        self,
        knowledge_file: KnowledgeFile,
        *,
        deduplicated: bool = False,
        replaced: bool = False,
        title_updated: bool = False,
    ) -> UploadedFileResult:
        """把文件元数据转换为上传接口返回结构。"""
        return UploadedFileResult(
            file_id=knowledge_file.id,
            original_filename=knowledge_file.original_filename,
            content_type=knowledge_file.content_type,
            size=knowledge_file.size,
            storage_key=knowledge_file.storage_key,
            uploaded_at=knowledge_file.uploaded_at,
            chunk_count=knowledge_file.chunk_count,
            storage_provider=knowledge_file.storage_provider.value,
            deduplicated=deduplicated,
            replaced=replaced,
            title_updated=title_updated,
        )

    def _rollback_actions(self, actions: list[object]) -> None:
        """在批量处理失败时回滚已完成动作。"""
        for action in reversed(actions):
            if isinstance(action, _NewUploadAction):
                self._rollback_new_upload(action)
                continue
            if isinstance(action, _TitleUpdateAction):
                self._rollback_title_update(action)
                continue
            if isinstance(action, _ReplacementAction):
                self._rollback_replacement(action)

    def _rollback_new_upload(self, action: _NewUploadAction) -> None:
        """回滚全新上传。"""
        with suppress(Exception):
            self._knowledge_file_repository.delete_file(action.knowledge_file.id)
        with suppress(Exception):
            self._chunk_store.delete_chunks_by_file_id(action.knowledge_file.id)
        with suppress(Exception):
            self._object_store.delete(action.knowledge_file.storage_key)

    def _rollback_title_update(self, action: _TitleUpdateAction) -> None:
        """回滚仅标题更新。"""
        with suppress(Exception):
            self._knowledge_file_repository.update_file(
                action.file_id,
                original_filename=action.previous_filename,
            )

    def _rollback_replacement(self, action: _ReplacementAction) -> None:
        """回滚覆盖更新。"""
        with suppress(Exception):
            self._knowledge_file_repository.delete_file(action.new_file.id)
        with suppress(Exception):
            self._chunk_store.delete_chunks_by_file_id(action.new_file.id)
        with suppress(Exception):
            self._object_store.delete(action.new_file.storage_key)
        with suppress(Exception):
            self._knowledge_file_repository.create_file(action.previous_file)

    def _finalize_actions(self, actions: list[object]) -> None:
        """在成功响应前完成替换旧资源的清理。"""
        for action in actions:
            if not isinstance(action, _ReplacementAction):
                continue
            with suppress(Exception):
                self._chunk_store.delete_chunks_by_file_id(action.previous_file.id)
            with suppress(Exception):
                self._object_store.delete(action.previous_file.storage_key)

    def _cleanup_staged_files(self, staged_files: list[StagedUploadFileResult]) -> None:
        """清理本次请求产生的所有临时文件。"""
        for staged_file in reversed(staged_files):
            with suppress(Exception):
                self._temp_file_store.delete(staged_file.temp_storage_key)
