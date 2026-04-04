"""基于 SQLAlchemy 的知识文件仓储实现。"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from baozhi_rag.domain.knowledge_file import (
    FileStorageProvider,
    FileVisibilityScope,
    KnowledgeFile,
)
from baozhi_rag.domain.knowledge_file_errors import (
    KnowledgeFileConflictError,
    KnowledgeFileNotFoundError,
)
from baozhi_rag.infra.database.models import KnowledgeFileModel


class SqlAlchemyKnowledgeFileRepository:
    """知识文件仓储的 SQLAlchemy 实现。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """初始化仓储。"""
        self._session_factory = session_factory

    def create_file(self, file: KnowledgeFile) -> KnowledgeFile:
        """创建文件记录。"""
        file_model = self._to_model(file)
        with self._session_factory() as session:
            session.add(file_model)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                self._raise_conflict_error(exc)
            session.refresh(file_model)
            return self._to_domain(file_model)

    def get_file_by_id(self, file_id: str) -> KnowledgeFile | None:
        """按文件 ID 查询文件。"""
        with self._session_factory() as session:
            file_model = session.get(KnowledgeFileModel, file_id)
            return self._to_domain(file_model) if file_model is not None else None

    def get_file_by_user_and_filename(
        self,
        uploader_user_id: str,
        original_filename: str,
    ) -> KnowledgeFile | None:
        """按上传者和文件名查询文件。"""
        with self._session_factory() as session:
            stmt = select(KnowledgeFileModel).where(
                KnowledgeFileModel.uploader_user_id == uploader_user_id,
                KnowledgeFileModel.original_filename == original_filename,
            )
            file_model = session.scalar(stmt)
            return self._to_domain(file_model) if file_model is not None else None

    def get_file_by_user_and_sha256(
        self,
        uploader_user_id: str,
        sha256: str,
    ) -> KnowledgeFile | None:
        """按上传者和内容哈希查询文件。"""
        with self._session_factory() as session:
            stmt = select(KnowledgeFileModel).where(
                KnowledgeFileModel.uploader_user_id == uploader_user_id,
                KnowledgeFileModel.sha256 == sha256,
            )
            file_model = session.scalar(stmt)
            return self._to_domain(file_model) if file_model is not None else None

    def get_files_by_ids(self, file_ids: list[str]) -> list[KnowledgeFile]:
        """批量查询文件元数据。"""
        if not file_ids:
            return []

        with self._session_factory() as session:
            stmt = select(KnowledgeFileModel).where(KnowledgeFileModel.id.in_(file_ids))
            file_models = session.scalars(stmt).all()

        file_map = {file_model.id: self._to_domain(file_model) for file_model in file_models}
        return [file_map[file_id] for file_id in file_ids if file_id in file_map]

    def update_file(
        self,
        file_id: str,
        *,
        original_filename: str | None = None,
        content_type: str | None = None,
        size: int | None = None,
        sha256: str | None = None,
        storage_provider: FileStorageProvider | None = None,
        storage_key: str | None = None,
        visibility_scope: FileVisibilityScope | None = None,
        chunk_count: int | None = None,
    ) -> KnowledgeFile | None:
        """更新文件记录。"""
        with self._session_factory() as session:
            file_model = session.get(KnowledgeFileModel, file_id)
            if file_model is None:
                return None

            if original_filename is not None:
                file_model.original_filename = original_filename
            if content_type is not None:
                file_model.content_type = content_type
            if size is not None:
                file_model.size = size
            if sha256 is not None:
                file_model.sha256 = sha256
            if storage_provider is not None:
                file_model.storage_provider = storage_provider.value
            if storage_key is not None:
                file_model.storage_key = storage_key
            if visibility_scope is not None:
                file_model.visibility_scope = visibility_scope.value
            if chunk_count is not None:
                file_model.chunk_count = chunk_count
            file_model.uploaded_at = datetime.now(UTC)
            file_model.updated_at = datetime.now(UTC)

            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                self._raise_conflict_error(exc)
            session.refresh(file_model)
            return self._to_domain(file_model)

    def replace_file(
        self,
        existing_file_id: str,
        replacement_file: KnowledgeFile,
    ) -> KnowledgeFile:
        """以新文件记录替换旧文件记录。"""
        with self._session_factory() as session:
            existing_model = session.get(KnowledgeFileModel, existing_file_id)
            if existing_model is None:
                raise KnowledgeFileNotFoundError()

            session.delete(existing_model)
            session.flush()

            replacement_model = self._to_model(replacement_file)
            session.add(replacement_model)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                self._raise_conflict_error(exc)
            session.refresh(replacement_model)
            return self._to_domain(replacement_model)

    def delete_file(self, file_id: str) -> bool:
        """删除文件记录。"""
        with self._session_factory() as session:
            file_model = session.get(KnowledgeFileModel, file_id)
            if file_model is None:
                return False
            session.delete(file_model)
            session.commit()
            return True

    def _raise_conflict_error(self, exc: IntegrityError) -> None:
        """将数据库唯一约束冲突翻译为领域错误。"""
        message = str(exc.orig).lower()
        if "uq_knowledge_files_uploader_filename" in message or "original_filename" in message:
            raise KnowledgeFileConflictError("同一用户的同名文件记录冲突") from exc
        raise KnowledgeFileConflictError() from exc

    def _to_model(self, file: KnowledgeFile) -> KnowledgeFileModel:
        """把领域对象转换为 ORM 模型。"""
        return KnowledgeFileModel(
            id=file.id,
            uploader_user_id=file.uploader_user_id,
            original_filename=file.original_filename,
            content_type=file.content_type,
            size=file.size,
            sha256=file.sha256,
            storage_provider=file.storage_provider.value,
            storage_key=file.storage_key,
            visibility_scope=file.visibility_scope.value,
            chunk_count=file.chunk_count,
            uploaded_at=file.uploaded_at,
            updated_at=file.updated_at,
        )

    def _to_domain(self, file_model: KnowledgeFileModel) -> KnowledgeFile:
        """把 ORM 模型转换为领域对象。"""
        return KnowledgeFile(
            id=file_model.id,
            uploader_user_id=file_model.uploader_user_id,
            original_filename=file_model.original_filename,
            content_type=file_model.content_type,
            size=file_model.size,
            sha256=file_model.sha256,
            storage_provider=FileStorageProvider(file_model.storage_provider),
            storage_key=file_model.storage_key,
            visibility_scope=FileVisibilityScope(file_model.visibility_scope),
            chunk_count=file_model.chunk_count,
            uploaded_at=file_model.uploaded_at,
            updated_at=file_model.updated_at,
        )

