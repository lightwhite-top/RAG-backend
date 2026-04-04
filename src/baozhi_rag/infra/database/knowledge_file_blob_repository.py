"""基于 SQLAlchemy 的原始文件 blob 仓储实现。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from baozhi_rag.domain.knowledge_file import FileStorageProvider
from baozhi_rag.domain.knowledge_file_blob import KnowledgeFileBlob
from baozhi_rag.domain.knowledge_file_errors import KnowledgeFileConflictError
from baozhi_rag.infra.database.models import KnowledgeFileBlobModel


class SqlAlchemyKnowledgeFileBlobRepository:
    """原始文件 blob 仓储的 SQLAlchemy 实现。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """初始化仓储。"""
        self._session_factory = session_factory

    def create_blob(self, blob: KnowledgeFileBlob) -> KnowledgeFileBlob:
        """创建 blob 记录。"""
        blob_model = self._to_model(blob)
        with self._session_factory() as session:
            session.add(blob_model)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise KnowledgeFileConflictError("原始文件 blob 记录冲突") from exc
            session.refresh(blob_model)
            return self._to_domain(blob_model)

    def get_blob_by_raw_sha256(self, raw_sha256: str) -> KnowledgeFileBlob | None:
        """按原始哈希查询 blob。"""
        with self._session_factory() as session:
            stmt = select(KnowledgeFileBlobModel).where(
                KnowledgeFileBlobModel.raw_sha256 == raw_sha256
            )
            blob_model = session.scalar(stmt)
            return self._to_domain(blob_model) if blob_model is not None else None

    def _to_model(self, blob: KnowledgeFileBlob) -> KnowledgeFileBlobModel:
        """把领域对象转换为 ORM 模型。"""
        return KnowledgeFileBlobModel(
            id=blob.id,
            raw_sha256=blob.raw_sha256,
            content_type=blob.content_type,
            size=blob.size,
            storage_provider=blob.storage_provider.value,
            storage_key=blob.storage_key,
            created_at=blob.created_at,
            updated_at=blob.updated_at,
        )

    def _to_domain(self, blob_model: KnowledgeFileBlobModel) -> KnowledgeFileBlob:
        """把 ORM 模型转换为领域对象。"""
        return KnowledgeFileBlob(
            id=blob_model.id,
            raw_sha256=blob_model.raw_sha256,
            content_type=blob_model.content_type,
            size=blob_model.size,
            storage_provider=FileStorageProvider(blob_model.storage_provider),
            storage_key=blob_model.storage_key,
            created_at=blob_model.created_at,
            updated_at=blob_model.updated_at,
        )
