"""基于 SQLAlchemy 的知识文件图片资产仓储实现。"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from baozhi_rag.domain.knowledge_file_image_asset import KnowledgeFileImageAsset
from baozhi_rag.infra.database.models import KnowledgeFileImageAssetModel


class SqlAlchemyKnowledgeFileImageAssetRepository:
    """知识文件图片资产仓储的 SQLAlchemy 实现。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create_assets(
        self,
        assets: list[KnowledgeFileImageAsset],
    ) -> list[KnowledgeFileImageAsset]:
        """批量创建图片资产。"""
        if not assets:
            return []

        models = [self._to_model(asset) for asset in assets]
        with self._session_factory() as session:
            session.add_all(models)
            session.commit()
            for model in models:
                session.refresh(model)
            return [self._to_domain(model) for model in models]

    def list_assets_by_file_id(self, file_id: str) -> list[KnowledgeFileImageAsset]:
        """按文件 ID 查询图片资产。"""
        with self._session_factory() as session:
            stmt = (
                select(KnowledgeFileImageAssetModel)
                .where(KnowledgeFileImageAssetModel.file_id == file_id)
                .order_by(
                    KnowledgeFileImageAssetModel.chunk_index.asc(),
                    KnowledgeFileImageAssetModel.asset_index.asc(),
                    KnowledgeFileImageAssetModel.id.asc(),
                )
            )
            return [self._to_domain(model) for model in session.scalars(stmt).all()]

    def list_assets_by_chunk_ids(self, chunk_ids: list[str]) -> list[KnowledgeFileImageAsset]:
        """按 chunk ID 列表查询图片资产。"""
        if not chunk_ids:
            return []

        with self._session_factory() as session:
            stmt = (
                select(KnowledgeFileImageAssetModel)
                .where(KnowledgeFileImageAssetModel.chunk_id.in_(chunk_ids))
                .order_by(
                    KnowledgeFileImageAssetModel.chunk_index.asc(),
                    KnowledgeFileImageAssetModel.asset_index.asc(),
                    KnowledgeFileImageAssetModel.id.asc(),
                )
            )
            return [self._to_domain(model) for model in session.scalars(stmt).all()]

    def get_asset_by_uploader_and_normalized_sha256(
        self,
        uploader_user_id: str,
        normalized_image_sha256: str,
    ) -> KnowledgeFileImageAsset | None:
        """按上传者和归一化图片哈希查询可复用图片资产。"""
        with self._session_factory() as session:
            stmt = (
                select(KnowledgeFileImageAssetModel)
                .where(
                    KnowledgeFileImageAssetModel.uploader_user_id == uploader_user_id,
                    KnowledgeFileImageAssetModel.normalized_image_sha256 == normalized_image_sha256,
                )
                .order_by(KnowledgeFileImageAssetModel.updated_at.desc())
            )
            model = session.scalar(stmt)
            return self._to_domain(model) if model is not None else None

    def delete_assets_by_file_id(self, file_id: str) -> int:
        """按文件 ID 删除图片资产。"""
        existing_assets = self.list_assets_by_file_id(file_id)
        with self._session_factory() as session:
            session.execute(
                delete(KnowledgeFileImageAssetModel).where(
                    KnowledgeFileImageAssetModel.file_id == file_id
                )
            )
            session.commit()
            return len(existing_assets)

    def _to_model(self, asset: KnowledgeFileImageAsset) -> KnowledgeFileImageAssetModel:
        """把领域对象转换为 ORM 模型。"""
        return KnowledgeFileImageAssetModel(
            id=asset.id,
            file_id=asset.file_id,
            chunk_id=asset.chunk_id,
            chunk_index=asset.chunk_index,
            asset_index=asset.asset_index,
            uploader_user_id=asset.uploader_user_id,
            source_anchor=asset.source_anchor,
            image_sha256=asset.image_sha256,
            normalized_image_sha256=asset.normalized_image_sha256,
            storage_key=asset.storage_key,
            thumbnail_storage_key=asset.thumbnail_storage_key,
            content_type=asset.content_type,
            width=asset.width,
            height=asset.height,
            ocr_text=asset.ocr_text,
            summary=asset.summary,
            image_type=asset.image_type,
            recognition_model=asset.recognition_model,
            created_at=asset.created_at,
            updated_at=asset.updated_at,
        )

    def _to_domain(self, model: KnowledgeFileImageAssetModel) -> KnowledgeFileImageAsset:
        """把 ORM 模型转换为领域对象。"""
        return KnowledgeFileImageAsset(
            id=model.id,
            file_id=model.file_id,
            chunk_id=model.chunk_id,
            chunk_index=model.chunk_index,
            asset_index=model.asset_index,
            uploader_user_id=model.uploader_user_id,
            source_anchor=model.source_anchor,
            image_sha256=model.image_sha256,
            normalized_image_sha256=model.normalized_image_sha256,
            storage_key=model.storage_key,
            thumbnail_storage_key=model.thumbnail_storage_key,
            content_type=model.content_type,
            width=model.width,
            height=model.height,
            ocr_text=model.ocr_text,
            summary=model.summary,
            image_type=model.image_type,
            recognition_model=model.recognition_model,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
