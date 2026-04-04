"""基于 SQLAlchemy 的知识文件上传任务仓储实现。"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, sessionmaker

from baozhi_rag.domain.knowledge_file_errors import (
    KnowledgeUploadTaskRetryNotAllowedError,
)
from baozhi_rag.domain.knowledge_upload_task import (
    KnowledgeUploadTask,
    KnowledgeUploadTaskStage,
    KnowledgeUploadTaskStatus,
)
from baozhi_rag.infra.database.models import KnowledgeUploadTaskModel


class SqlAlchemyKnowledgeUploadTaskRepository:
    """上传任务仓储的 SQLAlchemy 实现。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """初始化仓储。"""
        self._session_factory = session_factory

    def create_task(self, task: KnowledgeUploadTask) -> KnowledgeUploadTask:
        """创建上传任务。"""
        task_model = self._to_model(task)
        with self._session_factory() as session:
            session.add(task_model)
            session.commit()
            session.refresh(task_model)
            return self._to_domain(task_model)

    def get_task_by_id(self, task_id: str) -> KnowledgeUploadTask | None:
        """按任务标识查询任务。"""
        with self._session_factory() as session:
            task_model = session.get(KnowledgeUploadTaskModel, task_id)
            return self._to_domain(task_model) if task_model is not None else None

    def get_task_by_id_for_user(
        self,
        task_id: str,
        uploader_user_id: str,
    ) -> KnowledgeUploadTask | None:
        """按任务标识和上传用户查询任务。"""
        with self._session_factory() as session:
            stmt = select(KnowledgeUploadTaskModel).where(
                KnowledgeUploadTaskModel.id == task_id,
                KnowledgeUploadTaskModel.uploader_user_id == uploader_user_id,
            )
            task_model = session.scalar(stmt)
            return self._to_domain(task_model) if task_model is not None else None

    def get_task_by_user_and_raw_sha256(
        self,
        uploader_user_id: str,
        raw_sha256: str,
        ingest_version: str,
    ) -> KnowledgeUploadTask | None:
        """按用户、原始哈希和 ingest 版本查询任务。"""
        with self._session_factory() as session:
            stmt = select(KnowledgeUploadTaskModel).where(
                KnowledgeUploadTaskModel.uploader_user_id == uploader_user_id,
                KnowledgeUploadTaskModel.raw_sha256 == raw_sha256,
                KnowledgeUploadTaskModel.ingest_version == ingest_version,
            )
            task_model = session.scalar(stmt)
            return self._to_domain(task_model) if task_model is not None else None

    def list_tasks_by_user(
        self,
        uploader_user_id: str,
        *,
        limit: int,
    ) -> list[KnowledgeUploadTask]:
        """按用户倒序列出最近任务。"""
        with self._session_factory() as session:
            stmt = (
                select(KnowledgeUploadTaskModel)
                .where(KnowledgeUploadTaskModel.uploader_user_id == uploader_user_id)
                .order_by(
                    KnowledgeUploadTaskModel.created_at.desc(),
                    KnowledgeUploadTaskModel.id.desc(),
                )
                .limit(limit)
            )
            task_models = session.scalars(stmt).all()
            return [self._to_domain(task_model) for task_model in task_models]

    def update_submission_context(
        self,
        task_id: str,
        *,
        requested_filename: str,
        source_storage_key: str | None = None,
    ) -> KnowledgeUploadTask | None:
        """更新任务最近一次提交使用的标题与源文件位置。"""
        with self._session_factory() as session:
            task_model = session.get(KnowledgeUploadTaskModel, task_id)
            if task_model is None:
                return None
            task_model.requested_filename = requested_filename
            if source_storage_key is not None:
                task_model.source_storage_key = source_storage_key
            task_model.updated_at = datetime.now(UTC)
            session.commit()
            session.refresh(task_model)
            return self._to_domain(task_model)

    def claim_next_task(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_expires_at: datetime,
    ) -> KnowledgeUploadTask | None:
        """抢占一条待处理或租约过期的任务。"""
        with self._session_factory() as session:
            stmt = (
                select(KnowledgeUploadTaskModel)
                .where(
                    or_(
                        KnowledgeUploadTaskModel.status == KnowledgeUploadTaskStatus.QUEUED.value,
                        and_(
                            KnowledgeUploadTaskModel.status
                            == KnowledgeUploadTaskStatus.PROCESSING.value,
                            KnowledgeUploadTaskModel.lease_expires_at.is_not(None),
                            KnowledgeUploadTaskModel.lease_expires_at < now,
                        ),
                    )
                )
                .order_by(
                    KnowledgeUploadTaskModel.created_at.asc(),
                    KnowledgeUploadTaskModel.id.asc(),
                )
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            task_model = session.scalar(stmt)
            if task_model is None:
                session.rollback()
                return None

            task_model.status = KnowledgeUploadTaskStatus.PROCESSING.value
            task_model.worker_id = worker_id
            task_model.lease_expires_at = lease_expires_at
            task_model.last_heartbeat_at = now
            task_model.updated_at = now
            task_model.attempt_count += 1
            task_model.error_code = None
            task_model.error_message = None
            session.commit()
            session.refresh(task_model)
            return self._to_domain(task_model)

    def refresh_lease(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime,
        heartbeat_at: datetime,
    ) -> bool:
        """刷新任务租约与心跳。"""
        with self._session_factory() as session:
            task_model = session.get(KnowledgeUploadTaskModel, task_id)
            if (
                task_model is None
                or task_model.worker_id != worker_id
                or task_model.status != KnowledgeUploadTaskStatus.PROCESSING.value
            ):
                return False

            task_model.lease_expires_at = lease_expires_at
            task_model.last_heartbeat_at = heartbeat_at
            task_model.updated_at = heartbeat_at
            session.commit()
            return True

    def update_task_progress(
        self,
        task_id: str,
        *,
        worker_id: str,
        stage: KnowledgeUploadTaskStage,
        status: KnowledgeUploadTaskStatus = KnowledgeUploadTaskStatus.PROCESSING,
        content_sha256: str | None = None,
        file_id: str | None = None,
        chunk_count: int | None = None,
        deduplicated: bool | None = None,
        replaced: bool | None = None,
        title_updated: bool | None = None,
    ) -> KnowledgeUploadTask | None:
        """更新任务阶段性处理结果。"""
        with self._session_factory() as session:
            task_model = session.get(KnowledgeUploadTaskModel, task_id)
            if task_model is None or task_model.worker_id != worker_id:
                return None

            task_model.stage = stage.value
            task_model.status = status.value
            if content_sha256 is not None:
                task_model.content_sha256 = content_sha256
            if file_id is not None:
                task_model.file_id = file_id
            if chunk_count is not None:
                task_model.chunk_count = chunk_count
            if deduplicated is not None:
                task_model.deduplicated = deduplicated
            if replaced is not None:
                task_model.replaced = replaced
            if title_updated is not None:
                task_model.title_updated = title_updated
            task_model.updated_at = datetime.now(UTC)
            session.commit()
            session.refresh(task_model)
            return self._to_domain(task_model)

    def mark_succeeded(
        self,
        task_id: str,
        *,
        worker_id: str,
        stage: KnowledgeUploadTaskStage,
        content_sha256: str | None,
        file_id: str | None,
        chunk_count: int,
        deduplicated: bool,
        replaced: bool,
        title_updated: bool,
        completed_at: datetime,
    ) -> KnowledgeUploadTask | None:
        """把任务标记为成功。"""
        with self._session_factory() as session:
            task_model = session.get(KnowledgeUploadTaskModel, task_id)
            if task_model is None or task_model.worker_id != worker_id:
                return None

            task_model.status = KnowledgeUploadTaskStatus.SUCCEEDED.value
            task_model.stage = stage.value
            task_model.content_sha256 = content_sha256
            task_model.file_id = file_id
            task_model.chunk_count = chunk_count
            task_model.deduplicated = deduplicated
            task_model.replaced = replaced
            task_model.title_updated = title_updated
            task_model.completed_at = completed_at
            task_model.worker_id = None
            task_model.lease_expires_at = None
            task_model.last_heartbeat_at = completed_at
            task_model.updated_at = completed_at
            session.commit()
            session.refresh(task_model)
            return self._to_domain(task_model)

    def mark_failed(
        self,
        task_id: str,
        *,
        worker_id: str,
        error_code: str,
        error_message: str,
        failed_at: datetime,
    ) -> KnowledgeUploadTask | None:
        """把任务标记为失败。"""
        with self._session_factory() as session:
            task_model = session.get(KnowledgeUploadTaskModel, task_id)
            if task_model is None or task_model.worker_id != worker_id:
                return None

            task_model.status = KnowledgeUploadTaskStatus.FAILED.value
            task_model.stage = KnowledgeUploadTaskStage.FAILED.value
            task_model.error_code = error_code
            task_model.error_message = error_message
            task_model.completed_at = failed_at
            task_model.worker_id = None
            task_model.lease_expires_at = None
            task_model.last_heartbeat_at = failed_at
            task_model.updated_at = failed_at
            session.commit()
            session.refresh(task_model)
            return self._to_domain(task_model)

    def retry_task(
        self,
        task_id: str,
        *,
        uploader_user_id: str,
        queued_at: datetime,
    ) -> KnowledgeUploadTask | None:
        """将失败任务重新入队。"""
        with self._session_factory() as session:
            task_model = session.get(KnowledgeUploadTaskModel, task_id)
            if task_model is None or task_model.uploader_user_id != uploader_user_id:
                return None
            if task_model.status != KnowledgeUploadTaskStatus.FAILED.value:
                raise KnowledgeUploadTaskRetryNotAllowedError()

            task_model.status = KnowledgeUploadTaskStatus.QUEUED.value
            task_model.stage = KnowledgeUploadTaskStage.UPLOADED.value
            task_model.content_sha256 = None
            task_model.file_id = None
            task_model.chunk_count = 0
            task_model.deduplicated = False
            task_model.replaced = False
            task_model.title_updated = False
            task_model.error_code = None
            task_model.error_message = None
            task_model.worker_id = None
            task_model.lease_expires_at = None
            task_model.last_heartbeat_at = None
            task_model.completed_at = None
            task_model.updated_at = queued_at
            session.commit()
            session.refresh(task_model)
            return self._to_domain(task_model)

    def _to_model(self, task: KnowledgeUploadTask) -> KnowledgeUploadTaskModel:
        """把领域对象转换为 ORM 模型。"""
        return KnowledgeUploadTaskModel(
            id=task.id,
            request_id=task.request_id,
            uploader_user_id=task.uploader_user_id,
            uploader_role=task.uploader_role,
            raw_sha256=task.raw_sha256,
            content_sha256=task.content_sha256,
            source_storage_key=task.source_storage_key,
            requested_filename=task.requested_filename,
            content_type=task.content_type,
            size=task.size,
            ingest_version=task.ingest_version,
            status=task.status.value,
            stage=task.stage.value,
            file_id=task.file_id,
            chunk_count=task.chunk_count,
            deduplicated=task.deduplicated,
            replaced=task.replaced,
            title_updated=task.title_updated,
            error_code=task.error_code,
            error_message=task.error_message,
            attempt_count=task.attempt_count,
            worker_id=task.worker_id,
            lease_expires_at=task.lease_expires_at,
            last_heartbeat_at=task.last_heartbeat_at,
            created_at=task.created_at,
            updated_at=task.updated_at,
            completed_at=task.completed_at,
        )

    def _to_domain(self, task_model: KnowledgeUploadTaskModel) -> KnowledgeUploadTask:
        """把 ORM 模型转换为领域对象。"""
        return KnowledgeUploadTask(
            id=task_model.id,
            request_id=task_model.request_id,
            uploader_user_id=task_model.uploader_user_id,
            uploader_role=task_model.uploader_role,
            raw_sha256=task_model.raw_sha256,
            source_storage_key=task_model.source_storage_key,
            requested_filename=task_model.requested_filename,
            content_type=task_model.content_type,
            size=task_model.size,
            ingest_version=task_model.ingest_version,
            status=KnowledgeUploadTaskStatus(task_model.status),
            stage=KnowledgeUploadTaskStage(task_model.stage),
            content_sha256=task_model.content_sha256,
            file_id=task_model.file_id,
            chunk_count=task_model.chunk_count,
            deduplicated=task_model.deduplicated,
            replaced=task_model.replaced,
            title_updated=task_model.title_updated,
            error_code=task_model.error_code,
            error_message=task_model.error_message,
            attempt_count=task_model.attempt_count,
            worker_id=task_model.worker_id,
            lease_expires_at=task_model.lease_expires_at,
            last_heartbeat_at=task_model.last_heartbeat_at,
            created_at=task_model.created_at,
            updated_at=task_model.updated_at,
            completed_at=task_model.completed_at,
        )
