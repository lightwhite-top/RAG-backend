"""基于 SQLAlchemy 的聊天消息仓储实现。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from baozhi_rag.domain.chat_errors import ChatMessageConflictError, ChatSessionNotFoundError
from baozhi_rag.domain.chat_message import (
    ChatMessageCitationRecord,
    ChatMessageRecord,
    ChatMessageRole,
    ChatMessageStatus,
)
from baozhi_rag.domain.chat_session import ChatSessionStatus
from baozhi_rag.infra.database.models import (
    ChatMessageCitationModel,
    ChatMessageModel,
    ChatSessionModel,
)


class SqlAlchemyChatMessageRepository:
    """聊天消息仓储的 SQLAlchemy 实现。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def append_message(
        self,
        *,
        session_id: str,
        role: ChatMessageRole,
        status: ChatMessageStatus,
        plain_text: str,
        content_blocks: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        model_name: str | None = None,
        original_query: str | None = None,
        retrieval_query: str | None = None,
        rewrite_applied: bool = False,
        retrieval_size: int | None = None,
        temperature: float | None = None,
        finish_reason: str | None = None,
        latency_ms: int | None = None,
        usage: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> ChatMessageRecord:
        with self._session_factory() as db_session:
            session_model = self._lock_session(db_session, session_id)
            now = datetime.now(UTC)
            next_sequence_no = (
                int(
                    db_session.scalar(
                        select(func.max(ChatMessageModel.sequence_no)).where(
                            ChatMessageModel.session_id == session_id
                        )
                    )
                    or 0
                )
                + 1
            )
            message_model = ChatMessageModel(
                id=uuid4().hex,
                session_id=session_id,
                sequence_no=next_sequence_no,
                role=role.value,
                status=status.value,
                plain_text=plain_text,
                content_blocks_json=content_blocks,
                request_id=request_id,
                model_name=model_name,
                original_query=original_query,
                retrieval_query=retrieval_query,
                rewrite_applied=rewrite_applied,
                retrieval_size=retrieval_size,
                temperature=temperature,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                usage_json=usage,
                error_code=error_code,
                error_message=error_message,
                created_at=now,
                updated_at=now,
                completed_at=now if status is not ChatMessageStatus.STREAMING else None,
            )
            db_session.add(message_model)

            session_model.message_count += 1
            session_model.last_message_at = now
            session_model.updated_at = now
            if role is ChatMessageRole.USER:
                session_model.last_user_message_at = now
            elif role is ChatMessageRole.ASSISTANT:
                session_model.last_assistant_message_at = now

            try:
                db_session.commit()
            except IntegrityError as exc:
                db_session.rollback()
                raise ChatMessageConflictError() from exc
            db_session.refresh(message_model)
            return self._to_domain(message_model, [])

    def update_message(
        self,
        message_id: str,
        *,
        status: ChatMessageStatus | None = None,
        plain_text: str | None = None,
        content_blocks: list[dict[str, Any]] | None = None,
        model_name: str | None = None,
        original_query: str | None = None,
        retrieval_query: str | None = None,
        rewrite_applied: bool | None = None,
        retrieval_size: int | None = None,
        temperature: float | None = None,
        finish_reason: str | None = None,
        latency_ms: int | None = None,
        usage: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        completed: bool = False,
    ) -> ChatMessageRecord | None:
        with self._session_factory() as db_session:
            message_model = db_session.get(ChatMessageModel, message_id)
            if message_model is None:
                return None

            now = datetime.now(UTC)
            if status is not None:
                message_model.status = status.value
            if plain_text is not None:
                message_model.plain_text = plain_text
            if content_blocks is not None:
                message_model.content_blocks_json = content_blocks
            if model_name is not None:
                message_model.model_name = model_name
            if original_query is not None:
                message_model.original_query = original_query
            if retrieval_query is not None:
                message_model.retrieval_query = retrieval_query
            if rewrite_applied is not None:
                message_model.rewrite_applied = rewrite_applied
            if retrieval_size is not None:
                message_model.retrieval_size = retrieval_size
            message_model.temperature = temperature
            if finish_reason is not None:
                message_model.finish_reason = finish_reason
            if latency_ms is not None:
                message_model.latency_ms = latency_ms
            if usage is not None:
                message_model.usage_json = usage
            if error_code is not None:
                message_model.error_code = error_code
            if error_message is not None:
                message_model.error_message = error_message
            if completed:
                message_model.completed_at = now
            message_model.updated_at = now

            db_session.commit()
            db_session.refresh(message_model)
            citations = self._load_citations(db_session, [message_model.id]).get(
                message_model.id, []
            )
            return self._to_domain(message_model, citations)

    def replace_citations(
        self,
        message_id: str,
        citations: list[ChatMessageCitationRecord],
    ) -> list[ChatMessageCitationRecord]:
        with self._session_factory() as db_session:
            message_model = db_session.get(ChatMessageModel, message_id)
            if message_model is None:
                return []

            db_session.execute(
                delete(ChatMessageCitationModel).where(
                    ChatMessageCitationModel.message_id == message_id
                )
            )
            now = datetime.now(UTC)
            citation_records: list[ChatMessageCitationRecord] = []
            for citation in citations:
                record = ChatMessageCitationRecord(
                    id=citation.id or uuid4().hex,
                    message_id=message_id,
                    citation_index=citation.citation_index,
                    chunk_id=citation.chunk_id,
                    file_id=citation.file_id,
                    source_filename=citation.source_filename,
                    storage_key=citation.storage_key,
                    chunk_index=citation.chunk_index,
                    char_count=citation.char_count,
                    content=citation.content,
                    snippet=citation.snippet,
                    merged_terms=list(citation.merged_terms),
                    score=citation.score,
                    heading_path=list(citation.heading_path),
                    section_title=citation.section_title,
                    content_type=citation.content_type,
                    source_anchor=citation.source_anchor,
                    created_at=now,
                )
                db_session.add(self._citation_to_model(record))
                citation_records.append(record)

            message_model.updated_at = now
            db_session.commit()
            return citation_records

    def list_messages(
        self,
        *,
        session_id: str,
        before_sequence_no: int | None,
        limit: int,
    ) -> list[ChatMessageRecord]:
        with self._session_factory() as db_session:
            stmt = select(ChatMessageModel).where(ChatMessageModel.session_id == session_id)
            if before_sequence_no is not None:
                stmt = stmt.where(ChatMessageModel.sequence_no < before_sequence_no)
            stmt = stmt.order_by(ChatMessageModel.sequence_no.desc()).limit(limit)
            message_models = list(db_session.scalars(stmt).all())
            message_models.reverse()
            citations_by_message_id = self._load_citations(
                db_session,
                [message_model.id for message_model in message_models],
            )
            return [
                self._to_domain(message_model, citations_by_message_id.get(message_model.id, []))
                for message_model in message_models
            ]

    def get_recent_messages(
        self,
        *,
        session_id: str,
        limit: int,
    ) -> list[ChatMessageRecord]:
        return self.list_messages(session_id=session_id, before_sequence_no=None, limit=limit)

    def _lock_session(self, db_session: Session, session_id: str) -> ChatSessionModel:
        stmt = select(ChatSessionModel).where(ChatSessionModel.id == session_id).with_for_update()
        session_model = db_session.scalar(stmt)
        if session_model is None or session_model.status == ChatSessionStatus.DELETED.value:
            raise ChatSessionNotFoundError()
        return session_model

    def _load_citations(
        self,
        db_session: Session,
        message_ids: list[str],
    ) -> dict[str, list[ChatMessageCitationRecord]]:
        if not message_ids:
            return {}

        stmt = (
            select(ChatMessageCitationModel)
            .where(ChatMessageCitationModel.message_id.in_(message_ids))
            .order_by(ChatMessageCitationModel.message_id, ChatMessageCitationModel.citation_index)
        )
        result: dict[str, list[ChatMessageCitationRecord]] = {
            message_id: [] for message_id in message_ids
        }
        for citation_model in db_session.scalars(stmt).all():
            result.setdefault(citation_model.message_id, []).append(
                self._citation_to_domain(citation_model)
            )
        return result

    def _to_domain(
        self,
        message_model: ChatMessageModel,
        citations: list[ChatMessageCitationRecord],
    ) -> ChatMessageRecord:
        return ChatMessageRecord(
            id=message_model.id,
            session_id=message_model.session_id,
            sequence_no=message_model.sequence_no,
            role=ChatMessageRole(message_model.role),
            status=ChatMessageStatus(message_model.status),
            plain_text=message_model.plain_text,
            content_blocks=list(message_model.content_blocks_json or []),
            request_id=message_model.request_id,
            model_name=message_model.model_name,
            original_query=message_model.original_query,
            retrieval_query=message_model.retrieval_query,
            rewrite_applied=message_model.rewrite_applied,
            retrieval_size=message_model.retrieval_size,
            temperature=message_model.temperature,
            finish_reason=message_model.finish_reason,
            latency_ms=message_model.latency_ms,
            usage=message_model.usage_json,
            error_code=message_model.error_code,
            error_message=message_model.error_message,
            created_at=message_model.created_at,
            updated_at=message_model.updated_at,
            completed_at=message_model.completed_at,
            citations=citations,
        )

    def _citation_to_model(
        self,
        citation: ChatMessageCitationRecord,
    ) -> ChatMessageCitationModel:
        return ChatMessageCitationModel(
            id=citation.id,
            message_id=citation.message_id,
            citation_index=citation.citation_index,
            chunk_id=citation.chunk_id,
            file_id=citation.file_id,
            source_filename=citation.source_filename,
            storage_key=citation.storage_key,
            chunk_index=citation.chunk_index,
            char_count=citation.char_count,
            content=citation.content,
            snippet=citation.snippet,
            merged_terms_json=citation.merged_terms,
            score=citation.score,
            heading_path_json=citation.heading_path,
            section_title=citation.section_title,
            content_type=citation.content_type,
            source_anchor=citation.source_anchor,
            created_at=citation.created_at,
        )

    def _citation_to_domain(
        self,
        citation_model: ChatMessageCitationModel,
    ) -> ChatMessageCitationRecord:
        return ChatMessageCitationRecord(
            id=citation_model.id,
            message_id=citation_model.message_id,
            citation_index=citation_model.citation_index,
            chunk_id=citation_model.chunk_id,
            file_id=citation_model.file_id,
            source_filename=citation_model.source_filename,
            storage_key=citation_model.storage_key,
            chunk_index=citation_model.chunk_index,
            char_count=citation_model.char_count,
            content=citation_model.content,
            snippet=citation_model.snippet,
            merged_terms=list(citation_model.merged_terms_json or []),
            score=citation_model.score,
            heading_path=list(citation_model.heading_path_json or []),
            section_title=citation_model.section_title,
            content_type=citation_model.content_type,
            source_anchor=citation_model.source_anchor,
            created_at=citation_model.created_at,
        )
