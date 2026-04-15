"""基于 MongoDB 的聊天消息仓储实现。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pymongo import DESCENDING, ReturnDocument
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from baozhi_rag.domain.chat_errors import ChatMessageConflictError, ChatSessionNotFoundError
from baozhi_rag.domain.chat_message import (
    ChatMessageCitationRecord,
    ChatMessageRecord,
    ChatMessageRole,
    ChatMessageStatus,
)
from baozhi_rag.domain.chat_session import ChatSessionStatus
from baozhi_rag.infra.database.mongodb import MongoChatStorageManager


class MongoChatMessageRepository:
    """使用 MongoDB 持久化聊天消息与引用。"""

    def __init__(
        self,
        *,
        session_collection: Collection[dict[str, Any]],
        message_collection: Collection[dict[str, Any]],
    ) -> None:
        """初始化消息仓储。

        参数:
            session_collection: 聊天会话集合，用于维护序号与元数据。
            message_collection: 聊天消息集合。
        """
        self._session_collection = session_collection
        self._message_collection = message_collection

    @classmethod
    def from_manager(cls, manager: MongoChatStorageManager) -> MongoChatMessageRepository:
        """基于聊天记忆 MongoDB 管理器创建仓储。"""
        return cls(
            session_collection=manager.session_collection,
            message_collection=manager.message_collection,
        )

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
        """向会话末尾追加一条消息。"""
        now = datetime.now(UTC)
        sequence_no = self._reserve_sequence_no(session_id)
        document = {
            "_id": uuid4().hex,
            "session_id": session_id,
            "sequence_no": sequence_no,
            "role": role.value,
            "status": status.value,
            "plain_text": plain_text,
            "content_blocks": [dict(item) for item in content_blocks or []],
            "request_id": request_id,
            "model_name": model_name,
            "original_query": original_query,
            "retrieval_query": retrieval_query,
            "rewrite_applied": rewrite_applied,
            "retrieval_size": retrieval_size,
            "temperature": temperature,
            "finish_reason": finish_reason,
            "latency_ms": latency_ms,
            "usage": dict(usage) if usage is not None else None,
            "error_code": error_code,
            "error_message": error_message,
            "created_at": now,
            "updated_at": now,
            "completed_at": now if status is not ChatMessageStatus.STREAMING else None,
            "citations": [],
        }

        try:
            self._message_collection.insert_one(document)
        except DuplicateKeyError as exc:
            raise ChatMessageConflictError() from exc

        self._touch_session_after_append(session_id=session_id, role=role, now=now)
        return self._to_domain(document)

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
        """更新消息主体内容或执行状态。"""
        now = datetime.now(UTC)
        update_fields: dict[str, Any] = {
            "updated_at": now,
            "temperature": temperature,
        }
        if status is not None:
            update_fields["status"] = status.value
        if plain_text is not None:
            update_fields["plain_text"] = plain_text
        if content_blocks is not None:
            update_fields["content_blocks"] = [dict(item) for item in content_blocks]
        if model_name is not None:
            update_fields["model_name"] = model_name
        if original_query is not None:
            update_fields["original_query"] = original_query
        if retrieval_query is not None:
            update_fields["retrieval_query"] = retrieval_query
        if rewrite_applied is not None:
            update_fields["rewrite_applied"] = rewrite_applied
        if retrieval_size is not None:
            update_fields["retrieval_size"] = retrieval_size
        if finish_reason is not None:
            update_fields["finish_reason"] = finish_reason
        if latency_ms is not None:
            update_fields["latency_ms"] = latency_ms
        if usage is not None:
            update_fields["usage"] = dict(usage)
        if error_code is not None:
            update_fields["error_code"] = error_code
        if error_message is not None:
            update_fields["error_message"] = error_message
        if completed:
            update_fields["completed_at"] = now

        document = self._message_collection.find_one_and_update(
            {"_id": message_id},
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER,
        )
        return self._to_domain(document) if document is not None else None

    def replace_citations(
        self,
        message_id: str,
        citations: list[ChatMessageCitationRecord],
    ) -> list[ChatMessageCitationRecord]:
        """替换消息引用列表。"""
        document = self._message_collection.find_one({"_id": message_id})
        if document is None:
            return []

        now = datetime.now(UTC)
        citation_records = [
            ChatMessageCitationRecord(
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
                image_assets=[dict(item) for item in citation.image_assets],
            )
            for citation in citations
        ]
        self._message_collection.update_one(
            {"_id": message_id},
            {
                "$set": {
                    "citations": [self._citation_to_document(item) for item in citation_records],
                    "updated_at": now,
                }
            },
        )
        return citation_records

    def list_messages(
        self,
        *,
        session_id: str,
        before_sequence_no: int | None,
        limit: int,
    ) -> list[ChatMessageRecord]:
        """按游标查询消息历史。"""
        query: dict[str, Any] = {"session_id": session_id}
        if before_sequence_no is not None:
            query["sequence_no"] = {"$lt": before_sequence_no}

        documents = list(
            self._message_collection.find(query).sort([("sequence_no", DESCENDING)]).limit(limit)
        )
        documents.reverse()
        return [self._to_domain(item) for item in documents]

    def get_recent_messages(
        self,
        *,
        session_id: str,
        limit: int,
    ) -> list[ChatMessageRecord]:
        """获取最近一段消息窗口。"""
        return self.list_messages(session_id=session_id, before_sequence_no=None, limit=limit)

    def _reserve_sequence_no(self, session_id: str) -> int:
        """为会话原子分配下一条消息序号。"""
        document = self._session_collection.find_one_and_update(
            {
                "_id": session_id,
                "status": {"$ne": ChatSessionStatus.DELETED.value},
            },
            {"$inc": {"next_sequence_no": 1}},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise ChatSessionNotFoundError()
        return int(document.get("next_sequence_no", 0))

    def _touch_session_after_append(
        self,
        *,
        session_id: str,
        role: ChatMessageRole,
        now: datetime,
    ) -> None:
        """在消息落库后回写会话元数据。"""
        set_fields: dict[str, Any] = {
            "last_message_at": now,
            "updated_at": now,
        }
        if role is ChatMessageRole.USER:
            set_fields["last_user_message_at"] = now
        elif role is ChatMessageRole.ASSISTANT:
            set_fields["last_assistant_message_at"] = now

        self._session_collection.update_one(
            {"_id": session_id},
            {
                "$inc": {"message_count": 1},
                "$set": set_fields,
            },
        )

    def _to_domain(self, document: dict[str, Any]) -> ChatMessageRecord:
        """把 MongoDB 文档转换为领域消息对象。"""
        return ChatMessageRecord(
            id=str(document["_id"]),
            session_id=str(document["session_id"]),
            sequence_no=int(document["sequence_no"]),
            role=ChatMessageRole(str(document["role"])),
            status=ChatMessageStatus(str(document["status"])),
            plain_text=str(document.get("plain_text", "")),
            content_blocks=[dict(item) for item in document.get("content_blocks", [])],
            request_id=document.get("request_id"),
            model_name=document.get("model_name"),
            original_query=document.get("original_query"),
            retrieval_query=document.get("retrieval_query"),
            rewrite_applied=bool(document.get("rewrite_applied", False)),
            retrieval_size=document.get("retrieval_size"),
            temperature=document.get("temperature"),
            finish_reason=document.get("finish_reason"),
            latency_ms=document.get("latency_ms"),
            usage=dict(document["usage"]) if isinstance(document.get("usage"), dict) else None,
            error_code=document.get("error_code"),
            error_message=document.get("error_message"),
            created_at=document["created_at"],
            updated_at=document["updated_at"],
            completed_at=document.get("completed_at"),
            citations=[
                self._citation_to_domain(item)
                for item in document.get("citations", [])
                if isinstance(item, dict)
            ],
        )

    def _citation_to_document(self, citation: ChatMessageCitationRecord) -> dict[str, Any]:
        """把领域引用对象转换为 MongoDB 子文档。"""
        return {
            "id": citation.id,
            "message_id": citation.message_id,
            "citation_index": citation.citation_index,
            "chunk_id": citation.chunk_id,
            "file_id": citation.file_id,
            "source_filename": citation.source_filename,
            "storage_key": citation.storage_key,
            "chunk_index": citation.chunk_index,
            "char_count": citation.char_count,
            "content": citation.content,
            "snippet": citation.snippet,
            "merged_terms": list(citation.merged_terms),
            "score": citation.score,
            "heading_path": list(citation.heading_path),
            "section_title": citation.section_title,
            "content_type": citation.content_type,
            "source_anchor": citation.source_anchor,
            "created_at": citation.created_at,
            "image_assets": [dict(item) for item in citation.image_assets],
        }

    def _citation_to_domain(self, document: dict[str, Any]) -> ChatMessageCitationRecord:
        """把 MongoDB 子文档转换为领域引用对象。"""
        return ChatMessageCitationRecord(
            id=str(document["id"]),
            message_id=str(document["message_id"]),
            citation_index=int(document["citation_index"]),
            chunk_id=str(document["chunk_id"]),
            file_id=str(document["file_id"]),
            source_filename=str(document["source_filename"]),
            storage_key=str(document["storage_key"]),
            chunk_index=int(document["chunk_index"]),
            char_count=int(document["char_count"]),
            content=str(document["content"]),
            snippet=str(document["snippet"]),
            merged_terms=[str(item) for item in document.get("merged_terms", [])],
            score=document.get("score"),
            heading_path=[str(item) for item in document.get("heading_path", [])],
            section_title=document.get("section_title"),
            content_type=str(document["content_type"]),
            source_anchor=document.get("source_anchor"),
            created_at=document["created_at"],
            image_assets=[dict(item) for item in document.get("image_assets", [])],
        )
