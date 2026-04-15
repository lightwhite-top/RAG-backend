"""基于 MongoDB 的聊天会话仓储实现。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pymongo import DESCENDING, ReturnDocument
from pymongo.collection import Collection

from baozhi_rag.domain.chat_session import ChatSession, ChatSessionListPage, ChatSessionStatus
from baozhi_rag.infra.database.mongodb import MongoChatStorageManager


class MongoChatSessionRepository:
    """使用 MongoDB 持久化聊天会话。"""

    def __init__(self, collection: Collection[dict[str, Any]]) -> None:
        """初始化聊天会话集合。

        参数:
            collection: 聊天会话 MongoDB 集合。
        """
        self._collection = collection

    @classmethod
    def from_manager(cls, manager: MongoChatStorageManager) -> MongoChatSessionRepository:
        """基于聊天记忆 MongoDB 管理器创建仓储。"""
        return cls(manager.session_collection)

    def create_session(self, session: ChatSession) -> ChatSession:
        """创建聊天会话，并初始化内部序号计数器。"""
        self._collection.insert_one(self._to_document(session))
        return session

    def get_session_by_id(self, session_id: str) -> ChatSession | None:
        """按会话 ID 查询会话。"""
        document = self._collection.find_one({"_id": session_id})
        return self._to_domain(document) if document is not None else None

    def list_sessions(
        self,
        *,
        owner_user_id: str,
        page: int,
        page_size: int,
        status: ChatSessionStatus | None,
    ) -> ChatSessionListPage:
        """分页查询会话列表。"""
        query: dict[str, Any] = {"owner_user_id": owner_user_id}
        if status is None:
            query["status"] = {"$ne": ChatSessionStatus.DELETED.value}
        else:
            query["status"] = status.value

        cursor = (
            self._collection.find(query)
            .sort([("updated_at", DESCENDING), ("_id", DESCENDING)])
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        items = [self._to_domain(item) for item in cursor]
        total = self._collection.count_documents(query)
        return ChatSessionListPage(items=items, total=total, page=page, page_size=page_size)

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        status: ChatSessionStatus | None = None,
    ) -> ChatSession | None:
        """更新会话标题或状态。"""
        now = datetime.now(UTC)
        update_fields: dict[str, Any] = {"updated_at": now}
        if title is not None:
            update_fields["title"] = title
        if status is not None:
            update_fields["status"] = status.value
            update_fields["deleted_at"] = now if status is ChatSessionStatus.DELETED else None
        document = self._collection.find_one_and_update(
            {"_id": session_id},
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER,
        )
        return self._to_domain(document) if document is not None else None

    def _to_document(self, session: ChatSession) -> dict[str, Any]:
        """把领域会话对象转换为 MongoDB 文档。"""
        return {
            "_id": session.id,
            "owner_user_id": session.owner_user_id,
            "title": session.title,
            "status": session.status.value,
            "message_count": session.message_count,
            "summary_version": session.summary_version,
            "last_message_at": session.last_message_at,
            "last_user_message_at": session.last_user_message_at,
            "last_assistant_message_at": session.last_assistant_message_at,
            "deleted_at": session.deleted_at,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            # 使用独立序号计数器避免每次追加消息都扫描整段历史。
            "next_sequence_no": session.message_count,
        }

    def _to_domain(self, document: dict[str, Any]) -> ChatSession:
        """把 MongoDB 文档转换为领域会话对象。"""
        return ChatSession(
            id=str(document["_id"]),
            owner_user_id=str(document["owner_user_id"]),
            title=str(document["title"]),
            status=ChatSessionStatus(str(document["status"])),
            message_count=int(document.get("message_count", 0)),
            summary_version=int(document.get("summary_version", 0)),
            last_message_at=document.get("last_message_at"),
            last_user_message_at=document.get("last_user_message_at"),
            last_assistant_message_at=document.get("last_assistant_message_at"),
            deleted_at=document.get("deleted_at"),
            created_at=document["created_at"],
            updated_at=document["updated_at"],
        )
