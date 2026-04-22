"""基于 MongoDB 的聊天记录批量清理仓储。"""

from __future__ import annotations

from typing import Any

from pymongo.collection import Collection

from baozhi_rag.infra.database.mongodb import MongoChatStorageManager
from baozhi_rag.services.knowledge_file_delete import ChatRecordPurgeResult


class MongoChatMemoryCleanupRepository:
    """按用户或全站范围删除聊天会话、消息与记忆快照。"""

    def __init__(
        self,
        *,
        session_collection: Collection[dict[str, Any]],
        message_collection: Collection[dict[str, Any]],
        snapshot_collection: Collection[dict[str, Any]],
    ) -> None:
        """初始化聊天记录批量清理仓储。

        参数:
            session_collection: 聊天会话集合。
            message_collection: 聊天消息集合。
            snapshot_collection: 聊天记忆快照集合。
        """
        self._session_collection = session_collection
        self._message_collection = message_collection
        self._snapshot_collection = snapshot_collection

    @classmethod
    def from_manager(cls, manager: MongoChatStorageManager) -> MongoChatMemoryCleanupRepository:
        """基于聊天记忆 MongoDB 管理器创建仓储。"""
        return cls(
            session_collection=manager.session_collection,
            message_collection=manager.message_collection,
            snapshot_collection=manager.snapshot_collection,
        )

    def delete_chat_records_by_user(self, owner_user_id: str) -> ChatRecordPurgeResult:
        """删除指定用户的全部聊天记录。

        参数:
            owner_user_id: 需要清理聊天记录的用户 ID。

        返回:
            本次删除的会话、消息和快照数量摘要。
        """
        session_ids = self._list_session_ids(owner_user_id=owner_user_id)
        deleted_session_count = self._session_collection.delete_many(
            {"owner_user_id": owner_user_id}
        ).deleted_count
        if not session_ids:
            return ChatRecordPurgeResult(
                deleted_session_count=int(deleted_session_count),
                deleted_message_count=0,
                deleted_snapshot_count=0,
            )

        deleted_message_count = self._message_collection.delete_many(
            {"session_id": {"$in": session_ids}}
        ).deleted_count
        deleted_snapshot_count = self._snapshot_collection.delete_many(
            {"session_id": {"$in": session_ids}}
        ).deleted_count
        return ChatRecordPurgeResult(
            deleted_session_count=int(deleted_session_count),
            deleted_message_count=int(deleted_message_count),
            deleted_snapshot_count=int(deleted_snapshot_count),
        )

    def delete_all_chat_records(self) -> ChatRecordPurgeResult:
        """删除全站全部聊天记录。"""
        deleted_session_count = self._session_collection.delete_many({}).deleted_count
        deleted_message_count = self._message_collection.delete_many({}).deleted_count
        deleted_snapshot_count = self._snapshot_collection.delete_many({}).deleted_count
        return ChatRecordPurgeResult(
            deleted_session_count=int(deleted_session_count),
            deleted_message_count=int(deleted_message_count),
            deleted_snapshot_count=int(deleted_snapshot_count),
        )

    def _list_session_ids(self, *, owner_user_id: str) -> list[str]:
        """查询指定用户当前拥有的全部会话 ID。"""
        cursor = self._session_collection.find(
            {"owner_user_id": owner_user_id},
            projection={"_id": 1},
        )
        return [str(item["_id"]) for item in cursor if "_id" in item]
