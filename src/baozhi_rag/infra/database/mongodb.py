"""聊天记忆 MongoDB 存储管理。"""

from __future__ import annotations

import logging
from datetime import UTC
from functools import lru_cache
from typing import Any

from bson.codec_options import CodecOptions
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import OperationFailure

from baozhi_rag.core.config import Settings

LOGGER = logging.getLogger(__name__)


class MongoChatStorageManager:
    """封装聊天记忆所需的 MongoDB 客户端、集合与索引初始化。"""

    def __init__(
        self,
        *,
        uri: str,
        database_name: str,
        session_collection_name: str,
        message_collection_name: str,
        snapshot_collection_name: str,
    ) -> None:
        """初始化聊天记忆 MongoDB 客户端。

        参数:
            uri: MongoDB 连接串。
            database_name: 聊天记忆数据库名。
            session_collection_name: 聊天会话集合名。
            message_collection_name: 聊天消息集合名。
            snapshot_collection_name: 记忆快照集合名。
        """
        self._client: MongoClient[dict[str, Any]] = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
        )
        codec_options: CodecOptions = CodecOptions(tz_aware=True, tzinfo=UTC)
        self._database: Database[dict[str, Any]] = self._client.get_database(
            database_name,
            codec_options=codec_options,
        )
        self._session_collection = self._database[session_collection_name]
        self._message_collection = self._database[message_collection_name]
        self._snapshot_collection = self._database[snapshot_collection_name]

    @property
    def session_collection(self) -> Collection[dict[str, Any]]:
        """返回聊天会话集合。"""
        return self._session_collection

    @property
    def message_collection(self) -> Collection[dict[str, Any]]:
        """返回聊天消息集合。"""
        return self._message_collection

    @property
    def snapshot_collection(self) -> Collection[dict[str, Any]]:
        """返回聊天记忆快照集合。"""
        return self._snapshot_collection

    def ensure_ready(self) -> None:
        """校验 MongoDB 连接可用。"""
        self._client.admin.command("ping")

    def ensure_indexes(self) -> None:
        """确保聊天记忆所需的关键索引存在。

        线上业务账号可能只有读写权限，没有 `createIndexes` 权限。此时缺少索引
        主要影响查询性能，不应阻断整个服务启动，因此仅对未授权的建索引错误做
        告警降级，其余索引异常仍按原样抛出。
        """
        try:
            self._ensure_session_indexes()
            self._ensure_message_indexes()
            self._ensure_snapshot_indexes()
        except OperationFailure as exc:
            # 缺少 createIndexes 权限时允许服务继续启动，避免将可用性问题放大成启动故障。
            if not self._is_unauthorized_index_error(exc):
                raise
            LOGGER.warning(
                (
                    "chat_memory_mongodb_index_init_skipped reason=unauthorized detail=%s "
                    "followup=verify_required_indexes_precreated"
                ),
                self._extract_operation_failure_message(exc),
            )

    def _ensure_session_indexes(self) -> None:
        """创建会话集合所需索引。"""
        self._session_collection.create_index(
            [("owner_user_id", ASCENDING), ("updated_at", DESCENDING), ("_id", DESCENDING)],
            name="ix_chat_sessions_owner_updated_at",
        )
        self._session_collection.create_index(
            [
                ("owner_user_id", ASCENDING),
                ("status", ASCENDING),
                ("updated_at", DESCENDING),
                ("_id", DESCENDING),
            ],
            name="ix_chat_sessions_owner_status_updated_at",
        )

    def _ensure_message_indexes(self) -> None:
        """创建消息集合所需索引。"""
        self._message_collection.create_index(
            [("session_id", ASCENDING), ("sequence_no", ASCENDING)],
            name="uq_chat_messages_session_sequence",
            unique=True,
        )
        self._message_collection.create_index(
            [("session_id", ASCENDING), ("created_at", DESCENDING)],
            name="ix_chat_messages_session_created_at",
        )
        self._message_collection.create_index(
            [("request_id", ASCENDING)],
            name="ix_chat_messages_request_id",
        )

    def _ensure_snapshot_indexes(self) -> None:
        """创建快照集合所需索引。"""
        self._snapshot_collection.create_index(
            [("session_id", ASCENDING), ("version", ASCENDING)],
            name="uq_chat_session_memory_snapshots_session_version",
            unique=True,
        )
        self._snapshot_collection.create_index(
            [("session_id", ASCENDING), ("covered_until_sequence_no", ASCENDING)],
            name="ix_chat_session_memory_snapshots_session_covered",
        )

    @staticmethod
    def _is_unauthorized_index_error(exc: OperationFailure) -> bool:
        """判断索引初始化失败是否由权限不足导致。"""
        if exc.code == 13:
            return True
        if isinstance(exc.details, dict):
            return exc.details.get("codeName") == "Unauthorized"
        return False

    @staticmethod
    def _extract_operation_failure_message(exc: OperationFailure) -> str:
        """提取 MongoDB 命令失败的可读错误信息。"""
        if isinstance(exc.details, dict):
            error_message = exc.details.get("errmsg")
            if isinstance(error_message, str) and error_message:
                return error_message
        return str(exc)

    def close(self) -> None:
        """关闭 MongoDB 客户端。"""
        self._client.close()

    @classmethod
    def from_settings(cls, settings: Settings) -> MongoChatStorageManager:
        """基于应用配置返回缓存后的 MongoDB 管理器。"""
        return _build_chat_storage_manager(
            uri=settings.chat_memory_mongodb_uri,
            database_name=settings.chat_memory_mongodb_database,
            session_collection_name=settings.chat_memory_mongodb_session_collection_name,
            message_collection_name=settings.chat_memory_mongodb_message_collection_name,
            snapshot_collection_name=settings.chat_memory_mongodb_snapshot_collection_name,
        )


@lru_cache(maxsize=4)
def _build_chat_storage_manager(
    *,
    uri: str,
    database_name: str,
    session_collection_name: str,
    message_collection_name: str,
    snapshot_collection_name: str,
) -> MongoChatStorageManager:
    """按连接参数缓存聊天记忆 MongoDB 管理器。"""
    return MongoChatStorageManager(
        uri=uri,
        database_name=database_name,
        session_collection_name=session_collection_name,
        message_collection_name=message_collection_name,
        snapshot_collection_name=snapshot_collection_name,
    )
