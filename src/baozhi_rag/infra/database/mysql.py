"""MySQL 数据库管理。"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from baozhi_rag.core.config import Settings
from baozhi_rag.infra.database.models import Base


class DatabaseManager:
    """封装 SQLAlchemy Engine 与基础探活能力。"""

    def __init__(self, database_url: str) -> None:
        """初始化数据库引擎与会话工厂。"""
        self._engine = create_engine(
            database_url,
            pool_pre_ping=True,
            future=True,
        )
        self._session_factory = sessionmaker(
            bind=self._engine,
            class_=Session,
            expire_on_commit=False,
            future=True,
        )

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """返回可复用的会话工厂。"""
        return self._session_factory

    def ensure_ready(self) -> None:
        """校验数据库连接可用。"""
        try:
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError:
            raise

    def ensure_schema(self) -> None:
        """确保当前应用依赖的表结构存在。"""
        Base.metadata.create_all(self._engine, checkfirst=True)

    @classmethod
    def from_settings(cls, settings: Settings) -> DatabaseManager:
        """基于应用配置返回缓存后的数据库管理器。"""
        return _build_database_manager(settings.mysql_url)


@lru_cache(maxsize=4)
def _build_database_manager(database_url: str) -> DatabaseManager:
    """按连接串缓存数据库管理器，避免重复创建连接池。"""
    return DatabaseManager(database_url)
