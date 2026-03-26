"""应用配置管理。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """统一管理服务启动配置。"""

    model_config = SettingsConfigDict(
        env_prefix="BAOZHI_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Baozhi RAG Service", description="服务名称")
    app_env: str = Field(default="local", description="运行环境")
    debug: bool = Field(default=False, description="是否开启调试模式")
    version: str = Field(default="0.1.0", description="服务版本")
    log_level: str = Field(default="INFO", description="日志级别")
    upload_root_dir: Path = Field(
        default=Path("data/uploads"),
        description="上传文件本地存储根目录",
    )


@lru_cache
def get_settings() -> Settings:
    """缓存配置对象，避免重复读取环境变量。"""
    return Settings()
