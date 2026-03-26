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
    doc_chunk_size: int = Field(default=800, description="Word 文档切块大小")
    doc_chunk_overlap: int = Field(default=150, description="Word 文档切块重叠长度")
    doc_convert_temp_dir: Path = Field(
        default=Path("data/tmp/converted"),
        description="旧版 Word 转换临时目录",
    )


@lru_cache
def get_settings() -> Settings:
    """缓存并返回应用配置对象。

    返回:
        从环境变量和 `.env` 文件解析出的 Settings 实例。
    """
    return Settings()
