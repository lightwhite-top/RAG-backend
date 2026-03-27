"""应用配置管理。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """统一管理服务启动配置。"""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = Field(
        default="Baozhi RAG Service",
        description="服务名称",
        validation_alias=AliasChoices("APP_NAME"),
    )
    app_env: str = Field(
        default="local",
        description="运行环境",
        validation_alias=AliasChoices("APP_ENV"),
    )
    debug: bool = Field(
        default=False,
        description="是否开启调试模式",
        validation_alias=AliasChoices("APP_DEBUG"),
    )
    version: str = Field(
        default="0.1.0",
        description="服务版本",
        validation_alias=AliasChoices("APP_VERSION"),
    )
    log_level: str = Field(
        default="INFO",
        description="日志级别",
        validation_alias=AliasChoices("APP_LOG_LEVEL"),
    )
    upload_root_dir: Path = Field(
        default=Path("data/uploads"),
        description="上传文件本地存储根目录",
        validation_alias=AliasChoices("UPLOAD_ROOT_DIR"),
    )
    doc_chunk_size: int = Field(
        default=800,
        description="Word 文档切块大小",
        validation_alias=AliasChoices("DOC_CHUNK_SIZE"),
    )
    doc_chunk_overlap: int = Field(
        default=150,
        description="Word 文档切块重叠长度",
        validation_alias=AliasChoices("DOC_CHUNK_OVERLAP"),
    )
    doc_convert_temp_dir: Path = Field(
        default=Path("data/tmp/converted"),
        description="旧版 Word 转换临时目录",
        validation_alias=AliasChoices("DOC_CONVERT_TEMP_DIR"),
    )
    domain_dictionary_path: Path | None = Field(
        default=None,
        description="金融保险领域词典文件路径，按行存储词项",
        validation_alias=AliasChoices("DOMAIN_DICTIONARY_PATH"),
    )
    es_url: str = Field(
        default="http://127.0.0.1:9200",
        description="Elasticsearch 地址",
        validation_alias=AliasChoices("ES_URL"),
    )
    es_index_name: str = Field(
        default="document_chunks",
        description="chunk 索引名称",
        validation_alias=AliasChoices("ES_INDEX_NAME"),
    )
    es_username: str | None = Field(
        default=None,
        description="Elasticsearch 用户名",
        validation_alias=AliasChoices("ES_USERNAME"),
    )
    es_password: str | None = Field(
        default=None,
        description="Elasticsearch 密码",
        validation_alias=AliasChoices("ES_PASSWORD"),
    )
    es_api_key: str | None = Field(
        default=None,
        description="Elasticsearch API Key",
        validation_alias=AliasChoices("ES_API_KEY"),
    )
    es_verify_certs: bool = Field(
        default=True,
        description="是否校验 Elasticsearch 证书",
        validation_alias=AliasChoices("ES_VERIFY_CERTS"),
    )
    bailian_api_key: str | None = Field(
        default=None,
        description="阿里云百炼 DashScope API Key",
        validation_alias=AliasChoices("DASHSCOPE_API_KEY", "BAILIAN_API_KEY"),
    )
    bailian_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="阿里云百炼 OpenAI 兼容接口地址",
        validation_alias=AliasChoices("DASHSCOPE_BASE_URL", "BAILIAN_BASE_URL"),
    )
    bailian_timeout_seconds: float = Field(
        default=30.0,
        description="阿里云百炼模型调用超时时间（秒）",
        validation_alias=AliasChoices("BAILIAN_TIMEOUT_SECONDS"),
    )
    bailian_chat_model: str | None = Field(
        default=None,
        description="预留的阿里云百炼聊天模型名称",
        validation_alias=AliasChoices("BAILIAN_CHAT_MODEL"),
    )
    chunk_embedding_enabled: bool = Field(
        default=False,
        description="是否启用 chunk 向量化与语义检索",
        validation_alias=AliasChoices("CHUNK_EMBEDDING_ENABLED"),
    )
    chunk_embedding_model: str = Field(
        default="text-embedding-v4",
        description="chunk 向量化模型名称",
        validation_alias=AliasChoices("CHUNK_EMBEDDING_MODEL"),
    )
    chunk_embedding_dimensions: int = Field(
        default=1024,
        description="chunk 向量维度",
        validation_alias=AliasChoices("CHUNK_EMBEDDING_DIMENSIONS"),
    )
    chunk_embedding_batch_size: int = Field(
        default=10,
        description="单次批量向量化请求的最大文本条数",
        validation_alias=AliasChoices("CHUNK_EMBEDDING_BATCH_SIZE"),
    )
    search_default_size: int = Field(
        default=10,
        description="chunk 检索默认返回条数",
        validation_alias=AliasChoices("SEARCH_DEFAULT_SIZE"),
    )


@lru_cache
def get_settings() -> Settings:
    """缓存并返回应用配置对象。

    返回:
        从环境变量和 `.env` 文件解析出的 Settings 实例。
    """
    return Settings()
