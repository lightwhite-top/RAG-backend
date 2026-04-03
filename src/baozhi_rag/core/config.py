"""应用配置管理。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import quote_plus

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from baozhi_rag.core.request_context import REQUEST_ID_HEADER_NAME


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
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="允许跨域访问的来源列表，多个来源通过逗号分隔",
        validation_alias=AliasChoices("CORS_ALLOW_ORIGINS"),
    )
    cors_allow_origin_regex: str | None = Field(
        default=None,
        description="允许跨域访问的来源正则表达式",
        validation_alias=AliasChoices("CORS_ALLOW_ORIGIN_REGEX"),
    )
    cors_allow_credentials: bool = Field(
        default=False,
        description="是否允许跨域请求携带凭证",
        validation_alias=AliasChoices("CORS_ALLOW_CREDENTIALS"),
    )
    cors_allow_methods: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        description="允许跨域访问的方法列表，多个方法通过逗号分隔",
        validation_alias=AliasChoices("CORS_ALLOW_METHODS"),
    )
    cors_allow_headers: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"],
        description="允许跨域访问时携带的请求头列表，多个请求头通过逗号分隔",
        validation_alias=AliasChoices("CORS_ALLOW_HEADERS"),
    )
    cors_expose_headers: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [REQUEST_ID_HEADER_NAME],
        description="允许前端读取的响应头列表，多个响应头通过逗号分隔",
        validation_alias=AliasChoices("CORS_EXPOSE_HEADERS"),
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
    milvus_uri: str = Field(
        default="http://127.0.0.1:19530",
        description="Milvus 连接地址",
        validation_alias=AliasChoices("MILVUS_URI"),
    )
    milvus_token: str | None = Field(
        default=None,
        description="Milvus 认证令牌，格式通常为 user:password",
        validation_alias=AliasChoices("MILVUS_TOKEN"),
    )
    milvus_db_name: str = Field(
        default="default",
        description="Milvus 数据库名称",
        validation_alias=AliasChoices("MILVUS_DB_NAME"),
    )
    milvus_collection_name: str = Field(
        default="document_chunk_vectors",
        description="Milvus 向量集合名称",
        validation_alias=AliasChoices("MILVUS_COLLECTION_NAME"),
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
    chat_system_prompt: str = Field(
        default=(
            "你是金融保险问答助手。"
            "如果当前轮提供了知识库证据，优先只基于证据回答，并在相关句子后追加 [1][2] 这类证据编号。"
            "如果当前轮没有提供知识库证据，你仍可以回答问候、身份说明、能力介绍和一般性概念解释，"
            "但不得把未验证内容表述为具体条款、保障责任、免责结论、理赔结论或确定性承诺。"
            "遇到具体产品、保单、承保、理赔、免责、金额计算等高风险问题且缺少证据时，"
            "必须明确说明“当前没有检索到可支撑结论的知识库材料”，"
            "只能提供一般性说明，并建议补充材料或转人工核实。"
            "涉及理赔、承保、免责、保单解释时，要明确说明以正式合同条款、系统记录和人工审核结果为准。"
        ),
        description="聊天接口默认使用的系统提示词",
        validation_alias=AliasChoices("CHAT_SYSTEM_PROMPT"),
        min_length=1,
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
    mysql_host: str = Field(
        default="127.0.0.1",
        description="MySQL 主机地址",
        validation_alias=AliasChoices("MYSQL_HOST"),
    )
    mysql_port: int = Field(
        default=3306,
        description="MySQL 端口",
        validation_alias=AliasChoices("MYSQL_PORT"),
    )
    mysql_database: str = Field(
        default="baozhi_rag",
        description="MySQL 数据库名称",
        validation_alias=AliasChoices("MYSQL_DATABASE"),
    )
    mysql_username: str = Field(
        default="root",
        description="MySQL 用户名",
        validation_alias=AliasChoices("MYSQL_USERNAME"),
    )
    mysql_password: str = Field(
        default="",
        description="MySQL 密码",
        validation_alias=AliasChoices("MYSQL_PASSWORD"),
    )
    jwt_secret_key: str = Field(
        default="change-me",
        description="JWT 签名密钥",
        validation_alias=AliasChoices("JWT_SECRET_KEY"),
        min_length=1,
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT 签名算法",
        validation_alias=AliasChoices("JWT_ALGORITHM"),
        min_length=1,
    )
    jwt_access_token_expire_days: int = Field(
        default=7,
        description="JWT 访问令牌有效期（天）",
        validation_alias=AliasChoices("JWT_ACCESS_TOKEN_EXPIRE_DAYS"),
        ge=1,
    )
    smtp_host: str | None = Field(
        default=None,
        description="SMTP 服务器地址",
        validation_alias=AliasChoices("SMTP_HOST"),
    )
    smtp_port: int = Field(
        default=587,
        description="SMTP 服务器端口",
        validation_alias=AliasChoices("SMTP_PORT"),
        ge=1,
        le=65535,
    )
    smtp_username: str | None = Field(
        default=None,
        description="SMTP 登录用户名",
        validation_alias=AliasChoices("SMTP_USERNAME"),
    )
    smtp_password: str | None = Field(
        default=None,
        description="SMTP 登录密码",
        validation_alias=AliasChoices("SMTP_PASSWORD"),
    )
    smtp_use_tls: bool = Field(
        default=True,
        description="是否对 SMTP 明文连接启用 STARTTLS",
        validation_alias=AliasChoices("SMTP_USE_TLS"),
    )
    smtp_use_ssl: bool = Field(
        default=False,
        description="是否直接使用 SMTPS 连接",
        validation_alias=AliasChoices("SMTP_USE_SSL"),
    )
    smtp_from_email: str | None = Field(
        default=None,
        description="注册验证码邮件发件地址",
        validation_alias=AliasChoices("SMTP_FROM_EMAIL"),
    )
    smtp_from_name: str = Field(
        default="Baozhi RAG Service",
        description="注册验证码邮件发件人名称",
        validation_alias=AliasChoices("SMTP_FROM_NAME"),
    )
    smtp_timeout_seconds: float = Field(
        default=10.0,
        description="SMTP 连接与发送超时时间（秒）",
        validation_alias=AliasChoices("SMTP_TIMEOUT_SECONDS"),
        gt=0,
    )
    registration_code_secret: str = Field(
        default="change-me-registration-code-secret",
        description="注册验证码摘要签名密钥",
        validation_alias=AliasChoices("REGISTRATION_CODE_SECRET"),
        min_length=1,
    )
    registration_code_length: int = Field(
        default=6,
        description="注册验证码长度",
        validation_alias=AliasChoices("REGISTRATION_CODE_LENGTH"),
        ge=4,
        le=8,
    )
    registration_code_expire_minutes: int = Field(
        default=10,
        description="注册验证码有效期（分钟）",
        validation_alias=AliasChoices("REGISTRATION_CODE_EXPIRE_MINUTES"),
        ge=1,
        le=60,
    )
    registration_code_resend_interval_seconds: int = Field(
        default=60,
        description="注册验证码重发冷却时间（秒）",
        validation_alias=AliasChoices("REGISTRATION_CODE_RESEND_INTERVAL_SECONDS"),
        ge=0,
        le=3600,
    )
    registration_code_max_attempts: int = Field(
        default=5,
        description="单个验证码允许的最大错误尝试次数",
        validation_alias=AliasChoices("REGISTRATION_CODE_MAX_ATTEMPTS"),
        ge=1,
        le=10,
    )
    oss_region: str = Field(
        default="cn-hangzhou",
        description="阿里云 OSS 所属地域",
        validation_alias=AliasChoices("OSS_REGION"),
    )
    oss_endpoint: str = Field(
        default="https://oss-cn-hangzhou.aliyuncs.com",
        description="阿里云 OSS 访问域名",
        validation_alias=AliasChoices("OSS_ENDPOINT"),
    )
    oss_bucket_name: str = Field(
        default="",
        description="阿里云 OSS Bucket 名称",
        validation_alias=AliasChoices("OSS_BUCKET_NAME"),
    )
    oss_access_key_id: str = Field(
        default="",
        description="阿里云 OSS AccessKey ID",
        validation_alias=AliasChoices("OSS_ACCESS_KEY_ID"),
    )
    oss_access_key_secret: str = Field(
        default="",
        description="阿里云 OSS AccessKey Secret",
        validation_alias=AliasChoices("OSS_ACCESS_KEY_SECRET"),
    )
    oss_object_prefix: str = Field(
        default="knowledge-files",
        description="阿里云 OSS 对象统一前缀",
        validation_alias=AliasChoices("OSS_OBJECT_PREFIX"),
    )

    @field_validator(
        "cors_allow_origins",
        "cors_allow_methods",
        "cors_allow_headers",
        "cors_expose_headers",
        mode="before",
    )
    @classmethod
    def parse_csv_list(cls, value: object) -> object:
        """把 `.env` 中逗号分隔的配置解析为字符串列表。"""
        if not isinstance(value, str):
            return value

        if not value.strip():
            return []

        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("cors_allow_origin_regex", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        """把空字符串正则配置归一化为 `None`。"""
        if not isinstance(value, str):
            return value

        normalized_value = value.strip()
        return normalized_value or None

    @model_validator(mode="after")
    def validate_email_delivery_settings(self) -> Settings:
        """校验邮件发送配置中互斥的连接模式。"""
        if self.smtp_use_tls and self.smtp_use_ssl:
            raise ValueError("SMTP_USE_TLS 与 SMTP_USE_SSL 不能同时为 true")
        return self

    @property
    def mysql_url(self) -> str:
        """基于配置构造 SQLAlchemy 使用的 MySQL 连接串。"""
        quoted_username = quote_plus(self.mysql_username)
        quoted_password = quote_plus(self.mysql_password)
        quoted_database = quote_plus(self.mysql_database)
        return (
            "mysql+pymysql://"
            f"{quoted_username}:{quoted_password}@{self.mysql_host}:{self.mysql_port}/"
            f"{quoted_database}?charset=utf8mb4"
        )

    @property
    def normalized_oss_object_prefix(self) -> str:
        """返回清理首尾斜杠后的 OSS 对象前缀。"""
        return self.oss_object_prefix.strip().strip("/")


@lru_cache
def get_settings() -> Settings:
    """缓存并返回应用配置对象。

    返回:
        从环境变量和 `.env` 文件解析出的 Settings 实例。
    """
    return Settings()
