"""阿里云 OSS 文件存储适配。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from baozhi_rag.core.config import Settings
from baozhi_rag.domain.knowledge_file_errors import (
    ObjectStorageDependencyError,
    ObjectStorageError,
)

try:  # pragma: no cover - 依赖可选导入
    import alibabacloud_oss_v2 as imported_aliyun_oss  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover - 依赖缺失路径
    ALIYUN_OSS_MODULE: Any | None = None
    ALIYUN_OSS_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - 导入成功无需单独覆盖
    ALIYUN_OSS_MODULE = imported_aliyun_oss
    ALIYUN_OSS_IMPORT_ERROR = None


class AliyunOssFileStore:
    """负责上传、删除和探活阿里云 OSS 对象。"""

    def __init__(
        self,
        *,
        region: str,
        endpoint: str,
        bucket_name: str,
        access_key_id: str,
        access_key_secret: str,
    ) -> None:
        """初始化 OSS 存储适配器。"""
        self._region = region
        self._endpoint = endpoint
        self._bucket_name = bucket_name
        self._access_key_id = access_key_id
        self._access_key_secret = access_key_secret
        self._client: Any | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> AliyunOssFileStore:
        """基于应用配置构造 OSS 存储适配器。"""
        return cls(
            region=settings.oss_region,
            endpoint=settings.oss_endpoint,
            bucket_name=settings.oss_bucket_name,
            access_key_id=settings.oss_access_key_id,
            access_key_secret=settings.oss_access_key_secret,
        )

    def ensure_ready(self) -> None:
        """校验 OSS 客户端和目标 Bucket 可用。"""
        if not self._bucket_name.strip():
            raise ObjectStorageDependencyError("OSS_BUCKET_NAME 未配置")
        if not self._access_key_id.strip() or not self._access_key_secret.strip():
            raise ObjectStorageDependencyError("OSS_ACCESS_KEY_ID 或 OSS_ACCESS_KEY_SECRET 未配置")

        try:
            exists = self._get_client().is_bucket_exist(bucket=self._bucket_name)
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            raise ObjectStorageError("校验 OSS Bucket 可用性失败") from exc

        if not exists:
            raise ObjectStorageDependencyError(f"OSS Bucket 不存在: {self._bucket_name}")

    def upload_file(self, *, local_path: Path, storage_key: str) -> None:
        """把本地文件上传到 OSS。"""
        oss_module = self._get_oss_module()
        try:
            with local_path.open("rb") as file_obj:
                self._get_client().put_object(
                    cast(
                        Any,
                        oss_module.PutObjectRequest(
                            bucket=self._bucket_name,
                            key=storage_key,
                            body=file_obj,
                        ),
                    )
                )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            raise ObjectStorageError(f"上传 OSS 对象失败: {storage_key}") from exc

    def delete(self, storage_key: str) -> None:
        """删除指定 OSS 对象。"""
        oss_module = self._get_oss_module()
        try:
            self._get_client().delete_object(
                cast(
                    Any,
                    oss_module.DeleteObjectRequest(
                        bucket=self._bucket_name,
                        key=storage_key,
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            raise ObjectStorageError(f"删除 OSS 对象失败: {storage_key}") from exc

    def _get_client(self) -> Any:
        """延迟初始化 OSS 客户端。"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        """创建 OSS 客户端实例。"""
        oss_module = self._get_oss_module()
        credentials_provider = oss_module.credentials.StaticCredentialsProvider(
            access_key_id=self._access_key_id,
            access_key_secret=self._access_key_secret,
        )
        cfg = oss_module.config.load_default()
        cfg.region = self._region
        cfg.endpoint = self._endpoint
        cfg.credentials_provider = credentials_provider
        return oss_module.Client(cfg)

    def _get_oss_module(self) -> Any:
        """返回已加载的 OSS SDK 模块。"""
        if ALIYUN_OSS_MODULE is None:
            raise ObjectStorageDependencyError(
                "未安装 alibabacloud-oss-v2 依赖"
            ) from ALIYUN_OSS_IMPORT_ERROR
        return cast(Any, ALIYUN_OSS_MODULE)
