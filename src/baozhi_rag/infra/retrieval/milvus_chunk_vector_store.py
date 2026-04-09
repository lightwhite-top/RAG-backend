"""Milvus chunk 向量存储适配。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from fastapi import status

from baozhi_rag.core.exceptions import AppError

if TYPE_CHECKING:
    from baozhi_rag.core.config import Settings
    from baozhi_rag.services.document_chunking import DocumentChunk

MILVUS_CLIENT_CLASS: Any | None = None
MILVUS_DATA_TYPE: Any | None = None
MILVUS_IMPORT_ERROR: Exception | None = None
MILVUS_IMPORT_ATTEMPTED = False

LOGGER = logging.getLogger(__name__)


class MilvusStoreError(AppError):
    """Milvus 适配层异常。"""

    default_message = "Milvus 调用失败"
    default_error_code = "milvus_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class MilvusDependencyError(MilvusStoreError):
    """Milvus 客户端依赖缺失。"""

    default_message = "Milvus 客户端依赖缺失"
    default_error_code = "milvus_dependency_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class MilvusIndexError(MilvusStoreError):
    """Milvus 集合或写入异常。"""

    default_message = "Milvus 集合写入失败"
    default_error_code = "milvus_index_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class MilvusSearchError(MilvusStoreError):
    """Milvus 检索异常。"""

    default_message = "Milvus 检索失败"
    default_error_code = "milvus_search_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


@dataclass(frozen=True, slots=True)
class MilvusVectorSearchHit:
    """Milvus 向量命中结果。"""

    chunk_id: str
    score: float | None


class MilvusChunkVectorStore:
    """负责 chunk 向量集合创建、写入、删除与检索。"""

    _PRIMARY_FIELD_NAME = "chunk_id"
    _FILE_ID_FIELD_NAME = "file_id"
    _UPLOADER_USER_ID_FIELD_NAME = "uploader_user_id"
    _VISIBILITY_SCOPE_FIELD_NAME = "visibility_scope"
    _VECTOR_FIELD_NAME = "content_embedding"
    _VECTOR_INDEX_NAME = "content_embedding_idx"
    _VECTOR_INDEX_TYPE = "AUTOINDEX"
    _VECTOR_METRIC_TYPE = "COSINE"

    def __init__(
        self,
        *,
        uri: str,
        token: str | None,
        db_name: str,
        collection_name: str,
        embedding_dimensions: int,
    ) -> None:
        """初始化 Milvus 向量存储适配器。"""
        self._uri = uri
        self._token = token
        self._db_name = db_name
        self._collection_name = collection_name
        self._embedding_dimensions = embedding_dimensions
        self._client: Any | None = None
        self._collection_ready = False

    @classmethod
    def from_settings(cls, settings: Settings) -> MilvusChunkVectorStore:
        """基于应用配置创建 Milvus 向量存储适配器。"""
        return cls(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            db_name=settings.milvus_db_name,
            collection_name=settings.milvus_collection_name,
            embedding_dimensions=settings.chunk_embedding_dimensions,
        )

    def ensure_collection(self) -> None:
        """确保向量集合存在且已加载。"""
        if self._collection_ready:
            return

        client = self._get_client()
        try:
            if not client.has_collection(collection_name=self._collection_name):
                client.create_collection(
                    collection_name=self._collection_name,
                    schema=self._build_schema(),
                    index_params=self._build_index_params(),
                    consistency_level="Strong",
                )
            else:
                self._ensure_vector_index(client)
            client.load_collection(collection_name=self._collection_name)
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            msg = f"创建或加载 Milvus 集合失败: {self._collection_name}"
            raise MilvusIndexError(msg) from exc

        self._collection_ready = True

    def ensure_ready(self) -> None:
        """启动期就绪校验：确保客户端可创建且集合可存在/可创建。"""
        try:
            self._get_client()
        except MilvusStoreError:
            raise
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            raise MilvusDependencyError("初始化 Milvus 客户端失败") from exc

        self.ensure_collection()

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """将 chunk 向量批量写入 Milvus。"""
        if not chunks:
            return 0

        self.ensure_collection()
        entities = [self._build_entity(chunk) for chunk in chunks]
        try:
            self._get_client().upsert(
                collection_name=self._collection_name,
                data=entities,
            )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            raise MilvusIndexError("批量写入 Milvus chunk 向量失败") from exc
        return len(chunks)

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """删除指定文件的全部向量实体。"""
        self.ensure_collection()
        try:
            self._get_client().delete(
                collection_name=self._collection_name,
                filter=self._build_file_id_filter(file_id),
            )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            raise MilvusIndexError(f"删除 Milvus chunk 向量失败: {file_id}") from exc

    def search(
        self,
        query_embedding: list[float],
        size: int,
        *,
        viewer_user_id: str = "",
    ) -> list[MilvusVectorSearchHit]:
        """执行向量相似度检索。"""
        if not query_embedding:
            raise MilvusSearchError("查询向量不能为空")

        self.ensure_collection()
        search_kwargs: dict[str, object] = {
            "collection_name": self._collection_name,
            "data": [query_embedding],
            "limit": size,
            "anns_field": self._VECTOR_FIELD_NAME,
            "output_fields": [self._FILE_ID_FIELD_NAME],
            "search_params": {"metric_type": "COSINE", "params": {}},
        }
        if viewer_user_id:
            search_kwargs["filter"] = self._build_visibility_filter(viewer_user_id)

        try:
            response = self._get_client().search(**search_kwargs)
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            raise MilvusSearchError("执行 Milvus 向量检索失败") from exc

        return self._parse_search_result(response)

    def _get_client(self) -> Any:
        """延迟初始化 Milvus 客户端。"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        """创建 Milvus 客户端实例。"""
        milvus_client_class, _ = _load_milvus_dependencies()

        kwargs: dict[str, object] = {"uri": self._uri, "db_name": self._db_name}
        if self._token:
            kwargs["token"] = self._token
        return cast(Any, milvus_client_class(**kwargs))

    def _build_schema(self) -> Any:
        """构造 Milvus 集合 schema。"""
        milvus_client_class, milvus_data_type = _load_milvus_dependencies()

        schema = milvus_client_class.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        # Milvus schema 不支持像 MySQL 那样直接落字段 COMMENT，这里用紧邻中文注释固定字段语义。
        # chunk 唯一标识，也是集合主键与跨库回填锚点。
        schema.add_field(
            field_name=self._PRIMARY_FIELD_NAME,
            datatype=milvus_data_type.VARCHAR,
            is_primary=True,
            max_length=256,
        )
        # 所属文件 ID，用于按文件删除向量与检索后补全文件元数据。
        schema.add_field(
            field_name=self._FILE_ID_FIELD_NAME,
            datatype=milvus_data_type.VARCHAR,
            max_length=128,
        )
        # 上传用户 ID，用于权限过滤和审计。
        schema.add_field(
            field_name=self._UPLOADER_USER_ID_FIELD_NAME,
            datatype=milvus_data_type.VARCHAR,
            max_length=128,
        )
        # 文件可见范围，配合 viewer_user_id 生成检索过滤条件。
        schema.add_field(
            field_name=self._VISIBILITY_SCOPE_FIELD_NAME,
            datatype=milvus_data_type.VARCHAR,
            max_length=32,
        )
        # chunk 正文对应的向量表示，后续图片语义也是通过正文融合进入该向量。
        schema.add_field(
            field_name=self._VECTOR_FIELD_NAME,
            datatype=milvus_data_type.FLOAT_VECTOR,
            dim=self._embedding_dimensions,
        )
        return schema

    def _build_index_params(self) -> Any:
        """构造 Milvus 向量索引参数。"""
        milvus_client_class, _ = _load_milvus_dependencies()

        index_params = milvus_client_class.prepare_index_params()
        index_params.add_index(
            field_name=self._VECTOR_FIELD_NAME,
            index_name=self._VECTOR_INDEX_NAME,
            index_type=self._VECTOR_INDEX_TYPE,
            metric_type=self._VECTOR_METRIC_TYPE,
        )
        return index_params

    def _ensure_vector_index(self, client: Any) -> None:
        """确保向量字段已建立检索索引。"""
        index_names = client.list_indexes(
            collection_name=self._collection_name,
            field_name=self._VECTOR_FIELD_NAME,
        )
        if index_names:
            return

        LOGGER.info(
            "milvus_index_missing collection=%s field=%s index_name=%s creating_index=true",
            self._collection_name,
            self._VECTOR_FIELD_NAME,
            self._VECTOR_INDEX_NAME,
        )
        client.create_index(
            collection_name=self._collection_name,
            index_params=self._build_index_params(),
        )

    def _build_entity(self, chunk: DocumentChunk) -> dict[str, object]:
        """把单个 chunk 转换为 Milvus 实体。"""
        if chunk.content_embedding is None:
            raise MilvusIndexError(f"chunk 缺少向量，无法写入 Milvus: {chunk.chunk_id}")

        return {
            self._PRIMARY_FIELD_NAME: chunk.chunk_id,
            self._FILE_ID_FIELD_NAME: chunk.file_id,
            self._UPLOADER_USER_ID_FIELD_NAME: chunk.uploader_user_id,
            self._VISIBILITY_SCOPE_FIELD_NAME: chunk.visibility_scope,
            self._VECTOR_FIELD_NAME: chunk.content_embedding,
        }

    def _parse_search_result(self, response: object) -> list[MilvusVectorSearchHit]:
        """解析 Milvus 搜索结果。"""
        if not isinstance(response, list) or not response:
            return []

        first_batch = response[0]
        if not isinstance(first_batch, list):
            return []

        hits: list[MilvusVectorSearchHit] = []
        for raw_hit in first_batch:
            if isinstance(raw_hit, dict):
                chunk_id = raw_hit.get("id") or raw_hit.get(self._PRIMARY_FIELD_NAME)
                raw_score = raw_hit.get("distance", raw_hit.get("score"))
            else:
                chunk_id = getattr(raw_hit, "id", None)
                raw_score = getattr(raw_hit, "distance", getattr(raw_hit, "score", None))

            if chunk_id is None:
                continue

            hits.append(
                MilvusVectorSearchHit(
                    chunk_id=str(chunk_id),
                    score=float(raw_score) if isinstance(raw_score, (int, float)) else None,
                )
            )
        return hits

    @staticmethod
    def _build_file_id_filter(file_id: str) -> str:
        """构造按文件标识删除的 Milvus 过滤表达式。"""
        escaped_file_id = file_id.replace("\\", "\\\\").replace('"', '\\"')
        return f'file_id == "{escaped_file_id}"'

    @staticmethod
    def _build_visibility_filter(viewer_user_id: str) -> str:
        """构造按可见性过滤的 Milvus 表达式。"""
        escaped_user_id = viewer_user_id.replace("\\", "\\\\").replace('"', '\\"')
        return f'visibility_scope == "global" or uploader_user_id == "{escaped_user_id}"'


def _load_milvus_dependencies() -> tuple[Any, Any]:
    """延迟加载 pymilvus 依赖，避免模块导入阶段污染环境变量。

    返回:
        一个二元组，依次为 `MilvusClient` 类与 `DataType` 枚举。
    异常:
        MilvusDependencyError: 当运行环境未安装 `pymilvus` 时抛出。
    """
    global MILVUS_CLIENT_CLASS
    global MILVUS_DATA_TYPE
    global MILVUS_IMPORT_ERROR
    global MILVUS_IMPORT_ATTEMPTED

    if not MILVUS_IMPORT_ATTEMPTED:
        MILVUS_IMPORT_ATTEMPTED = True
        try:  # pragma: no cover - 是否安装依赖取决于运行环境
            from pymilvus import (  # type: ignore[import-untyped]
                DataType as ImportedMilvusDataType,
            )
            from pymilvus import (  # type: ignore[import-untyped]
                MilvusClient as ImportedMilvusClient,
            )
        except ImportError as exc:  # pragma: no cover - 测试环境可通过可选导入绕过
            MILVUS_IMPORT_ERROR = exc
        else:  # pragma: no cover - 导入成功路径不需要单独覆盖
            MILVUS_CLIENT_CLASS = ImportedMilvusClient
            MILVUS_DATA_TYPE = ImportedMilvusDataType
            MILVUS_IMPORT_ERROR = None

    if MILVUS_CLIENT_CLASS is None or MILVUS_DATA_TYPE is None:
        raise MilvusDependencyError(
            "未安装 pymilvus 依赖，无法启用 Milvus 向量存储"
        ) from MILVUS_IMPORT_ERROR

    return MILVUS_CLIENT_CLASS, MILVUS_DATA_TYPE
