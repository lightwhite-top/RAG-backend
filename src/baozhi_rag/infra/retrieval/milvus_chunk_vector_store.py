"""Milvus chunk 向量存储适配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from fastapi import status

from baozhi_rag.core.exceptions import AppError

if TYPE_CHECKING:
    from baozhi_rag.core.config import Settings
    from baozhi_rag.services.document_chunking import DocumentChunk

try:  # pragma: no cover - 是否安装依赖取决于运行环境
    from pymilvus import DataType as ImportedMilvusDataType  # type: ignore[import-untyped]
    from pymilvus import MilvusClient as ImportedMilvusClient  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover - 测试环境可通过可选导入绕过
    MILVUS_CLIENT_CLASS: Any | None = None
    MILVUS_DATA_TYPE: Any | None = None
    MILVUS_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - 导入成功路径不需要单独覆盖
    MILVUS_CLIENT_CLASS = ImportedMilvusClient
    MILVUS_DATA_TYPE = ImportedMilvusDataType
    MILVUS_IMPORT_ERROR = None


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
    _VECTOR_FIELD_NAME = "content_embedding"

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
                    consistency_level="Strong",
                )
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
            msg = "初始化 Milvus 客户端失败"
            raise MilvusDependencyError(msg) from exc

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
            msg = "批量写入 Milvus chunk 向量失败"
            raise MilvusIndexError(msg) from exc
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
            msg = f"删除 Milvus chunk 向量失败: {file_id}"
            raise MilvusIndexError(msg) from exc

    def search(self, query_embedding: list[float], size: int) -> list[MilvusVectorSearchHit]:
        """执行向量相似度检索。"""
        if not query_embedding:
            msg = "查询向量不能为空"
            raise MilvusSearchError(msg)

        self.ensure_collection()
        try:
            response = self._get_client().search(
                collection_name=self._collection_name,
                data=[query_embedding],
                limit=size,
                anns_field=self._VECTOR_FIELD_NAME,
                output_fields=[self._FILE_ID_FIELD_NAME],
                search_params={"metric_type": "COSINE", "params": {}},
            )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            msg = "执行 Milvus 向量检索失败"
            raise MilvusSearchError(msg) from exc

        return self._parse_search_result(response)

    def _get_client(self) -> Any:
        """延迟初始化 Milvus 客户端。"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        """创建 Milvus 客户端实例。"""
        if MILVUS_CLIENT_CLASS is None:
            msg = "未安装 pymilvus 依赖，无法启用 Milvus 向量存储"
            raise MilvusDependencyError(msg) from MILVUS_IMPORT_ERROR

        kwargs: dict[str, object] = {"uri": self._uri, "db_name": self._db_name}
        if self._token:
            kwargs["token"] = self._token
        return cast(Any, MILVUS_CLIENT_CLASS(**kwargs))

    def _build_schema(self) -> Any:
        """构造 Milvus 集合 schema。"""
        if MILVUS_CLIENT_CLASS is None or MILVUS_DATA_TYPE is None:
            msg = "未安装 pymilvus 依赖，无法构造 Milvus schema"
            raise MilvusDependencyError(msg) from MILVUS_IMPORT_ERROR

        schema = MILVUS_CLIENT_CLASS.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name=self._PRIMARY_FIELD_NAME,
            datatype=MILVUS_DATA_TYPE.VARCHAR,
            is_primary=True,
            max_length=256,
        )
        schema.add_field(
            field_name=self._FILE_ID_FIELD_NAME,
            datatype=MILVUS_DATA_TYPE.VARCHAR,
            max_length=128,
        )
        schema.add_field(
            field_name=self._VECTOR_FIELD_NAME,
            datatype=MILVUS_DATA_TYPE.FLOAT_VECTOR,
            dim=self._embedding_dimensions,
        )
        return schema

    def _build_entity(self, chunk: DocumentChunk) -> dict[str, object]:
        """把单个 chunk 转换为 Milvus 实体。"""
        if chunk.content_embedding is None:
            msg = f"chunk 缺少向量，无法写入 Milvus: {chunk.chunk_id}"
            raise MilvusIndexError(msg)

        return {
            self._PRIMARY_FIELD_NAME: chunk.chunk_id,
            self._FILE_ID_FIELD_NAME: chunk.file_id,
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
