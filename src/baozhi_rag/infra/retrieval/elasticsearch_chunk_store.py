"""Elasticsearch chunk 文本检索适配。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest

if TYPE_CHECKING:
    from baozhi_rag.core.config import Settings
    from baozhi_rag.services.document_chunking import DocumentChunk

try:  # pragma: no cover - 是否安装依赖取决于运行环境
    from elasticsearch import Elasticsearch as ImportedElasticsearchClient
except ImportError as exc:  # pragma: no cover - 测试环境可通过可选导入绕过
    ELASTICSEARCH_CLIENT_CLASS: Any | None = None
    ELASTICSEARCH_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - 导入成功路径不需要单独覆盖
    ELASTICSEARCH_CLIENT_CLASS = ImportedElasticsearchClient
    ELASTICSEARCH_IMPORT_ERROR = None


class ElasticsearchStoreError(AppError):
    """Elasticsearch 适配层异常。"""

    default_message = "Elasticsearch 调用失败"
    default_error_code = "elasticsearch_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class ElasticsearchDependencyError(ElasticsearchStoreError):
    """Elasticsearch 客户端依赖缺失。"""

    default_message = "Elasticsearch 客户端依赖缺失"
    default_error_code = "elasticsearch_dependency_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class ElasticsearchIndexError(ElasticsearchStoreError):
    """Elasticsearch 索引或写入异常。"""

    default_message = "Elasticsearch 索引写入失败"
    default_error_code = "elasticsearch_index_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class ElasticsearchSearchError(ElasticsearchStoreError):
    """Elasticsearch 检索异常。"""

    default_message = "Elasticsearch 检索失败"
    default_error_code = "elasticsearch_search_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class ElasticsearchChunkStore:
    """负责 chunk 文档索引创建、写入、读取与检索。"""

    def __init__(
        self,
        *,
        index_name: str,
        url: str,
        api_key: str | None,
        username: str | None,
        password: str | None,
        verify_certs: bool,
        embedding_dimensions: int,
    ) -> None:
        """初始化 ES 文档存储适配器。

        参数:
            index_name: chunk 文档索引名称。
            url: Elasticsearch 服务地址。
            api_key: ES API Key；启用时优先于用户名密码。
            username: ES 用户名。
            password: ES 密码。
            verify_certs: 是否校验 HTTPS 证书。
            embedding_dimensions: 预留的向量维度配置，当前主要用于保持构造参数一致性。

        返回:
            None。
        """
        self._index_name = index_name
        self._url = url
        self._api_key = api_key
        self._username = username
        self._password = password
        self._verify_certs = verify_certs
        self._embedding_dimensions = embedding_dimensions
        self._client: Any | None = None
        self._index_ready = False

    @classmethod
    def from_settings(cls, settings: Settings) -> ElasticsearchChunkStore:
        """基于应用配置创建 ES 适配器。

        参数:
            settings: 当前应用配置对象。

        返回:
            已按配置完成连接参数装配的 ES 存储实例。
        """
        return cls(
            index_name=settings.es_index_name,
            url=settings.es_url,
            api_key=settings.es_api_key,
            username=settings.es_username,
            password=settings.es_password,
            verify_certs=settings.es_verify_certs,
            embedding_dimensions=settings.chunk_embedding_dimensions,
        )

    def ensure_index(self) -> None:
        """确保 chunk 索引存在。

        返回:
            None。

        异常:
            ElasticsearchIndexError: 当索引检查或创建失败时抛出。
        """
        if self._index_ready:
            return

        client = self._get_client()
        try:
            if not client.indices.exists(index=self._index_name):
                client.indices.create(
                    index=self._index_name,
                    mappings=self._build_mappings(),
                )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            msg = f"创建或检查 ES 索引失败: {self._index_name}"
            raise ElasticsearchIndexError(msg) from exc

        self._index_ready = True

    def ensure_ready(self) -> None:
        """启动期就绪校验：确保客户端可创建且索引可存在/可创建。

        返回:
            None。

        异常:
            ElasticsearchDependencyError: 当客户端依赖或初始化失败时抛出。
            ElasticsearchIndexError: 当索引检查或创建失败时抛出。
        """
        try:
            self._get_client()
        except ElasticsearchStoreError:
            raise
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            msg = "初始化 ES 客户端失败"
            raise ElasticsearchDependencyError(msg) from exc

        self.ensure_index()

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """将 chunk 文档批量写入 ES。

        参数:
            chunks: 待写入 ES 的 chunk 列表。

        返回:
            实际写入的 chunk 数量。

        异常:
            ElasticsearchIndexError: 当批量写入失败时抛出。
        """
        if not chunks:
            return 0

        self.ensure_index()
        operations = self._build_bulk_operations(chunks)

        try:
            response = self._get_client().bulk(operations=operations, refresh="wait_for")
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            msg = "批量写入 ES chunk 失败"
            raise ElasticsearchIndexError(msg) from exc

        if response.get("errors"):
            error_reason = self._extract_bulk_error_reason(response)
            msg = f"批量写入 ES chunk 失败: {error_reason}"
            raise ElasticsearchIndexError(msg)

        return len(chunks)

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """删除指定文件的全部 chunk 文档。

        参数:
            file_id: 需要删除的文件唯一标识。

        返回:
            None。

        异常:
            ElasticsearchIndexError: 当删除失败时抛出。
        """
        self.ensure_index()
        try:
            self._get_client().delete_by_query(
                index=self._index_name,
                query={"term": {"file_id": file_id}},
                refresh=True,
                conflicts="proceed",
            )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            msg = f"删除 ES chunk 失败: {file_id}"
            raise ElasticsearchIndexError(msg) from exc

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        """执行 chunk 词法检索。

        参数:
            request: 已完成领域词抽取和查询向量生成的检索请求。

        返回:
            ES 返回的词法检索命中结果列表。

        异常:
            ElasticsearchSearchError: 当检索执行失败时抛出。
        """
        self.ensure_index()

        try:
            response = self._get_client().search(
                index=self._index_name,
                query=self.build_search_query(request),
                size=request.size,
                source=self._build_source_fields(),
            )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            msg = "执行 ES chunk 检索失败"
            raise ElasticsearchSearchError(msg) from exc

        hits = response.get("hits", {}).get("hits", [])
        return [self._parse_hit(hit) for hit in hits]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[ChunkSearchHit]:
        """按 chunk 标识批量获取文档内容。

        参数:
            chunk_ids: 需要回查的 chunk 标识列表。

        返回:
            与输入顺序一致的 chunk 命中列表；缺失文档会被自动跳过。

        异常:
            ElasticsearchSearchError: 当批量读取失败时抛出。
        """
        if not chunk_ids:
            return []

        self.ensure_index()
        try:
            response = self._get_client().mget(index=self._index_name, ids=chunk_ids)
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            msg = "按 chunk_id 批量读取 ES 文档失败"
            raise ElasticsearchSearchError(msg) from exc

        docs = response.get("docs", [])
        if not isinstance(docs, list):
            return []

        hit_map: dict[str, ChunkSearchHit] = {}
        for doc in docs:
            if not isinstance(doc, dict) or not doc.get("found"):
                continue
            source = doc.get("_source", {})
            if not isinstance(source, dict):
                continue
            hit = self._parse_hit({"_source": source, "_score": None})
            hit_map[hit.chunk_id] = hit

        return [hit_map[chunk_id] for chunk_id in chunk_ids if chunk_id in hit_map]

    @classmethod
    def build_search_query(cls, request: ChunkSearchRequest) -> dict[str, object]:
        """构造结合全文与领域词的 ES 词法查询。

        参数:
            request: 当前检索请求，包含查询文本与 `merged_terms`。

        返回:
            可直接传给 Elasticsearch 的查询 DSL。
        """
        should_queries: list[dict[str, object]] = [
            {
                "match": {
                    "content": {
                        "query": request.query_text,
                        "boost": 3.0,
                    }
                }
            }
        ]

        if request.merged_terms:
            should_queries.append(
                {
                    "constant_score": {
                        "filter": {"terms": {"merged_terms": request.merged_terms}},
                        "boost": 6.0,
                    }
                }
            )

        return {
            "bool": {
                "should": should_queries,
                "minimum_should_match": 1,
            }
        }

    def _get_client(self) -> Any:
        """延迟初始化 ES 客户端。"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        """创建 ES 客户端实例。"""
        if ELASTICSEARCH_CLIENT_CLASS is None:
            msg = "未安装 elasticsearch 依赖，无法启用 ES 检索"
            raise ElasticsearchDependencyError(msg) from ELASTICSEARCH_IMPORT_ERROR

        if self._api_key:
            return cast(
                Any,
                ELASTICSEARCH_CLIENT_CLASS(
                    hosts=[self._url],
                    verify_certs=self._verify_certs,
                    api_key=self._api_key,
                ),
            )
        if self._username and self._password:
            return cast(
                Any,
                ELASTICSEARCH_CLIENT_CLASS(
                    hosts=[self._url],
                    verify_certs=self._verify_certs,
                    basic_auth=(self._username, self._password),
                ),
            )
        return cast(
            Any,
            ELASTICSEARCH_CLIENT_CLASS(
                hosts=[self._url],
                verify_certs=self._verify_certs,
            ),
        )

    def _build_bulk_operations(self, chunks: list[DocumentChunk]) -> list[dict[str, object]]:
        """构造 ES bulk 写入载荷。"""
        operations: list[dict[str, object]] = []
        for chunk in chunks:
            operations.append({"index": {"_index": self._index_name, "_id": chunk.chunk_id}})
            operations.append(chunk.to_search_document())
        return operations

    def _build_mappings(self) -> dict[str, object]:
        """构造 chunk 索引 mapping。"""
        properties: dict[str, object] = {
            "chunk_id": {"type": "keyword"},
            "file_id": {"type": "keyword"},
            "source_filename": {"type": "keyword"},
            "storage_key": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "char_count": {"type": "integer"},
            "content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            "merged_terms": {"type": "keyword"},
        }

        return {
            "dynamic": "strict",
            "properties": properties,
        }

    @staticmethod
    def _build_source_fields() -> list[str]:
        """限定检索返回字段。"""
        return [
            "chunk_id",
            "file_id",
            "source_filename",
            "storage_key",
            "chunk_index",
            "char_count",
            "content",
            "merged_terms",
        ]

    @staticmethod
    def _extract_bulk_error_reason(response: dict[str, object]) -> str:
        """从 bulk 响应中提取首个错误原因。"""
        items = response.get("items", [])
        if not isinstance(items, list):
            return "未知错误"

        for item in items:
            if not isinstance(item, dict):
                continue
            action_result = next(iter(item.values()), None)
            if not isinstance(action_result, dict):
                continue
            error = action_result.get("error")
            if not isinstance(error, dict):
                continue
            reason = error.get("reason")
            if isinstance(reason, str) and reason:
                return reason
        return "未知错误"

    @staticmethod
    def _parse_hit(hit: dict[str, object]) -> ChunkSearchHit:
        """解析 ES 命中结果。"""
        source = hit.get("_source", {})
        score = hit.get("_score")
        if not isinstance(source, dict):
            source = {}

        return ChunkSearchHit(
            chunk_id=str(source.get("chunk_id", "")),
            file_id=str(source.get("file_id", "")),
            source_filename=str(source.get("source_filename", "")),
            storage_key=str(source.get("storage_key", "")),
            chunk_index=int(source.get("chunk_index", 0)),
            char_count=int(source.get("char_count", 0)),
            content=str(source.get("content", "")),
            merged_terms=_as_string_list(source.get("merged_terms")),
            score=float(score) if isinstance(score, (int, float)) else None,
        )


def _as_string_list(value: object) -> list[str]:
    """将未知值安全转换为字符串列表。"""
    if isinstance(value, list):
        return [str(item) for item in value]
    return []
