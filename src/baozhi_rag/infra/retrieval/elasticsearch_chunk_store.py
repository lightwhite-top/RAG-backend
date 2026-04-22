"""Elasticsearch chunk 文本检索适配。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest
from baozhi_rag.services.document_chunking import ChunkImageAsset
from baozhi_rag.services.retrieval_signals import (
    extract_exact_search_terms,
    query_prefers_old_version,
    query_requests_latest_version,
)

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
        """构造结合全文、领域词与用户可见性的 ES 词法查询。"""
        should_queries: list[dict[str, object]] = [
            {
                "match": {
                    "searchable_text": {
                        "query": request.query_text,
                        "boost": 3.0,
                    }
                }
            },
            {
                "match": {
                    "source_filename_text": {
                        "query": request.query_text,
                        "boost": 1.6,
                    }
                }
            },
        ]
        filter_queries: list[dict[str, object]] = []
        exact_search_terms = cls._extract_exact_search_terms(request.query_text)

        if request.merged_terms:
            should_queries.append(
                {
                    "constant_score": {
                        "filter": {"terms": {"merged_terms": request.merged_terms}},
                        "boost": 6.0,
                    }
                }
            )

        should_queries.extend(cls._build_phrase_queries(request, exact_search_terms))
        should_queries.extend(cls._build_structure_queries(request))
        should_queries.extend(cls._build_exact_term_queries(exact_search_terms))

        if request.viewer_user_id:
            filter_queries.append(
                {
                    "bool": {
                        "should": [
                            {"term": {"visibility_scope": "global"}},
                            {"term": {"uploader_user_id": request.viewer_user_id}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )

        bool_query: dict[str, object] = {
            "bool": {
                "should": should_queries,
                "minimum_should_match": 1,
                "filter": filter_queries,
            }
        }
        if cls._should_wrap_with_version_boost(request.query_text):
            return {
                "function_score": {
                    "query": bool_query,
                    "functions": [
                        {
                            "field_value_factor": {
                                "field": "version_rank",
                                "factor": 0.12,
                                "modifier": "none",
                                "missing": 0.0,
                            }
                        }
                    ],
                    "boost_mode": "sum",
                    "score_mode": "sum",
                }
            }
        return bool_query

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
        # ES mapping 不支持像 MySQL 那样直接落字段 COMMENT，这里用紧邻中文注释固定字段语义。
        properties: dict[str, object] = {
            # chunk 唯一标识，也是跨库回填时的主锚点。
            "chunk_id": {"type": "keyword"},
            # 所属文件 ID，用于删除、审计和回填文件元数据。
            "file_id": {"type": "keyword"},
            # 原始文件名，供搜索结果展示与引用回填使用。
            "source_filename": {"type": "keyword"},
            # 从文件名规整得到的标题文本，用于文件名语义召回。
            "source_filename_text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # 从文件名解析出的版本号，供版本链排序提权。
            "version_rank": {"type": "integer"},
            # 文件在对象存储中的稳定对象键。
            "storage_key": {"type": "keyword"},
            "page_number": {"type": "integer"},
            "source_anchor": {"type": "keyword"},
            # 上传者用户 ID，用于权限过滤与审计追踪。
            "uploader_user_id": {"type": "keyword"},
            # 文件可见范围，控制 owner_only/global 检索边界。
            "visibility_scope": {"type": "keyword"},
            # chunk 类型，区分正文 chunk 与图片语义 chunk。
            "chunk_type": {"type": "keyword"},
            # 原始解析片段 ID，用于命中后合并图片与正文关系。
            "segment_id": {"type": "keyword"},
            # chunk 在原文件中的顺序编号。
            "chunk_index": {"type": "integer"},
            # chunk 字符数，便于前端展示与检索调试。
            "char_count": {"type": "integer"},
            # chunk 所属标题路径，支持按章节上下文召回。
            "heading_path": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # chunk 所属末级标题，支持标题命中提权。
            "section_title": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # chunk 内容类型，当前主要区分 paragraph / table。
            "content_type": {"type": "keyword"},
            # chunk 正文；文本 chunk 存正文，图片语义 chunk 存图片稳定语义文本。
            "content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # 原始可引用正文，用于与增强检索文本分离。
            "raw_content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # 轻量上下文化文本，仅用于增强检索。
            "contextual_text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # 检索专用拼接文本，通常由 contextual_text + raw_content 构成。
            "searchable_text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # 文本来源类型，便于对 OCR 文本进行额外精确提权。
            "source_type": {"type": "keyword"},
            # 数字量词、样例编号等精确锚点，用于 constant_score 精确匹配。
            "exact_search_terms": {"type": "keyword"},
            # 表格 schema 文本，用于表格类问题显式提权。
            "table_schema_text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # 领域词命中结果，服务混合召回加权。
            "merged_terms": {"type": "keyword"},
            # 文本 chunk 上挂载的图片引用键列表。
            "image_asset_refs": {
                "type": "nested",
                "properties": {
                    # 图片所属原始片段 ID。
                    "segment_id": {"type": "keyword"},
                    # 图片资产唯一标识。
                    "asset_id": {"type": "keyword"},
                },
            },
            # 图片资产投影，供检索命中后补全前端渲染信息。
            "image_assets": {
                "type": "nested",
                "properties": {
                    # 图片所属原始片段 ID。
                    "segment_id": {"type": "keyword"},
                    # 图片资产唯一标识。
                    "asset_id": {"type": "keyword"},
                    # 图片在原文中的稳定锚点。
                    "source_anchor": {"type": "keyword"},
                    # 原图在对象存储中的稳定对象键。
                    "storage_key": {"type": "keyword"},
                    # 缩略图在对象存储中的稳定对象键。
                    "thumbnail_storage_key": {"type": "keyword"},
                    # 图片类型，如流程图、印章、手写批注等。
                    "image_type": {"type": "keyword"},
                    # 图片语义摘要，参与全文检索补充。
                    "summary": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    # 图片 OCR 文本，参与全文检索补充。
                    "ocr_text": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                },
            },
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
            "source_filename_text",
            "version_rank",
            "storage_key",
            "page_number",
            "source_anchor",
            "uploader_user_id",
            "visibility_scope",
            "chunk_type",
            "segment_id",
            "chunk_index",
            "char_count",
            "heading_path",
            "section_title",
            "content_type",
            "content",
            "raw_content",
            "contextual_text",
            "searchable_text",
            "source_type",
            "exact_search_terms",
            "table_schema_text",
            "merged_terms",
            "image_asset_refs",
            "image_assets",
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
            chunk_type=str(source.get("chunk_type", "text")),
            segment_id=str(source.get("segment_id", "")),
            source_filename=str(source.get("source_filename", "")),
            storage_key=str(source.get("storage_key", "")),
            page_number=int(source.get("page_number", 0))
            if source.get("page_number") is not None
            else None,
            source_anchor=str(source.get("source_anchor", "")).strip() or None,
            uploader_user_id=str(source.get("uploader_user_id", "")),
            visibility_scope=str(source.get("visibility_scope", "")),
            chunk_index=int(source.get("chunk_index", 0)),
            char_count=int(source.get("char_count", 0)),
            heading_path=_as_string_list(source.get("heading_path")),
            section_title=str(source["section_title"]).strip()
            if source.get("section_title") is not None
            else None,
            content_type=str(source.get("content_type", "paragraph")),
            content=str(source.get("content", "")),
            merged_terms=_as_string_list(source.get("merged_terms")),
            image_assets=_as_image_assets(source.get("image_assets")),
            score=float(score) if isinstance(score, (int, float)) else None,
        )

    @classmethod
    def _build_structure_queries(cls, request: ChunkSearchRequest) -> list[dict[str, object]]:
        """构造标题、章节和内容类型相关的结构化检索子句。"""
        title_boost = 2.0
        heading_boost = 1.5
        if request.query_intent == "document_location":
            title_boost = 4.0
            heading_boost = 3.0
        elif request.query_intent == "procedure":
            title_boost = 2.6
            heading_boost = 2.0

        structure_queries: list[dict[str, object]] = [
            {
                "match": {
                    "section_title": {
                        "query": request.query_text,
                        "boost": title_boost,
                    }
                }
            },
            {
                "match": {
                    "heading_path": {
                        "query": request.query_text,
                        "boost": heading_boost,
                    }
                }
            },
            {
                "match": {
                    "contextual_text": {
                        "query": request.query_text,
                        "boost": 2.2 if request.query_intent != "document_location" else 3.2,
                    }
                }
            },
        ]

        if request.query_intent == "structured":
            structure_queries.append(
                {
                    "constant_score": {
                        "filter": {"term": {"content_type": "table"}},
                        "boost": 2.5,
                    }
                }
            )
            structure_queries.append(
                {
                    "match": {
                        "table_schema_text": {
                            "query": request.query_text,
                            "boost": 3.2,
                        }
                    }
                }
            )

        return structure_queries

    @staticmethod
    def _build_phrase_queries(
        request: ChunkSearchRequest,
        exact_search_terms: list[str],
    ) -> list[dict[str, object]]:
        """构造短语精确匹配子句。"""
        phrase_queries: list[dict[str, object]] = [
            {
                "match_phrase": {
                    "searchable_text": {
                        "query": request.query_text,
                        "boost": 4.2 if request.query_intent == "document_location" else 2.8,
                    }
                }
            },
            {
                "match_phrase": {
                    "source_filename_text": {
                        "query": request.query_text,
                        "boost": 2.6,
                    }
                }
            },
        ]
        for exact_term in exact_search_terms:
            phrase_queries.append(
                {
                    "match_phrase": {
                        "searchable_text": {
                            "query": exact_term,
                            "boost": 3.6,
                        }
                    }
                }
            )
        return phrase_queries

    @staticmethod
    def _build_exact_term_queries(exact_search_terms: list[str]) -> list[dict[str, object]]:
        """构造基于 exact_search_terms 的常量分值精确匹配。"""
        if not exact_search_terms:
            return []

        return [
            {
                "constant_score": {
                    "filter": {"terms": {"exact_search_terms": exact_search_terms}},
                    "boost": 6.8,
                }
            },
            {
                "constant_score": {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"source_type": "ocr"}},
                                {"terms": {"exact_search_terms": exact_search_terms}},
                            ]
                        }
                    },
                    "boost": 7.2,
                }
            },
        ]

    @staticmethod
    def _extract_exact_search_terms(query_text: str) -> list[str]:
        """从查询文本中提取精确匹配锚点。"""
        return extract_exact_search_terms(query_text)

    @staticmethod
    def _should_wrap_with_version_boost(query_text: str) -> bool:
        """判断是否应按版本号对结果做额外提权。"""
        if query_prefers_old_version(query_text):
            return False
        return query_requests_latest_version(query_text)


def _as_string_list(value: object) -> list[str]:
    """将未知值安全转换为字符串列表。"""
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _as_image_assets(value: object) -> list[ChunkImageAsset]:
    """把 ES 投影中的图片资产转换为 chunk 图片资产对象。"""
    if not isinstance(value, list):
        return []

    image_assets: list[ChunkImageAsset] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        image_assets.append(
            ChunkImageAsset(
                segment_id=str(item.get("segment_id", "")),
                asset_id=str(item.get("asset_id", "")),
                asset_index=index,
                source_anchor=str(item.get("source_anchor", "")),
                content_type="",
                extension=Path(str(item.get("storage_key", ""))).suffix or ".bin",
                image_bytes=None,
                storage_key=str(item.get("storage_key", "")),
                thumbnail_storage_key=str(item["thumbnail_storage_key"])
                if item.get("thumbnail_storage_key") is not None
                else None,
                summary=str(item.get("summary", "")),
                ocr_text=str(item.get("ocr_text", "")),
                image_type=str(item.get("image_type", "")),
            )
        )
    return image_assets
