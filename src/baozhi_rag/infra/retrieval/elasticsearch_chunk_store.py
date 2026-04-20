"""Elasticsearch chunk 文本检索适配。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest
from baozhi_rag.services.document_chunking import ChunkImageAsset

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

    _PRECISE_SEARCH_TERM_PATTERN = re.compile(
        r"(样例\s*\d+|案例\s*\d+|版本\s*[vV]?\s*\d+|第\s*\d+\s*页|"
        r"\d+(?:\.\d+)?\s*(?:个工作日|工作日|小时|分钟|秒|天|周|月|年|元|万元|次|位|页|条|%))",
        re.IGNORECASE,
    )

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
        mappings = self._build_mappings()
        try:
            if not client.indices.exists(index=self._index_name):
                client.indices.create(
                    index=self._index_name,
                    mappings=mappings,
                )
            else:
                # 历史索引可能启用了 strict mapping，但缺少后来新增的字段。
                # 这里统一补齐当前版本需要的属性，避免因为旧 mapping 缺字段导致批量写入失败。
                client.indices.put_mapping(
                    index=self._index_name,
                    dynamic=mappings["dynamic"],
                    properties=mappings["properties"],
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
        should_queries = cls._build_lane_queries(request)
        filter_queries: list[dict[str, object]] = []

        if request.merged_terms:
            should_queries.append(
                {
                    "constant_score": {
                        "filter": {"terms": {"merged_terms": request.merged_terms}},
                        "boost": 6.0,
                    }
                }
            )

        if request.target_document_types:
            filter_queries.append({"terms": {"document_type": request.target_document_types}})

        precise_terms = cls._extract_precise_search_terms(request.query_text)
        if precise_terms:
            should_queries.append(
                {
                    "constant_score": {
                        "filter": {"terms": {"exact_search_terms": precise_terms}},
                        "boost": 12.0 if request.query_intent == "document_location" else 8.0,
                    }
                }
            )
            should_queries.extend(cls._build_ocr_precise_queries(precise_terms))

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

        if request.lane_type == "table":
            filter_queries.append({"term": {"content_type": "table"}})
        if request.lane_type == "ocr":
            filter_queries.append({"term": {"source_type": "ocr"}})

        base_query: dict[str, object] = {
            "bool": {
                "should": should_queries,
                "minimum_should_match": 1,
                "filter": filter_queries,
            }
        }
        version_functions = cls._build_version_functions(request.version_preference)
        if not version_functions:
            return base_query
        return {
            "function_score": {
                "query": base_query,
                "score_mode": "sum",
                "boost_mode": "sum",
                "functions": version_functions,
            }
        }

    @classmethod
    def _build_lane_queries(cls, request: ChunkSearchRequest) -> list[dict[str, object]]:
        """按 lane 类型构造查询子句。"""
        if request.lane_type == "title":
            return cls._build_title_lane_queries(request)
        if request.lane_type == "table":
            return cls._build_table_lane_queries(request)
        if request.lane_type == "ocr":
            return cls._build_ocr_lane_queries(request)
        return cls._build_body_lane_queries(request)

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
            # 去掉编号前缀后的文件名文本，供标题级精确匹配使用。
            "source_filename_text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # 文件在对象存储中的稳定对象键。
            "storage_key": {"type": "keyword"},
            # 版本链排序辅助字段，数字越大代表越新。
            "version_rank": {"type": "integer"},
            # 文档类型，供候选池治理和 lane 过滤使用。
            "document_type": {"type": "keyword"},
            # 版本链所属主题组，供组内选代表使用。
            "version_chain_group": {"type": "keyword"},
            # 来源类型，当前区分普通文本与 OCR 文本。
            "source_type": {"type": "keyword"},
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
            # 表格 schema 文本，用于表格类问题显式提权。
            "table_schema_text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # 行级表格 chunk 的行序号。
            "table_row_index": {"type": "integer"},
            # 行级表格 chunk 的表头摘要。
            "table_header_text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # 领域词命中结果，服务混合召回加权。
            "merged_terms": {"type": "keyword"},
            # OCR、版本链和模板化文档的精确锚点数组。
            "exact_search_terms": {"type": "keyword"},
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
            "storage_key",
            "version_rank",
            "document_type",
            "version_chain_group",
            "source_type",
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
            "table_schema_text",
            "table_row_index",
            "table_header_text",
            "merged_terms",
            "exact_search_terms",
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
            source_filename_text=str(source.get("source_filename_text", "")),
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
            document_type=str(source.get("document_type", "policy")),
            source_type=str(source.get("source_type", "text")),
            version_rank=int(source.get("version_rank", 0)),
            version_chain_group=str(source["version_chain_group"]).strip()
            if source.get("version_chain_group") is not None
            else None,
            table_row_index=int(source.get("table_row_index", 0))
            if source.get("table_row_index") is not None
            else None,
            table_header_text=str(source["table_header_text"]).strip()
            if source.get("table_header_text") is not None
            else None,
            searchable_text=str(source.get("searchable_text", "")),
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

    @classmethod
    def _build_title_lane_queries(cls, request: ChunkSearchRequest) -> list[dict[str, object]]:
        """构造标题 lane 查询。"""
        return cls._build_filename_queries(request) + [
            {
                "match": {
                    "section_title": {
                        "query": request.query_text,
                        "boost": 5.0,
                    }
                }
            },
            {
                "match_phrase": {
                    "section_title": {
                        "query": request.query_text,
                        "boost": 6.5,
                    }
                }
            },
            {
                "match": {
                    "heading_path": {
                        "query": request.query_text,
                        "boost": 3.8,
                    }
                }
            },
        ]

    @classmethod
    def _build_body_lane_queries(cls, request: ChunkSearchRequest) -> list[dict[str, object]]:
        """构造正文 lane 查询。"""
        body_queries: list[dict[str, object]] = [
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
                    "contextual_text": {
                        "query": request.query_text,
                        "boost": 2.2 if request.query_intent != "document_location" else 3.2,
                    }
                }
            },
        ]
        body_queries.extend(
            cls._build_precise_phrase_queries(
                query_text=request.query_text,
                query_intent=request.query_intent,
            )
        )
        if request.query_intent in {"structured", "procedure"}:
            body_queries.extend(cls._build_structure_queries(request))
        return body_queries

    @classmethod
    def _build_table_lane_queries(cls, request: ChunkSearchRequest) -> list[dict[str, object]]:
        """构造表格 lane 查询。"""
        return [
            {
                "match": {
                    "table_schema_text": {
                        "query": request.query_text,
                        "boost": 4.8,
                    }
                }
            },
            {
                "match": {
                    "table_header_text": {
                        "query": request.query_text,
                        "boost": 4.6,
                    }
                }
            },
            {
                "match": {
                    "content": {
                        "query": request.query_text,
                        "boost": 3.5,
                    }
                }
            },
            {
                "match": {
                    "searchable_text": {
                        "query": request.query_text,
                        "boost": 3.8,
                    }
                }
            },
        ]

    @classmethod
    def _build_ocr_lane_queries(cls, request: ChunkSearchRequest) -> list[dict[str, object]]:
        """构造 OCR lane 查询。"""
        queries = cls._build_body_lane_queries(request)
        queries.extend(cls._build_title_lane_queries(request))
        return queries

    @classmethod
    def _build_filename_queries(cls, request: ChunkSearchRequest) -> list[dict[str, object]]:
        """为文件名标题追加单独的精确匹配子句。"""
        match_boost = 6.0 if request.query_intent == "document_location" else 4.2
        phrase_boost = 8.0 if request.query_intent == "document_location" else 5.4
        return [
            {
                "match": {
                    "source_filename_text": {
                        "query": request.query_text,
                        "boost": match_boost,
                    }
                }
            },
            {
                "match_phrase": {
                    "source_filename_text": {
                        "query": request.query_text,
                        "boost": phrase_boost,
                    }
                }
            },
        ]

    @classmethod
    def _build_precise_phrase_queries(
        cls,
        *,
        query_text: str,
        query_intent: str,
    ) -> list[dict[str, object]]:
        """为 OCR、版本链和数字型事实问题追加精确短语查询。"""
        normalized_query = query_text.strip()
        if not normalized_query:
            return []

        phrase_boost = 2.4 if query_intent != "document_location" else 3.2
        precise_queries: list[dict[str, object]] = [
            {
                "match_phrase": {
                    "searchable_text": {
                        "query": normalized_query,
                        "boost": phrase_boost,
                    }
                }
            }
        ]

        compact_query = re.sub(r"\s+", "", normalized_query)
        if compact_query and compact_query != normalized_query and len(compact_query) >= 4:
            precise_queries.append(
                {
                    "match_phrase": {
                        "searchable_text": {
                            "query": compact_query,
                            "boost": phrase_boost + 0.5,
                        }
                    }
                }
            )

        for precise_term in cls._extract_precise_search_terms(normalized_query)[:6]:
            precise_queries.append(
                {
                    "match_phrase": {
                        "searchable_text": {
                            "query": precise_term,
                            "boost": phrase_boost + 0.8,
                        }
                    }
                }
            )
            precise_queries.append(
                {
                    "match_phrase": {
                        "contextual_text": {
                            "query": precise_term,
                            "boost": phrase_boost + 0.4,
                        }
                    }
                }
            )

        return precise_queries

    @classmethod
    def _build_ocr_precise_queries(cls, precise_terms: list[str]) -> list[dict[str, object]]:
        """为 OCR 来源文档追加更强的精确锚点查询。"""
        ocr_queries: list[dict[str, object]] = [
            {
                "constant_score": {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"source_type": "ocr"}},
                                {"terms": {"exact_search_terms": precise_terms}},
                            ]
                        }
                    },
                    "boost": 4.5,
                }
            }
        ]

        fuzzy_terms = cls._build_numeric_fuzzy_terms(precise_terms)
        if fuzzy_terms:
            ocr_queries.append(
                {
                    "constant_score": {
                        "filter": {
                            "bool": {
                                "must": [
                                    {"term": {"source_type": "ocr"}},
                                    {"terms": {"exact_search_terms": fuzzy_terms}},
                                ]
                            }
                        },
                        "boost": 2.5,
                    }
                }
            )
        return ocr_queries

    @classmethod
    def _build_numeric_fuzzy_terms(cls, precise_terms: list[str]) -> list[str]:
        """为数字量词构造 +/-1 容错查询词。"""
        fuzzy_terms: list[str] = []
        seen_terms: set[str] = set()
        for term in precise_terms:
            matched = re.fullmatch(
                r"(\d+)(?:\s*(个工作日|工作日|小时|分钟|秒|天|周|月|年|元|万元|次|位|页|条|%))?",
                term,
            )
            if matched is None:
                continue
            value = int(matched.group(1))
            suffix = matched.group(2) or ""
            for offset in (-1, 1):
                candidate_value = value + offset
                if candidate_value < 0:
                    continue
                candidates = [f"{candidate_value}{suffix}"]
                if suffix:
                    candidates.append(f"{candidate_value} {suffix}")
                for candidate in candidates:
                    if candidate in seen_terms:
                        continue
                    seen_terms.add(candidate)
                    fuzzy_terms.append(candidate)
        return fuzzy_terms

    @classmethod
    def _build_version_functions(cls, version_preference: str) -> list[dict[str, object]]:
        """根据版本偏好决定是否对版本号做检索阶段提权。"""
        if version_preference == "oldest":
            return []
        factor = 0.12 if version_preference == "latest" else 0.08
        return [
            {
                "field_value_factor": {
                    "field": "version_rank",
                    "factor": factor,
                    "modifier": "none",
                    "missing": 0.0,
                }
            }
        ]

    @classmethod
    def _extract_precise_search_terms(cls, query_text: str) -> list[str]:
        """提取查询中的编号与数字量词，供短语匹配提权。"""
        terms: list[str] = []
        for matched_text in cls._PRECISE_SEARCH_TERM_PATTERN.findall(query_text):
            cleaned_text = matched_text.strip()
            if cleaned_text:
                terms.append(cleaned_text)
                compact_text = re.sub(r"\s+", "", cleaned_text)
                if compact_text and compact_text != cleaned_text:
                    terms.append(compact_text)

        deduplicated_terms: list[str] = []
        seen_terms: set[str] = set()
        for term in terms:
            normalized_term = term.strip()
            if not normalized_term or normalized_term in seen_terms:
                continue
            seen_terms.add(normalized_term)
            deduplicated_terms.append(normalized_term)
        return deduplicated_terms


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
