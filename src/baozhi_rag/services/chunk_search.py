"""chunk 检索服务与抽象。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.term_matching import MaximumMatchingTermMatcher

if TYPE_CHECKING:
    from baozhi_rag.services.document_chunking import DocumentChunk


class ChunkSearchValidationError(AppError):
    """检索请求参数非法。"""

    default_message = "检索参数非法"
    default_error_code = "chunk_search_validation_error"
    default_status_code = status.HTTP_400_BAD_REQUEST


@dataclass(frozen=True, slots=True)
class ChunkSearchRequest:
    """chunk 检索请求。"""

    # 用户输入的原始查询文本，经过基础标准化后传给底层检索。
    query_text: str
    # 期望返回的命中数量上限。
    size: int
    # 从查询文本中抽取出的领域词项，用于 ES 检索时显式提权。
    merged_terms: list[str]
    # 查询文本对应的向量表示，用于 Milvus 语义检索。
    query_embedding: list[float]


@dataclass(frozen=True, slots=True)
class ChunkSearchHit:
    """chunk 检索命中结果。"""

    # 命中 chunk 的唯一标识。
    chunk_id: str
    # 命中 chunk 所属文件的唯一标识。
    file_id: str
    # 命中 chunk 对应的原始文件名。
    source_filename: str
    # 命中 chunk 所属文件在本地存储中的相对路径。
    storage_key: str
    # 命中 chunk 在原始文件中的顺序号。
    chunk_index: int
    # 命中 chunk 的字符数。
    char_count: int
    # 命中 chunk 的正文内容。
    content: str
    # 命中 chunk 预先抽取好的领域词项，用于调试和证据展示。
    merged_terms: list[str]
    # 最终返回给上层的相关性分数；可能是 ES 分数或融合分数。
    score: float | None


class ChunkSearchStore(Protocol):
    """chunk 混合检索存储抽象。"""

    def ensure_index(self) -> None:
        """确保底层检索索引或集合已经就绪。

        返回:
            None。
        """
        ...

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """写入 chunk 文档。

        参数:
            chunks: 待写入检索存储的 chunk 列表。

        返回:
            实际写入的 chunk 数量。
        """
        ...

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """按文件标识删除 chunk。

        参数:
            file_id: 需要删除的文件唯一标识。

        返回:
            None。
        """
        ...

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        """执行 chunk 检索。

        参数:
            request: 已完成查询词标准化、领域词抽取和向量化的检索请求。

        返回:
            按相关性排序后的 chunk 命中结果列表。
        """
        ...


class ChunkSearchService:
    """编排查询词分解、向量化与混合检索。"""

    def __init__(
        self,
        term_matcher: MaximumMatchingTermMatcher,
        store: ChunkSearchStore,
        chunk_embedding_service: ChunkEmbeddingService,
    ) -> None:
        """初始化检索服务。

        参数:
            term_matcher: 领域词匹配器，用于从查询文本中抽取 `merged_terms`。
            store: 混合检索存储抽象，负责执行 ES 和 Milvus 检索。
            chunk_embedding_service: 查询向量化服务，用于生成语义检索向量。

        返回:
            None。
        """
        self._term_matcher = term_matcher
        self._store = store
        self._chunk_embedding_service = chunk_embedding_service

    def search(self, query_text: str, size: int) -> list[ChunkSearchHit]:
        """执行基于 ES 词法与 Milvus 语义的混合检索。

        参数:
            query_text: 用户原始查询文本。
            size: 期望返回的命中数量。

        返回:
            由底层混合检索存储返回的 chunk 命中结果列表。

        异常:
            ChunkSearchValidationError: 当查询文本为空或 `size` 非法时抛出。
        """

        # 查询词标准化：去除首尾空白字符，确保后续处理基于规范化文本进行
        normalized_query = query_text.strip()

        if not normalized_query:
            msg = "查询文本不能为空"
            raise ChunkSearchValidationError(msg)
        if size <= 0:
            msg = "size 必须大于 0"
            raise ChunkSearchValidationError(msg)

        # 领域词抽取：基于最大匹配算法从查询文本中抽取领域词项，并进行合并去重
        terms = self._term_matcher.extract_terms(normalized_query)

        #  构建检索请求：将标准化查询文本、期望返回数量、抽取的领域词项和查询向量封装成检索请求对象
        request = ChunkSearchRequest(
            query_text=normalized_query,
            size=size,
            merged_terms=terms.merged_terms,
            query_embedding=self._chunk_embedding_service.embed_query(normalized_query),
        )
        return self._store.search(request)
