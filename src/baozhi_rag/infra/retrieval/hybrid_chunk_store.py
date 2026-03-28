"""ES 文本检索 + Milvus 向量检索混合适配。"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.infra.retrieval.elasticsearch_chunk_store import (
    ElasticsearchChunkStore,
    ElasticsearchDependencyError,
    ElasticsearchSearchError,
    ElasticsearchStoreError,
)
from baozhi_rag.infra.retrieval.milvus_chunk_vector_store import (
    MilvusChunkVectorStore,
    MilvusDependencyError,
    MilvusSearchError,
    MilvusStoreError,
    MilvusVectorSearchHit,
)
from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest, ChunkSearchStore
from baozhi_rag.services.document_chunking import DocumentChunk

if TYPE_CHECKING:
    from baozhi_rag.core.config import Settings


class HybridDocumentStore(Protocol):
    """混合检索依赖的文档存储协议。"""

    def ensure_ready(self) -> None:
        """启动期检查文档存储。"""
        ...

    def ensure_index(self) -> None:
        """确保文档索引存在。"""
        ...

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """批量写入文档 chunk。"""
        ...

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """按文件标识删除 chunk。"""
        ...

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        """执行文档检索。"""
        ...

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[ChunkSearchHit]:
        """按 chunk_id 获取文档。"""
        ...


class HybridVectorStore(Protocol):
    """混合检索依赖的向量存储协议。"""

    def ensure_ready(self) -> None:
        """启动期检查向量存储。"""
        ...

    def ensure_collection(self) -> None:
        """确保向量集合存在。"""
        ...

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """批量写入向量。"""
        ...

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """按文件标识删除向量。"""
        ...

    def search(self, query_embedding: list[float], size: int) -> list[MilvusVectorSearchHit]:
        """执行向量检索。"""
        ...


class HybridChunkStoreError(AppError):
    """混合检索适配层异常。"""

    default_message = "混合检索执行失败"
    default_error_code = "hybrid_chunk_store_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class HybridChunkStoreDependencyError(HybridChunkStoreError):
    """混合检索依赖缺失或不可用。"""

    default_message = "混合检索依赖不可用"
    default_error_code = "hybrid_chunk_store_dependency_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class HybridChunkStoreSearchError(HybridChunkStoreError):
    """混合检索执行异常。"""

    default_message = "混合检索执行失败"
    default_error_code = "hybrid_chunk_store_search_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class HybridChunkStore(ChunkSearchStore):
    """负责 ES 文档检索与 Milvus 向量检索的统一编排。"""

    _RRF_K = 60

    def __init__(
        self,
        document_store: HybridDocumentStore,
        vector_store: HybridVectorStore,
    ) -> None:
        """初始化混合检索存储。

        参数:
            document_store: 词法检索侧文档存储，当前实现通常为 Elasticsearch。
            vector_store: 语义检索侧向量存储，当前实现通常为 Milvus。

        返回:
            None。
        """
        self._document_store = document_store
        self._vector_store = vector_store

    @classmethod
    def from_settings(cls, settings: Settings) -> HybridChunkStore:
        """基于应用配置创建混合检索存储。

        参数:
            settings: 当前应用配置对象。

        返回:
            已按配置完成依赖装配的混合检索存储实例。
        """
        return cls(
            document_store=ElasticsearchChunkStore.from_settings(settings),
            vector_store=MilvusChunkVectorStore.from_settings(settings),
        )

    def ensure_ready(self) -> None:
        """启动期同时校验 ES 与 Milvus 可用。

        返回:
            None。

        异常:
            HybridChunkStoreDependencyError: 当 ES 或 Milvus 依赖不可用时抛出。
            HybridChunkStoreError: 当底层存储初始化失败时抛出。
        """
        try:
            self._document_store.ensure_ready()
            self._vector_store.ensure_ready()
        except (ElasticsearchDependencyError, MilvusDependencyError) as exc:
            raise HybridChunkStoreDependencyError(str(exc)) from exc
        except (ElasticsearchStoreError, MilvusStoreError) as exc:
            raise HybridChunkStoreError(str(exc)) from exc

    def ensure_index(self) -> None:
        """确保 ES 索引和 Milvus 集合均已就绪。

        返回:
            None。

        异常:
            HybridChunkStoreDependencyError: 当底层依赖不可用时抛出。
            HybridChunkStoreError: 当索引或集合初始化失败时抛出。
        """
        try:
            self._document_store.ensure_index()
            self._vector_store.ensure_collection()
        except (ElasticsearchDependencyError, MilvusDependencyError) as exc:
            raise HybridChunkStoreDependencyError(str(exc)) from exc
        except (ElasticsearchStoreError, MilvusStoreError) as exc:
            raise HybridChunkStoreError(str(exc)) from exc

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """同时写入 ES 文档与 Milvus 向量。

        参数:
            chunks: 已完成切块和向量化的 chunk 列表。

        返回:
            实际完成双写的 chunk 数量。

        异常:
            HybridChunkStoreDependencyError: 当 ES 或 Milvus 依赖不可用时抛出。
            HybridChunkStoreError: 当双写过程失败时抛出。
        """
        if not chunks:
            return 0

        self.ensure_index()
        file_ids = list(dict.fromkeys(chunk.file_id for chunk in chunks))
        try:
            self._document_store.index_chunks(chunks)
            self._vector_store.index_chunks(chunks)
        except (ElasticsearchDependencyError, MilvusDependencyError) as exc:
            self._rollback_file_ids(file_ids)
            raise HybridChunkStoreDependencyError(str(exc)) from exc
        except (ElasticsearchStoreError, MilvusStoreError) as exc:
            self._rollback_file_ids(file_ids)
            raise HybridChunkStoreError(str(exc)) from exc
        return len(chunks)

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """同时删除 ES 文档与 Milvus 向量。

        参数:
            file_id: 需要删除的文件唯一标识。

        返回:
            None。

        异常:
            HybridChunkStoreDependencyError: 当底层依赖不可用时抛出。
            HybridChunkStoreError: 当删除过程失败时抛出。
        """
        first_error: Exception | None = None

        for delete_operation in (
            lambda: self._vector_store.delete_chunks_by_file_id(file_id),
            lambda: self._document_store.delete_chunks_by_file_id(file_id),
        ):
            try:
                delete_operation()
            except (ElasticsearchDependencyError, MilvusDependencyError) as exc:
                if first_error is None:
                    first_error = HybridChunkStoreDependencyError(str(exc))
            except (ElasticsearchStoreError, MilvusStoreError) as exc:
                if first_error is None:
                    first_error = HybridChunkStoreError(str(exc))

        if first_error is not None:
            raise first_error

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        """执行 ES 词法检索与 Milvus 向量检索并做结果融合。

        参数:
            request: 已包含查询文本、领域词和查询向量的检索请求。

        返回:
            经过 RRF 融合后的 chunk 命中结果列表。

        异常:
            HybridChunkStoreDependencyError: 当底层依赖不可用时抛出。
            HybridChunkStoreSearchError: 当检索执行失败时抛出。
        """
        try:
            # 文档检索
            lexical_hits = self._document_store.search(request)
            # 向量检索
            semantic_hits = self._vector_store.search(request.query_embedding, request.size)

            return self._fuse_hits(
                lexical_hits=lexical_hits,
                semantic_hits=semantic_hits,
                size=request.size,
            )
        except (ElasticsearchDependencyError, MilvusDependencyError) as exc:
            raise HybridChunkStoreDependencyError(str(exc)) from exc
        except (
            ElasticsearchSearchError,
            MilvusSearchError,
            ElasticsearchStoreError,
            MilvusStoreError,
        ) as exc:
            raise HybridChunkStoreSearchError(str(exc)) from exc

    def _rollback_file_ids(self, file_ids: list[str]) -> None:
        """在双写失败时尽量回滚 ES 与 Milvus。

        参数:
            file_ids: 本次已经部分完成写入的文件标识列表。

        返回:
            None。回滚过程中的异常会被抑制。
        """
        for file_id in reversed(file_ids):
            with suppress(Exception):
                self.delete_chunks_by_file_id(file_id)

    def _fuse_hits(
        self,
        *,
        lexical_hits: list[ChunkSearchHit],
        semantic_hits: list[MilvusVectorSearchHit],
        size: int,
    ) -> list[ChunkSearchHit]:
        """使用 RRF 融合词法结果与向量结果。

        参数:
            lexical_hits: ES 返回的词法检索命中列表。
            semantic_hits: Milvus 返回的向量检索命中列表。
            size: 融合后最多保留的结果数量。

        返回:
            经过融合、补全文档载荷并重新排序后的 chunk 列表。
        """
        fused_scores: dict[str, float] = {}
        hit_map = {hit.chunk_id: hit for hit in lexical_hits}

        for rank, hit in enumerate(lexical_hits, start=1):
            fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + self._rrf_score(rank)

        for rank, semantic_hit in enumerate(semantic_hits, start=1):
            fused_scores[semantic_hit.chunk_id] = fused_scores.get(
                semantic_hit.chunk_id,
                0.0,
            ) + self._rrf_score(rank)

        semantic_only_ids = [hit.chunk_id for hit in semantic_hits if hit.chunk_id not in hit_map]
        if semantic_only_ids:
            for hit in self._document_store.get_chunks_by_ids(semantic_only_ids):
                hit_map[hit.chunk_id] = hit

        ordered_chunk_ids = [
            chunk_id
            for chunk_id, _ in sorted(
                fused_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if chunk_id in hit_map
        ]

        fused_hits: list[ChunkSearchHit] = []
        for chunk_id in ordered_chunk_ids[:size]:
            fused_hits.append(
                replace(
                    hit_map[chunk_id],
                    score=round(fused_scores[chunk_id], 6),
                )
            )
        return fused_hits

    def _rrf_score(self, rank: int) -> float:
        """计算 Reciprocal Rank Fusion 分值。

        参数:
            rank: 某条结果在对应检索通道中的排名，从 1 开始。

        返回:
            当前排名对应的 RRF 分值。
        """
        return 1.0 / (self._RRF_K + rank)
