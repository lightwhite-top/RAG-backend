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

    def search(
        self,
        query_embedding: list[float],
        size: int,
        *,
        viewer_user_id: str = "",
    ) -> list[MilvusVectorSearchHit]:
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

    _RRF_K = 30

    def __init__(
        self,
        document_store: HybridDocumentStore,
        vector_store: HybridVectorStore,
        *,
        lexical_candidate_size: int,
        vector_candidate_size: int,
        lexical_rrf_weight: float,
        vector_rrf_weight: float,
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
        self._lexical_candidate_size = max(1, lexical_candidate_size)
        self._vector_candidate_size = max(1, vector_candidate_size)
        self._lexical_rrf_weight = max(0.0, lexical_rrf_weight)
        self._vector_rrf_weight = max(0.0, vector_rrf_weight)

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
            lexical_candidate_size=settings.search_lexical_candidate_size,
            vector_candidate_size=settings.search_vector_candidate_size,
            lexical_rrf_weight=settings.search_rrf_lexical_weight,
            vector_rrf_weight=settings.search_rrf_vector_weight,
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
        lexical_weight, vector_weight = self._resolve_rrf_weights(request.query_intent)
        if request.lexical_rrf_weight is not None:
            lexical_weight = request.lexical_rrf_weight
        if request.vector_rrf_weight is not None:
            vector_weight = request.vector_rrf_weight

        lexical_candidate_size = (
            self._lexical_candidate_size
            if request.lexical_candidate_size is None
            else max(0, request.lexical_candidate_size)
        )
        vector_candidate_size = (
            self._vector_candidate_size
            if request.vector_candidate_size is None
            else max(0, request.vector_candidate_size)
        )

        try:
            # 文档检索
            lexical_hits: list[ChunkSearchHit] = []
            if lexical_candidate_size > 0 and lexical_weight > 0:
                lexical_hits = self._document_store.search(
                    replace(
                        request,
                        size=max(request.size, lexical_candidate_size),
                    )
                )
            # 向量检索
            semantic_hits: list[MilvusVectorSearchHit] = []
            if vector_candidate_size > 0 and vector_weight > 0 and request.query_embedding:
                semantic_hits = self._vector_store.search(
                    request.query_embedding,
                    max(request.size, vector_candidate_size),
                    viewer_user_id=request.viewer_user_id,
                )

            # 结果融合：使用 RRF 算法对词法检索结果和向量检索结果进行融合排序，并补全文档载荷
            return self._fuse_hits(
                lexical_hits=lexical_hits,
                semantic_hits=semantic_hits,
                size=request.size,
                query_text=request.query_text,
                query_intent=request.query_intent,
                lexical_weight=lexical_weight,
                vector_weight=vector_weight,
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
        query_text: str,
        query_intent: str,
        lexical_weight: float,
        vector_weight: float,
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
            fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + self._rrf_score(
                rank,
                lexical_weight,
            )

        for rank, semantic_hit in enumerate(semantic_hits, start=1):
            fused_scores[semantic_hit.chunk_id] = fused_scores.get(
                semantic_hit.chunk_id,
                0.0,
            ) + self._rrf_score(
                rank,
                vector_weight,
            )

        semantic_only_ids = [hit.chunk_id for hit in semantic_hits if hit.chunk_id not in hit_map]
        if semantic_only_ids:
            for hit in self._document_store.get_chunks_by_ids(semantic_only_ids):
                hit_map[hit.chunk_id] = hit

        ordered_hits = [
            replace(hit_map[chunk_id], score=round(fused_scores[chunk_id], 6))
            for chunk_id, _ in sorted(
                fused_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if chunk_id in hit_map
        ]
        collapsed_hits = self._collapse_hits(
            hits=ordered_hits,
            query_text=query_text,
        )
        return collapsed_hits[:size]

    def _collapse_hits(
        self,
        *,
        hits: list[ChunkSearchHit],
        query_text: str,
    ) -> list[ChunkSearchHit]:
        """对融合候选做文件级去重，并折叠版本链文档。"""
        deduplicated_by_file: list[ChunkSearchHit] = []
        seen_file_ids: set[str] = set()
        for hit in hits:
            if hit.file_id in seen_file_ids:
                continue
            seen_file_ids.add(hit.file_id)
            deduplicated_by_file.append(hit)

        prefer_old_version = self._query_prefers_old_version(query_text)
        grouped_hits: dict[str, list[ChunkSearchHit]] = {}
        passthrough_hits: list[ChunkSearchHit] = []

        for hit in deduplicated_by_file:
            version_group_key = hit.version_chain_group
            if version_group_key is None:
                passthrough_hits.append(hit)
                continue
            grouped_hits.setdefault(version_group_key, []).append(hit)

        collapsed_version_hits = passthrough_hits + [
            self._select_version_group_representative(
                candidates=candidates,
                prefer_old_version=prefer_old_version,
            )
            for candidates in grouped_hits.values()
        ]
        return sorted(
            collapsed_version_hits,
            key=lambda item: (item.score or 0.0, -item.chunk_index),
            reverse=True,
        )

    @staticmethod
    def _query_prefers_old_version(query_text: str) -> bool:
        """判断当前查询是否明确指向旧版。"""
        normalized_query = query_text.lower()
        return any(token in normalized_query for token in ("旧版", "历史", "v1", "v2"))

    def _select_version_group_representative(
        self,
        *,
        candidates: list[ChunkSearchHit],
        prefer_old_version: bool,
    ) -> ChunkSearchHit:
        """在同一版本链组中选择代表文档。"""
        return max(
            candidates,
            key=lambda item: self._version_selection_key(
                hit=item,
                prefer_old_version=prefer_old_version,
            ),
        )

    def _version_selection_key(
        self,
        *,
        hit: ChunkSearchHit,
        prefer_old_version: bool,
    ) -> tuple[float, float]:
        """构造版本链折叠时的优先级。"""
        version_rank = hit.version_rank
        if prefer_old_version:
            return (-float(version_rank), hit.score or 0.0)
        return (float(version_rank), hit.score or 0.0)

    def _rrf_score(self, rank: int, weight: float) -> float:
        """计算 Reciprocal Rank Fusion 分值。

        参数:
            rank: 某条结果在对应检索通道中的排名，从 1 开始。
            weight: 当前检索通道的融合权重。

        返回:
            当前排名对应的 RRF 分值。
        """
        return weight / (self._RRF_K + rank)

    def _resolve_rrf_weights(self, query_intent: str) -> tuple[float, float]:
        """根据查询意图动态调整词法与向量通道的融合权重。"""
        lexical_weight = self._lexical_rrf_weight
        vector_weight = self._vector_rrf_weight

        if query_intent == "document_location":
            return lexical_weight * 1.25, vector_weight * 0.9
        if query_intent == "definition":
            return lexical_weight * 0.95, vector_weight * 1.15
        if query_intent == "comparison":
            return lexical_weight * 0.95, vector_weight * 1.2
        if query_intent == "procedure":
            return lexical_weight, vector_weight * 1.15
        if query_intent == "structured":
            return lexical_weight * 1.1, vector_weight * 1.05
        return lexical_weight, vector_weight
