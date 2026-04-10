"""chunk 检索服务与抽象。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.domain.knowledge_file_repository import KnowledgeFileRepository
from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.evidence_sufficiency import EvidenceAssessment, EvidenceSufficiencyService
from baozhi_rag.services.fast_rerank import FastRerankService
from baozhi_rag.services.query_intent import QueryIntentService
from baozhi_rag.services.retrieval_pipeline import (
    RetrievalPipelineExecutionResult,
    RetrievalPipelineService,
)
from baozhi_rag.services.retrieval_plan import RetrievalLanePlan, RetrievalPlan, RetrievalPlanner
from baozhi_rag.services.retrieval_trace import RetrievalTrace, RetrievalTraceService
from baozhi_rag.services.term_matching import MaximumMatchingTermMatcher

if TYPE_CHECKING:
    from baozhi_rag.services.document_chunking import ChunkImageAsset, DocumentChunk


def _empty_str_list() -> list[str]:
    """返回空字符串列表。"""
    return []


def _empty_image_asset_list() -> list[ChunkImageAsset]:
    """返回空图片资产列表。"""
    return []


class ChunkSearchValidationError(AppError):
    """检索请求参数非法。"""

    default_message = "检索参数非法"
    default_error_code = "chunk_search_validation_error"
    default_status_code = status.HTTP_400_BAD_REQUEST


@dataclass(frozen=True, slots=True)
class ChunkSearchRequest:
    """chunk 检索请求。"""

    query_text: str
    size: int
    merged_terms: list[str]
    query_embedding: list[float]
    query_intent: str = "general"
    lexical_candidate_size: int | None = None
    vector_candidate_size: int | None = None
    lexical_rrf_weight: float | None = None
    vector_rrf_weight: float | None = None
    viewer_user_id: str = ""


@dataclass(frozen=True, slots=True)
class ChunkSearchHit:
    """chunk 检索命中结果。"""

    chunk_id: str
    file_id: str
    chunk_type: str
    segment_id: str
    source_filename: str
    storage_key: str
    chunk_index: int
    char_count: int
    content: str
    merged_terms: list[str]
    score: float | None
    heading_path: list[str] = field(default_factory=_empty_str_list)
    section_title: str | None = None
    content_type: str = "paragraph"
    image_assets: list[ChunkImageAsset] = field(default_factory=_empty_image_asset_list)
    uploader_user_id: str = ""
    visibility_scope: str = ""


@dataclass(frozen=True, slots=True)
class ChunkSearchExecutionResult:
    """带 trace 的检索执行结果。"""

    hits: list[ChunkSearchHit]
    query_intent: str
    retrieval_trace: RetrievalTrace
    evidence_assessment: EvidenceAssessment


class ChunkSearchStore(Protocol):
    """chunk 混合检索存储抽象。"""

    def ensure_index(self) -> None:
        """确保底层检索索引或集合已经就绪。"""
        ...

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """写入 chunk 文档。"""
        ...

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        """按文件标识删除 chunk。"""
        ...

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        """执行 chunk 检索。"""
        ...


class ChunkSearchService:
    """编排查询词分解、向量化、权限过滤与元数据补齐。"""

    def __init__(
        self,
        term_matcher: MaximumMatchingTermMatcher,
        store: ChunkSearchStore,
        chunk_embedding_service: ChunkEmbeddingService,
        knowledge_file_repository: KnowledgeFileRepository | None = None,
        query_intent_service: QueryIntentService | None = None,
        retrieval_planner: RetrievalPlanner | None = None,
        retrieval_pipeline: RetrievalPipelineService | None = None,
        evidence_sufficiency_service: EvidenceSufficiencyService | None = None,
        retrieval_trace_service: RetrievalTraceService | None = None,
        fast_rerank_service: FastRerankService | None = None,
    ) -> None:
        """初始化检索服务。"""
        self._term_matcher = term_matcher
        self._store = store
        self._chunk_embedding_service = chunk_embedding_service
        self._knowledge_file_repository = knowledge_file_repository
        self._query_intent_service = query_intent_service or QueryIntentService()
        self._retrieval_planner = retrieval_planner
        self._retrieval_pipeline = retrieval_pipeline or RetrievalPipelineService(
            term_matcher=term_matcher,
            store=store,
            chunk_embedding_service=chunk_embedding_service,
        )
        self._evidence_sufficiency_service = (
            evidence_sufficiency_service or EvidenceSufficiencyService()
        )
        self._retrieval_trace_service = retrieval_trace_service or RetrievalTraceService()
        self._fast_rerank_service = fast_rerank_service or FastRerankService()

    def search(
        self,
        query_text: str,
        size: int,
        *,
        viewer_user_id: str = "",
        retrieval_mode: str = "search",
    ) -> list[ChunkSearchHit]:
        """执行基于 ES 词法与 Milvus 语义的混合检索。"""
        normalized_query = query_text.strip()

        if not normalized_query:
            raise ChunkSearchValidationError("查询文本不能为空")
        if size <= 0:
            raise ChunkSearchValidationError("size 必须大于 0")

        return self.search_with_trace(
            query_text,
            size,
            viewer_user_id=viewer_user_id,
            retrieval_mode=retrieval_mode,
        ).hits

    def search_with_trace(
        self,
        query_text: str,
        size: int,
        *,
        viewer_user_id: str = "",
        retrieval_mode: str = "search",
    ) -> ChunkSearchExecutionResult:
        """执行检索并返回 trace 与证据判断。"""
        normalized_query = query_text.strip()

        if not normalized_query:
            raise ChunkSearchValidationError("查询文本不能为空")
        if size <= 0:
            raise ChunkSearchValidationError("size 必须大于 0")

        query_intent = self._query_intent_service.detect(normalized_query)
        retrieval_plan = self._build_retrieval_plan(
            query_text=normalized_query,
            query_intent=query_intent,
            size=size,
            retrieval_mode=retrieval_mode,
        )
        pipeline_result = self._execute_retrieval_plan(
            retrieval_plan=retrieval_plan,
            viewer_user_id=viewer_user_id,
        )
        hydrated_hits = self._hydrate_file_metadata(pipeline_result.hits)
        reranked_hits = self._fast_rerank_service.rerank(
            query_text=normalized_query,
            query_intent=query_intent,
            hits=hydrated_hits,
        )
        evidence_assessment = self._evidence_sufficiency_service.assess(reranked_hits)
        retrieval_trace = self._retrieval_trace_service.build(
            execution_result=pipeline_result,
            final_hits=reranked_hits,
            evidence_assessment=evidence_assessment,
        )
        return ChunkSearchExecutionResult(
            hits=reranked_hits,
            query_intent=query_intent,
            retrieval_trace=retrieval_trace,
            evidence_assessment=evidence_assessment,
        )

    def _build_retrieval_plan(
        self,
        *,
        query_text: str,
        query_intent: str,
        size: int,
        retrieval_mode: str,
    ) -> RetrievalPlan:
        """构造当前查询的检索计划。"""
        if self._retrieval_planner is None:
            return RetrievalPlan(
                mode=retrieval_mode,
                query_text=query_text,
                query_intent=query_intent,
                result_size=size,
                lanes=[
                    RetrievalLanePlan(
                        lane_id="hybrid-default",
                        query_text=query_text,
                        lane_weight=1.0,
                        result_size=size,
                        lexical_candidate_size=size,
                        vector_candidate_size=size,
                        query_intent=query_intent,
                    )
                ],
            )

        return self._retrieval_planner.build(
            query_text=query_text,
            query_intent=query_intent,
            result_size=size,
            mode=retrieval_mode,
        )

    def _execute_retrieval_plan(
        self,
        *,
        retrieval_plan: RetrievalPlan,
        viewer_user_id: str,
    ) -> RetrievalPipelineExecutionResult:
        """执行多 lane 检索并做全局融合。"""
        return self._retrieval_pipeline.execute(
            retrieval_plan=retrieval_plan,
            viewer_user_id=viewer_user_id,
        )

    def _hydrate_file_metadata(self, hits: list[ChunkSearchHit]) -> list[ChunkSearchHit]:
        """用数据库中的最新文件元数据覆盖索引内的旧标题。"""
        if not hits or self._knowledge_file_repository is None:
            return hits

        file_ids = list(dict.fromkeys(hit.file_id for hit in hits))
        knowledge_files = self._knowledge_file_repository.get_files_by_ids(file_ids)
        file_map = {knowledge_file.id: knowledge_file for knowledge_file in knowledge_files}

        hydrated_hits: list[ChunkSearchHit] = []
        for hit in hits:
            knowledge_file = file_map.get(hit.file_id)
            if knowledge_file is None:
                continue
            hydrated_hits.append(
                replace(
                    hit,
                    source_filename=knowledge_file.original_filename,
                    storage_key=knowledge_file.storage_key,
                    uploader_user_id=knowledge_file.uploader_user_id,
                    visibility_scope=knowledge_file.visibility_scope.value,
                )
            )
        return hydrated_hits
