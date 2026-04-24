"""chunk 检索服务与抽象。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.domain.knowledge_file import FileVisibilityScope
from baozhi_rag.domain.knowledge_file_repository import KnowledgeFileRepository
from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.evidence_sufficiency import EvidenceAssessment, EvidenceSufficiencyService
from baozhi_rag.services.fast_rerank import FastRerankService
from baozhi_rag.services.query_intent import QueryIntentService
from baozhi_rag.services.retrieval_pipeline import (
    RetrievalPipelineExecutionResult,
    RetrievalPipelineService,
)
from baozhi_rag.services.retrieval_plan import (
    RetrievalLanePlan,
    RetrievalLaneSeed,
    RetrievalPlan,
    RetrievalPlanner,
)
from baozhi_rag.services.retrieval_signals import (
    QueryFileAnchor,
    belongs_to_query_file_family,
    extract_query_file_anchor,
    extract_query_terms,
    matches_query_file_anchor,
)
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
    retrieval_mode: str = "search"
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
    page_number: int | None
    source_anchor: str | None
    chunk_index: int
    char_count: int
    content: str
    merged_terms: list[str]
    score: float | None
    heading_path: list[str] = field(default_factory=_empty_str_list)
    section_title: str | None = None
    content_type: str = "paragraph"
    file_content_type: str | None = None
    file_extension: str | None = None
    file_size: int | None = None
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

    _NEGATIVE_EVIDENCE_QUERY_MARKERS = (
        "有没有",
        "是否",
        "有无",
        "是否允许",
        "是否明确",
        "有没有写明",
        "有没有规定",
        "是否要求",
        "是否写了",
        "有没有提到",
    )
    _GUIDANCE_QUERY_MARKERS = (
        "证据不足时",
        "怎么说",
        "如何表述",
        "表述要求",
        "边界条件",
        "边界规则",
        "下一条边界规则",
    )
    _NEGATIVE_QUERY_STOP_TERMS = {
        "有没有",
        "是否",
        "有无",
        "允许",
        "明确",
        "写明",
        "规定",
        "要求",
        "提及",
        "内容",
        "文档",
        "材料",
        "知识库",
        "规则",
        "相关",
        "当前",
        "现有",
        "所有",
        "必须",
        "需要",
        "文件名",
    }

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
        hydrated_pipeline_result = self._hydrate_pipeline_execution_result(pipeline_result)
        visible_pipeline_result = self._filter_pipeline_execution_result_for_viewer(
            execution_result=hydrated_pipeline_result,
            viewer_user_id=viewer_user_id,
        )
        reranked_hits = self._fast_rerank_service.rerank(
            query_text=normalized_query,
            query_intent=query_intent,
            hits=visible_pipeline_result.hits,
        )
        query_file_anchor = extract_query_file_anchor(normalized_query)
        anchored_execution_result, reranked_hits, anchor_reason_code = (
            self._apply_query_file_anchor_constraint(
                query_file_anchor=query_file_anchor,
                hydrated_execution_result=hydrated_pipeline_result,
                visible_execution_result=visible_pipeline_result,
                reranked_hits=reranked_hits,
            )
        )
        if self._should_suppress_negative_query_hits(
            query_text=normalized_query,
            hits=reranked_hits,
        ):
            reranked_hits = []
            anchor_reason_code = anchor_reason_code or "no_evidence"
        evidence_assessment = self._evidence_sufficiency_service.assess(
            reranked_hits,
            override_reason_code=anchor_reason_code,
            top_score_override=visible_pipeline_result.hits[0].score
            if visible_pipeline_result.hits
            else None,
        )
        retrieval_trace = self._retrieval_trace_service.build(
            execution_result=anchored_execution_result,
            final_hits=reranked_hits,
            evidence_assessment=evidence_assessment,
        )
        return ChunkSearchExecutionResult(
            hits=reranked_hits,
            query_intent=query_intent,
            retrieval_trace=retrieval_trace,
            evidence_assessment=evidence_assessment,
        )

    def _should_suppress_negative_query_hits(
        self,
        *,
        query_text: str,
        hits: list[ChunkSearchHit],
    ) -> bool:
        """对“有没有/是否”类问题压制不具备直接支撑能力的弱相关命中。"""
        if not hits:
            return False
        if self._query_is_guidance_or_boundary_request(query_text):
            return False
        if not self._query_requests_negative_evidence_judgement(query_text):
            return False
        return not self._hits_provide_direct_negative_query_support(
            query_text=query_text,
            hits=hits,
        )

    def _query_requests_negative_evidence_judgement(self, query_text: str) -> bool:
        """判断查询是否在确认“文档中是否存在某项明确要求”。"""
        normalized_query = " ".join(query_text.split())
        return any(marker in normalized_query for marker in self._NEGATIVE_EVIDENCE_QUERY_MARKERS)

    def _query_is_guidance_or_boundary_request(self, query_text: str) -> bool:
        """判断查询是否是在询问治理口径/边界规则。"""
        normalized_query = " ".join(query_text.split())
        return any(marker in normalized_query for marker in self._GUIDANCE_QUERY_MARKERS)

    def _hits_provide_direct_negative_query_support(
        self,
        *,
        query_text: str,
        hits: list[ChunkSearchHit],
    ) -> bool:
        """判断命中结果是否真正覆盖了负样本查询里的核心约束词。"""
        normalized_query = " ".join(query_text.split())
        query_file_anchor = extract_query_file_anchor(normalized_query)
        query_without_anchor = (
            normalized_query.replace(query_file_anchor.raw_filename, " ")
            if query_file_anchor is not None
            else normalized_query
        )
        focus_terms = [
            term.casefold()
            for term in extract_query_terms(query_without_anchor)
            if len(term.strip()) >= 2 and term.casefold() not in self._NEGATIVE_QUERY_STOP_TERMS
        ]
        if not focus_terms:
            return False

        hit_text = " ".join(
            " ".join(
                filter(
                    None,
                    [
                        hit.source_filename,
                        " ".join(hit.heading_path),
                        hit.section_title or "",
                        hit.content,
                    ],
                )
            ).casefold()
            for hit in hits[:5]
        )
        matched_terms = {term for term in focus_terms if term in hit_text}
        return matched_terms.issuperset(focus_terms)

    def _apply_query_file_anchor_constraint(
        self,
        *,
        query_file_anchor: QueryFileAnchor | None,
        hydrated_execution_result: RetrievalPipelineExecutionResult,
        visible_execution_result: RetrievalPipelineExecutionResult,
        reranked_hits: list[ChunkSearchHit],
    ) -> tuple[RetrievalPipelineExecutionResult, list[ChunkSearchHit], str | None]:
        """对显式点名文件的查询执行最终文件约束，阻止无关文件兜底作答。"""
        if query_file_anchor is None:
            return visible_execution_result, reranked_hits, None

        anchored_hits = self._filter_hits_for_query_file_anchor(
            hits=reranked_hits,
            query_file_anchor=query_file_anchor,
        )
        if anchored_hits:
            anchored_execution_result = self._filter_execution_result_for_query_file_anchor(
                execution_result=visible_execution_result,
                query_file_anchor=query_file_anchor,
            )
            return anchored_execution_result, anchored_hits, None

        anchored_execution_result = self._filter_execution_result_for_query_file_anchor(
            execution_result=visible_execution_result,
            query_file_anchor=query_file_anchor,
        )
        return (
            anchored_execution_result,
            [],
            self._resolve_query_file_anchor_miss_reason(
                query_file_anchor=query_file_anchor,
                hydrated_hits=hydrated_execution_result.hits,
                visible_hits=visible_execution_result.hits,
            ),
        )

    def _filter_execution_result_for_query_file_anchor(
        self,
        *,
        execution_result: RetrievalPipelineExecutionResult,
        query_file_anchor: QueryFileAnchor,
    ) -> RetrievalPipelineExecutionResult:
        """同步过滤全局命中和各 lane 命中，保持 trace 与最终结果一致。"""
        return replace(
            execution_result,
            lane_executions=[
                replace(
                    lane_execution,
                    hits=self._filter_hits_for_query_file_anchor(
                        hits=lane_execution.hits,
                        query_file_anchor=query_file_anchor,
                    ),
                )
                for lane_execution in execution_result.lane_executions
            ],
            hits=self._filter_hits_for_query_file_anchor(
                hits=execution_result.hits,
                query_file_anchor=query_file_anchor,
            ),
        )

    def _filter_hits_for_query_file_anchor(
        self,
        *,
        hits: list[ChunkSearchHit],
        query_file_anchor: QueryFileAnchor,
    ) -> list[ChunkSearchHit]:
        """只保留满足显式目标文件锚点的命中结果。"""
        return [
            hit for hit in hits if matches_query_file_anchor(query_file_anchor, hit.source_filename)
        ]

    def _resolve_query_file_anchor_miss_reason(
        self,
        *,
        query_file_anchor: QueryFileAnchor,
        hydrated_hits: list[ChunkSearchHit],
        visible_hits: list[ChunkSearchHit],
    ) -> str:
        """区分“目标文件不可见”和“版本/文件锚点不匹配”两类失败原因。"""
        if any(
            matches_query_file_anchor(query_file_anchor, hit.source_filename)
            for hit in hydrated_hits
        ):
            return "target_file_inaccessible"

        if query_file_anchor.version_rank is not None and any(
            belongs_to_query_file_family(query_file_anchor, hit.source_filename)
            for hit in visible_hits
        ):
            return "target_file_version_mismatch"

        return "target_file_mismatch"

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
            return self._build_default_retrieval_plan(
                query_text=query_text,
                query_intent=query_intent,
                size=size,
                retrieval_mode=retrieval_mode,
            )

        default_plan = self._retrieval_planner.build(
            query_text=query_text,
            query_intent=query_intent,
            result_size=size,
            mode=retrieval_mode,
        )
        anchor_lane_seeds = self._build_query_file_anchor_lane_seeds(query_text=query_text)
        if not anchor_lane_seeds:
            return default_plan

        # 统一扩展检索框架由 planner 内部接管，ChunkSearchService 仅负责把
        # query/query_intent/mode/size 传入，避免在服务层硬编码 MQE/HyDE 细节。
        anchor_plan = self._retrieval_planner.build(
            query_text=query_text,
            query_intent=query_intent,
            result_size=size,
            mode=retrieval_mode,
            lane_seeds=anchor_lane_seeds,
        )
        return replace(
            default_plan,
            lanes=default_plan.lanes + anchor_plan.lanes,
        )

    def _build_default_retrieval_plan(
        self,
        *,
        query_text: str,
        query_intent: str,
        size: int,
        retrieval_mode: str,
    ) -> RetrievalPlan:
        """构造兼容兜底的单 lane 检索计划。"""
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

    def _build_query_file_anchor_lane_seeds(
        self,
        *,
        query_text: str,
    ) -> list[RetrievalLaneSeed] | None:
        """为显式点名文件的查询补充 filename-focused 检索 lane。"""
        query_file_anchor = extract_query_file_anchor(query_text)
        if query_file_anchor is None:
            return None

        lane_seeds = [
            RetrievalLaneSeed(
                lane_id="filename-anchor",
                lexical_query_text=query_file_anchor.raw_filename,
                embedding_source_text=query_file_anchor.raw_filename,
                source_strategy="filename_anchor",
                source_query=query_text,
                strategy_label="filename_anchor",
                lane_weight=1.15,
                lexical_candidate_size=30,
                vector_candidate_size=0,
                lexical_rrf_weight=1.4,
                vector_rrf_weight=0.0,
                query_intent="document_location",
            )
        ]
        if query_file_anchor.normalized_title:
            lane_seeds.append(
                RetrievalLaneSeed(
                    lane_id="title-anchor",
                    lexical_query_text=query_file_anchor.normalized_title,
                    embedding_source_text=query_file_anchor.normalized_title,
                    source_strategy="title_anchor",
                    source_query=query_text,
                    strategy_label="title_anchor",
                    lane_weight=1.0,
                    lexical_candidate_size=24,
                    vector_candidate_size=0,
                    lexical_rrf_weight=1.2,
                    vector_rrf_weight=0.0,
                    query_intent="document_location",
                )
            )
        return lane_seeds

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
                    file_content_type=knowledge_file.content_type,
                    file_extension=self._infer_file_extension(knowledge_file.original_filename),
                    file_size=knowledge_file.size,
                    uploader_user_id=knowledge_file.uploader_user_id,
                    visibility_scope=knowledge_file.visibility_scope.value,
                )
            )
        return hydrated_hits

    def _hydrate_pipeline_execution_result(
        self,
        execution_result: RetrievalPipelineExecutionResult,
    ) -> RetrievalPipelineExecutionResult:
        """把 pipeline 结果中的全局命中与 lane 命中统一回填最新文件元数据。"""
        return replace(
            execution_result,
            lane_executions=[
                replace(
                    lane_execution,
                    hits=self._hydrate_file_metadata(lane_execution.hits),
                )
                for lane_execution in execution_result.lane_executions
            ],
            hits=self._hydrate_file_metadata(execution_result.hits),
        )

    def _filter_pipeline_execution_result_for_viewer(
        self,
        *,
        execution_result: RetrievalPipelineExecutionResult,
        viewer_user_id: str,
    ) -> RetrievalPipelineExecutionResult:
        """对 pipeline 结果和各 lane 命中同步执行服务层权限兜底过滤。"""
        if not viewer_user_id:
            return execution_result

        return replace(
            execution_result,
            lane_executions=[
                replace(
                    lane_execution,
                    hits=self._filter_hits_for_viewer(
                        hits=lane_execution.hits,
                        viewer_user_id=viewer_user_id,
                    ),
                )
                for lane_execution in execution_result.lane_executions
            ],
            hits=self._filter_hits_for_viewer(
                hits=execution_result.hits,
                viewer_user_id=viewer_user_id,
            ),
        )

    def _filter_hits_for_viewer(
        self,
        *,
        hits: list[ChunkSearchHit],
        viewer_user_id: str,
    ) -> list[ChunkSearchHit]:
        """在服务层再次执行权限过滤，兜底底层检索侧的过滤漂移。"""
        if not hits or not viewer_user_id:
            return hits

        visible_hits: list[ChunkSearchHit] = []
        for hit in hits:
            if hit.visibility_scope == FileVisibilityScope.GLOBAL.value:
                visible_hits.append(hit)
                continue
            if hit.uploader_user_id == viewer_user_id:
                visible_hits.append(hit)
        return visible_hits

    def _infer_file_extension(self, filename: str) -> str | None:
        """从原始文件名中提取扩展名，供聊天链路展示使用。"""
        normalized_name = filename.rsplit("/", maxsplit=1)[-1].strip()
        if "." not in normalized_name:
            return None
        extension = normalized_name.rsplit(".", maxsplit=1)[-1].strip().lower()
        return extension or None
