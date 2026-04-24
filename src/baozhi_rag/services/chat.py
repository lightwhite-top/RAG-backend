"""RAG 聊天编排服务。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime
from http import HTTPStatus
from typing import Protocol

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.chunk_search import ChunkSearchExecutionResult, ChunkSearchHit
from baozhi_rag.services.context_packing import ContextPackingBudgetConfig, ContextPackingService
from baozhi_rag.services.deep_rerank import DeepRerankService
from baozhi_rag.services.document_chunking import ChunkImageAsset
from baozhi_rag.services.evidence_sufficiency import EvidenceAssessment
from baozhi_rag.services.llm import ChatMessage, ChatModelClient
from baozhi_rag.services.query_rewrite import QueryRewriteService
from baozhi_rag.services.rerank import ImageRerankService
from baozhi_rag.services.retrieval_trace import RetrievalTrace


def _empty_image_asset_list() -> list[ChunkImageAsset]:
    """返回空图片资产列表。"""
    return []


def _empty_block_asset_list() -> list[ChatBlockAsset]:
    """返回空正文块资产列表。"""
    return []


@dataclass(frozen=True, slots=True)
class ChatBlockAsset:
    """结构化正文块统一使用的资产投影。"""

    asset_id: str
    display_name: str
    storage_key: str
    content_type: str | None = None
    extension: str | None = None
    size: int | None = None
    preview_storage_key: str | None = None
    source_anchor: str | None = None
    summary: str | None = None
    ocr_text: str | None = None


class ChatCompletionValidationError(AppError):
    """聊天请求参数非法。"""

    default_message = "聊天请求参数非法"
    default_error_code = "chat_completion_validation_error"
    default_status_code = int(HTTPStatus.BAD_REQUEST)


@dataclass(frozen=True, slots=True)
class ChatCitation:
    """聊天回答引用的证据片段。"""

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
    citation_id: str = ""
    snippet: str = ""
    heading_path: list[str] = field(default_factory=list)
    section_title: str | None = None
    content_type: str = "paragraph"
    page_number: int | None = None
    source_anchor: str | None = None
    file_content_type: str | None = None
    extension: str | None = None
    size: int | None = None
    image_assets: list[ChunkImageAsset] = field(default_factory=_empty_image_asset_list)


@dataclass(frozen=True, slots=True)
class ChatContentBlock:
    """聊天回答的结构化正文块。"""

    block_id: str
    block_type: str
    text: str
    citation_ids: list[str]
    sequence: int
    files_assets: list[ChatBlockAsset] = field(default_factory=_empty_block_asset_list)


@dataclass(frozen=True, slots=True)
class ChatCompletionResult:
    """单次聊天补全结果。"""

    answer: str
    retrieval_query: str
    citations: list[ChatCitation]
    finish_reason: str
    plain_text: str = ""
    content_blocks: list[ChatContentBlock] = field(default_factory=list)
    original_query: str = ""
    rewrite_applied: bool = False
    query_intent: str = "general"
    retrieval_trace: RetrievalTrace | None = None
    evidence_assessment: EvidenceAssessment | None = None
    message_id: str | None = None
    session_id: str | None = None
    sequence_no: int | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    applied_retrieval_size: int | None = None
    applied_temperature: float | None = None


@dataclass(frozen=True, slots=True)
class ChatStreamEvent:
    """聊天流式事件。"""

    event: str
    data: dict[str, object]


@dataclass(frozen=True, slots=True)
class ChatRuntimePolicy:
    """聊天补全链路的后端生效策略。"""

    retrieval_size: int
    temperature: float | None


class ChatChunkSearcher(Protocol):
    """聊天服务依赖的检索协议。"""

    def search(
        self,
        query_text: str,
        size: int,
        *,
        viewer_user_id: str = "",
        retrieval_mode: str = "search",
    ) -> list[ChunkSearchHit]:
        """按查询文本返回相关 chunk。"""
        ...

    def search_with_trace(
        self,
        query_text: str,
        size: int,
        *,
        viewer_user_id: str = "",
        retrieval_mode: str = "search",
    ) -> ChunkSearchExecutionResult:
        """按查询文本返回带 trace 的相关 chunk。"""
        ...


@dataclass(frozen=True, slots=True)
class _PreparedChatCompletion:
    """聊天补全前的预处理结果。"""

    original_query: str
    retrieval_query: str
    citations: list[ChatCitation]
    model_messages: list[ChatMessage]
    rewrite_applied: bool = False
    query_intent: str = "general"
    retrieval_trace: RetrievalTrace | None = None
    evidence_assessment: EvidenceAssessment | None = None
    runtime_policy: ChatRuntimePolicy | None = None


class ChatService:
    """负责检索增强、风控提示和模型对话编排。"""

    _CITATION_PATTERN = re.compile(r"\[(\d+)\]")
    _BLOCK_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
    _EXCESSIVE_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
    _ASCII_CJK_BOUNDARY_PATTERN = re.compile(r"(?P<ascii>[A-Z]{1,4})(?P<cjk>[\u4e00-\u9fff])")
    _MAX_CONTEXT_CHARS = 1200
    _MAX_SNIPPET_CHARS = 180
    _DEFAULT_RETRIEVAL_SIZE = 5
    _FOLLOWUP_RETRIEVAL_SIZE = 7
    _MAX_RETRIEVAL_SIZE = 10
    _DEFAULT_TEMPERATURE = 0.0
    _DEFAULT_MAX_PROMPT_TOKENS = 4096
    _DEFAULT_RESERVED_ANSWER_TOKENS = 768
    _DEFAULT_RESERVED_MARGIN_TOKENS = 256
    _NO_EVIDENCE_ANSWER_MARKERS = (
        "未提及",
        "未检索到",
        "未规定",
        "未写明",
        "未明确",
        "没有规定",
        "无法给出明确结论",
        "无法提供对应值",
        "暂时不能直接给出确定答复",
        "没有检索到可支撑结论的知识库材料",
        "未找到可支撑",
    )
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
    _DIRECT_FALLBACK_REASON_CODES = frozenset(
        {
            "no_evidence",
            "low_score",
            "context_budget_exhausted",
            "target_file_inaccessible",
            "target_file_mismatch",
            "target_file_version_mismatch",
        }
    )
    _FALLBACK_ANSWER = (
        "当前知识库中未检索到足以支撑结论的材料，暂时不能直接给出确定答复。"
        "建议补充问题细节、上传相关文档，或转人工进一步核实。"
    )
    _GUIDANCE_CANONICAL_NOTICE = (
        "处理此类问题时，应明确说明当前依据有限、需要补充材料，不能编造超出文档证据的结论。"
    )
    _BOUNDARY_CANONICAL_NOTICE = "处理前需先确认适用对象、版本、时效边界和材料完整度。"

    def __init__(
        self,
        chat_client: ChatModelClient,
        chunk_search_service: ChatChunkSearcher,
        system_prompt: str,
        deep_rerank_service: DeepRerankService | None = None,
        image_rerank_service: ImageRerankService | None = None,
        query_rewrite_service: QueryRewriteService | None = None,
        context_packing_service: ContextPackingService | None = None,
    ) -> None:
        """初始化聊天服务。

        参数:
            chat_client: 负责调用底层聊天模型的客户端。
            chunk_search_service: 负责执行检索增强的 chunk 检索服务。
            system_prompt: 传给模型的基础系统提示词，用于约束高风险知识问答场景回答。

        返回:
            None。
        """
        self._chat_client = chat_client
        self._chunk_search_service = chunk_search_service
        self._system_prompt = system_prompt
        self._deep_rerank_service = deep_rerank_service
        self._image_rerank_service = image_rerank_service
        self._query_rewrite_service = query_rewrite_service or QueryRewriteService()
        self._context_packing_service = context_packing_service or ContextPackingService(
            max_context_chars=self._MAX_CONTEXT_CHARS
        )

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        retrieval_size: int,
        temperature: float | None = None,
        viewer_user_id: str = "",
    ) -> ChatCompletionResult:
        """执行一次带检索增强的非流式聊天补全。

        参数:
            messages: 当前会话消息列表，至少需要包含一条 user 消息。
            retrieval_size: 本次检索需要召回的 chunk 数量。
            temperature: 可选采样温度。

        返回:
            包含最终回答、检索查询和证据列表的聊天结果。

        异常:
            ChatCompletionValidationError: 当消息列表或检索参数不合法时抛出。
        """
        completion = self._prepare_completion(
            messages,
            retrieval_size,
            requested_temperature=temperature,
            viewer_user_id=viewer_user_id,
        )
        runtime_policy = completion.runtime_policy or ChatRuntimePolicy(
            retrieval_size=self._DEFAULT_RETRIEVAL_SIZE,
            temperature=self._DEFAULT_TEMPERATURE,
        )
        finish_reason = "stop"
        guidance_or_boundary_query = self._query_is_guidance_or_boundary_request(
            completion.original_query
        )
        if self._should_force_conservative_answer(completion):
            # 证据不足时直接走确定性兜底，避免模型在无证据场景下继续补充想象性内容。
            answer = self._FALLBACK_ANSWER
            finish_reason = "evidence_insufficient"
        else:
            answer = self._chat_client.complete_chat(
                completion.model_messages,
                temperature=runtime_policy.temperature,
            ).strip()
            if not answer:
                answer = self._FALLBACK_ANSWER
                if not completion.citations:
                    # 即使已经放开“无命中仍调用模型”，也要为模型空输出保留明确兜底。
                    finish_reason = "context_exhausted"
            elif self._has_insufficient_evidence(completion):
                finish_reason = "evidence_insufficient"
        answer = self._append_guidance_notice_if_needed(
            answer=answer,
            original_query=completion.original_query,
        )
        answer = self._normalize_answer_spacing(answer)
        if guidance_or_boundary_query:
            response_evidence_assessment = completion.evidence_assessment
            render_citations = completion.citations if finish_reason == "stop" else []
        else:
            response_evidence_assessment = self._resolve_output_evidence_assessment(
                answer=answer,
                citations=completion.citations,
                evidence_assessment=completion.evidence_assessment,
                original_query=completion.original_query,
            )
            render_citations = self._resolve_response_citations(
                citations=completion.citations,
                finish_reason=finish_reason,
                evidence_assessment=response_evidence_assessment,
            )
        plain_text, content_blocks = self._build_render_content(
            answer,
            render_citations,
            original_query=completion.original_query,
            retrieval_query=completion.retrieval_query,
            finish_reason=finish_reason,
        )

        return ChatCompletionResult(
            answer=answer,
            plain_text=plain_text,
            content_blocks=content_blocks,
            original_query=completion.original_query,
            retrieval_query=completion.retrieval_query,
            citations=render_citations,
            finish_reason=finish_reason,
            rewrite_applied=completion.rewrite_applied,
            query_intent=completion.query_intent,
            retrieval_trace=completion.retrieval_trace,
            evidence_assessment=response_evidence_assessment,
            applied_retrieval_size=runtime_policy.retrieval_size,
            applied_temperature=runtime_policy.temperature,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        retrieval_size: int,
        temperature: float | None = None,
        viewer_user_id: str = "",
    ) -> Iterator[ChatStreamEvent]:
        """执行一次带检索增强的流式聊天补全。

        参数:
            messages: 当前会话消息列表，至少需要包含一条 user 消息。
            retrieval_size: 本次检索需要召回的 chunk 数量。
            temperature: 可选采样温度。

        返回:
            先返回检索上下文，再持续返回模型文本增量，最后返回完成事件。

        异常:
            ChatCompletionValidationError: 当消息列表或检索参数不合法时抛出。
        """
        completion = self._prepare_completion(
            messages,
            retrieval_size,
            requested_temperature=temperature,
            viewer_user_id=viewer_user_id,
        )
        runtime_policy = completion.runtime_policy or ChatRuntimePolicy(
            retrieval_size=self._DEFAULT_RETRIEVAL_SIZE,
            temperature=self._DEFAULT_TEMPERATURE,
        )
        guidance_or_boundary_query = self._query_is_guidance_or_boundary_request(
            completion.original_query
        )
        forced_fallback = self._should_force_conservative_answer(completion)
        if guidance_or_boundary_query:
            response_citations = completion.citations if not forced_fallback else []
        else:
            response_citations = self._resolve_response_citations(
                citations=completion.citations,
                finish_reason="evidence_insufficient" if forced_fallback else "stop",
                evidence_assessment=completion.evidence_assessment,
            )
        citations_payload = [self._serialize_citation(item) for item in response_citations]

        # 先把检索上下文透出给调用方，便于前端同步展示证据和审计信息。
        yield ChatStreamEvent(
            event="context",
            data={
                "original_query": completion.original_query,
                "retrieval_query": completion.retrieval_query,
                "rewrite_applied": completion.rewrite_applied,
                "query_intent": completion.query_intent,
                "retrieval_trace": self._serialize_retrieval_trace(completion.retrieval_trace),
                "evidence_assessment": self._serialize_evidence_assessment(
                    completion.evidence_assessment
                ),
                "citations": citations_payload,
                "applied_retrieval_size": runtime_policy.retrieval_size,
                "applied_temperature": runtime_policy.temperature,
            },
        )

        if forced_fallback:
            plain_text, content_blocks = self._build_render_content(
                self._FALLBACK_ANSWER,
                response_citations,
                original_query=completion.original_query,
                retrieval_query=completion.retrieval_query,
                finish_reason="evidence_insufficient",
            )
            yield ChatStreamEvent(
                event="delta",
                data={"content": self._FALLBACK_ANSWER},
            )
            yield ChatStreamEvent(
                event="done",
                data={
                    "answer": self._FALLBACK_ANSWER,
                    "plain_text": plain_text,
                    "content_blocks": [
                        self._serialize_content_block(item) for item in content_blocks
                    ],
                    "original_query": completion.original_query,
                    "retrieval_query": completion.retrieval_query,
                    "citations": citations_payload,
                    "finish_reason": "evidence_insufficient",
                    "rewrite_applied": completion.rewrite_applied,
                    "query_intent": completion.query_intent,
                    "retrieval_trace": self._serialize_retrieval_trace(completion.retrieval_trace),
                    "evidence_assessment": self._serialize_evidence_assessment(
                        completion.evidence_assessment
                    ),
                    "applied_retrieval_size": runtime_policy.retrieval_size,
                    "applied_temperature": runtime_policy.temperature,
                },
            )
            return

        answer_parts: list[str] = []
        pending_buffer = ""
        current_markdown_index = 1
        current_block_sequence = 1
        next_visible_sequence = 1
        current_block_anchor_id: str | None = None
        current_block_emitted_text = ""
        for delta in self._chat_client.stream_chat(
            completion.model_messages,
            temperature=runtime_policy.temperature,
        ):
            if not delta:
                continue
            answer_parts.append(delta)
            pending_buffer += delta
            completed_raw_blocks, pending_buffer = self._extract_completed_stream_blocks(
                pending_buffer
            )
            for raw_block in completed_raw_blocks:
                markdown_block, _ = self._build_markdown_block_from_raw_block(
                    raw_block,
                    completion.citations,
                    block_id=f"blk-{current_markdown_index}",
                    sequence=current_block_sequence,
                )
                if markdown_block is None:
                    current_block_emitted_text = ""
                    continue

                for append_event in self._build_stream_append_events(
                    markdown_block=markdown_block,
                    emitted_text=current_block_emitted_text,
                    after_block_id=current_block_anchor_id,
                ):
                    yield append_event
                current_block_emitted_text = markdown_block.text

                inserted_blocks = self._build_support_blocks_for_markdown_block(
                    markdown_block=markdown_block,
                    citations=completion.citations,
                    original_query=completion.original_query,
                    retrieval_query=completion.retrieval_query,
                )
                for insert_event in self._build_stream_insert_events(
                    markdown_block=markdown_block,
                    inserted_blocks=inserted_blocks,
                ):
                    yield insert_event
                next_visible_sequence += 1 + len(inserted_blocks)
                current_markdown_index += 1
                current_block_sequence = next_visible_sequence
                current_block_anchor_id = (
                    inserted_blocks[-1].block_id if inserted_blocks else markdown_block.block_id
                )
                current_block_emitted_text = ""

            partial_markdown_block = self._build_partial_stream_markdown_block(
                raw_block=pending_buffer,
                citations=completion.citations,
                block_id=f"blk-{current_markdown_index}",
                sequence=current_block_sequence,
            )
            if partial_markdown_block is None:
                continue
            for append_event in self._build_stream_append_events(
                markdown_block=partial_markdown_block,
                emitted_text=current_block_emitted_text,
                after_block_id=current_block_anchor_id,
            ):
                yield append_event
            current_block_emitted_text = partial_markdown_block.text

        answer = "".join(answer_parts).strip()
        finish_reason = "stop"
        if not answer:
            answer = self._FALLBACK_ANSWER
            if not completion.citations:
                finish_reason = "context_exhausted"
        elif (
            completion.evidence_assessment is not None
            and not completion.evidence_assessment.sufficient
        ):
            finish_reason = "evidence_insufficient"
        answer = self._append_guidance_notice_if_needed(
            answer=answer,
            original_query=completion.original_query,
        )
        answer = self._normalize_answer_spacing(answer)
        final_pending_markdown = self._build_markdown_block_from_raw_block(
            pending_buffer,
            completion.citations,
            block_id=f"blk-{current_markdown_index}",
            sequence=current_block_sequence,
        )[0]
        if final_pending_markdown is not None:
            for append_event in self._build_stream_append_events(
                markdown_block=final_pending_markdown,
                emitted_text=current_block_emitted_text,
                after_block_id=current_block_anchor_id,
            ):
                yield append_event
            for insert_event in self._build_stream_insert_events(
                markdown_block=final_pending_markdown,
                inserted_blocks=self._build_support_blocks_for_markdown_block(
                    markdown_block=final_pending_markdown,
                    citations=completion.citations,
                    original_query=completion.original_query,
                    retrieval_query=completion.retrieval_query,
                ),
            ):
                yield insert_event
        plain_text, content_blocks = self._build_render_content(
            answer,
            completion.citations,
            original_query=completion.original_query,
            retrieval_query=completion.retrieval_query,
            finish_reason=finish_reason,
        )
        if guidance_or_boundary_query:
            response_evidence_assessment = completion.evidence_assessment
            response_citations = completion.citations if finish_reason == "stop" else []
        else:
            response_evidence_assessment = self._resolve_output_evidence_assessment(
                answer=answer,
                citations=completion.citations,
                evidence_assessment=completion.evidence_assessment,
                original_query=completion.original_query,
            )
            response_citations = self._resolve_response_citations(
                citations=completion.citations,
                finish_reason=finish_reason,
                evidence_assessment=response_evidence_assessment,
            )
        if response_citations != completion.citations:
            plain_text, content_blocks = self._build_render_content(
                answer,
                response_citations,
                original_query=completion.original_query,
                retrieval_query=completion.retrieval_query,
                finish_reason=finish_reason,
            )
        citations_payload = [self._serialize_citation(item) for item in response_citations]
        yield ChatStreamEvent(
            event="done",
            data={
                "answer": answer,
                "plain_text": plain_text,
                "content_blocks": [self._serialize_content_block(item) for item in content_blocks],
                "original_query": completion.original_query,
                "retrieval_query": completion.retrieval_query,
                "citations": citations_payload,
                "finish_reason": finish_reason,
                "rewrite_applied": completion.rewrite_applied,
                "query_intent": completion.query_intent,
                "retrieval_trace": self._serialize_retrieval_trace(completion.retrieval_trace),
                "evidence_assessment": self._serialize_evidence_assessment(
                    response_evidence_assessment
                ),
                "applied_retrieval_size": runtime_policy.retrieval_size,
                "applied_temperature": runtime_policy.temperature,
            },
        )

    def _prepare_completion(
        self,
        messages: list[ChatMessage],
        retrieval_size: int,
        *,
        requested_temperature: float | None = None,
        viewer_user_id: str = "",
    ) -> _PreparedChatCompletion:
        """完成消息校验、检索和模型提示构造。"""
        if not messages:
            msg = "messages 不能为空"
            raise ChatCompletionValidationError(msg)
        if retrieval_size <= 0:
            msg = "retrieval_size 必须大于 0"
            raise ChatCompletionValidationError(msg)

        normalized_messages = self._normalize_messages(messages)
        runtime_policy = self._resolve_runtime_policy(
            normalized_messages,
            requested_retrieval_size=retrieval_size,
            requested_temperature=requested_temperature,
        )
        # 当前最稳定的检索查询是最后一条用户追问，而不是整段会话拼接文本。
        original_query = self._resolve_retrieval_query(normalized_messages)
        rewrite_result = self._query_rewrite_service.rewrite(
            normalized_messages,
            original_query=original_query,
        )
        retrieval_query = rewrite_result.rewritten_query
        rewrite_applied = rewrite_result.rewrite_applied
        search_result = self._chunk_search_service.search_with_trace(
            retrieval_query,
            runtime_policy.retrieval_size,
            viewer_user_id=viewer_user_id,
            retrieval_mode="chat",
        )
        hits = search_result.hits
        deep_rerank_triggered = False
        if self._deep_rerank_service is not None:
            hits, deep_rerank_triggered = self._deep_rerank_service.rerank(
                query_text=retrieval_query,
                hits=hits,
            )
        preliminary_citations = [
            self._build_citation(hit, index=index) for index, hit in enumerate(hits, start=1)
        ]
        context_prompt_result = self._context_packing_service.build_context_prompt_result(
            retrieval_query=retrieval_query,
            citations=preliminary_citations,
            budget_config=self._build_context_budget_config(),
            history_messages=normalized_messages,
        )
        included_citation_ids = set(context_prompt_result.included_citation_ids)
        citations = [
            citation
            for citation in preliminary_citations
            if citation.citation_id in included_citation_ids
        ]
        evidence_assessment = search_result.evidence_assessment
        context_prompt = context_prompt_result.prompt
        if preliminary_citations and not citations:
            evidence_assessment = EvidenceAssessment(
                sufficient=False,
                reason_code="context_budget_exhausted",
                citation_count=0,
                top_score=search_result.evidence_assessment.top_score
                if search_result.evidence_assessment is not None
                else None,
            )
            context_prompt = self._context_packing_service.build_no_knowledge_context_prompt(
                retrieval_query=retrieval_query
            )
        retrieval_trace = search_result.retrieval_trace
        if retrieval_trace is not None and deep_rerank_triggered:
            retrieval_trace = replace(
                retrieval_trace,
                deep_rerank_triggered=True,
                final_hit_count=len(hits),
            )

        return _PreparedChatCompletion(
            original_query=original_query,
            retrieval_query=retrieval_query,
            citations=citations,
            model_messages=self._build_model_messages(
                messages=normalized_messages,
                context_prompt=context_prompt,
                evidence_assessment=evidence_assessment,
            ),
            rewrite_applied=rewrite_applied,
            query_intent=search_result.query_intent,
            retrieval_trace=retrieval_trace,
            evidence_assessment=evidence_assessment,
            runtime_policy=runtime_policy,
        )

    def _should_force_conservative_answer(
        self,
        completion: _PreparedChatCompletion,
    ) -> bool:
        """判断当前是否必须跳过模型直答，直接返回保守兜底。

        只有在“完全无证据”或“命中分过低”这类明确无法支撑回答的场景下，
        才直接走固定兜底，避免把证据冲突、上下文预算不足等可解释场景也
        误判成“完全没有材料”。
        """
        if not self._has_insufficient_evidence(completion):
            return False

        assessment = completion.evidence_assessment
        if assessment is None:
            return False
        return assessment.reason_code in self._DIRECT_FALLBACK_REASON_CODES

    def _has_insufficient_evidence(
        self,
        completion: _PreparedChatCompletion,
    ) -> bool:
        """判断当前检索结果是否处于证据不足状态。"""
        return bool(
            completion.evidence_assessment is not None
            and not completion.evidence_assessment.sufficient
        )

    def _resolve_response_citations(
        self,
        *,
        citations: list[ChatCitation],
        finish_reason: str,
        evidence_assessment: EvidenceAssessment | None,
    ) -> list[ChatCitation]:
        """根据完成原因决定是否继续向调用方暴露引用。"""
        if finish_reason != "stop":
            return []
        if evidence_assessment is not None and not evidence_assessment.sufficient:
            return []
        return citations

    def _resolve_output_evidence_assessment(
        self,
        *,
        answer: str,
        citations: list[ChatCitation],
        evidence_assessment: EvidenceAssessment | None,
        original_query: str,
    ) -> EvidenceAssessment | None:
        """当模型明确给出“未检索到/未提及”类结论时，回写为证据不足。"""
        if (
            not citations
            or self._query_is_guidance_or_boundary_request(original_query)
            or not self._answer_indicates_missing_evidence(answer)
            or not self._query_requests_negative_evidence_judgement(original_query)
        ):
            return evidence_assessment
        top_score = evidence_assessment.top_score if evidence_assessment is not None else None
        return EvidenceAssessment(
            sufficient=False,
            reason_code="answer_indicates_no_evidence",
            citation_count=0,
            top_score=top_score,
        )

    def _answer_indicates_missing_evidence(self, answer: str) -> bool:
        """判断模型最终回答是否明确表达了“当前缺少支撑证据”。"""
        normalized_answer = " ".join(answer.split())
        return any(marker in normalized_answer for marker in self._NO_EVIDENCE_ANSWER_MARKERS)

    def _query_requests_negative_evidence_judgement(self, query_text: str) -> bool:
        """判断问题是否是在确认“文档里有没有/是否存在”某项结论。"""
        normalized_query = " ".join(query_text.split())
        return any(marker in normalized_query for marker in self._NEGATIVE_EVIDENCE_QUERY_MARKERS)

    def _query_is_guidance_or_boundary_request(self, query_text: str) -> bool:
        """判断问题是否在询问治理口径或边界规则，而非真假存在性判断。"""
        normalized_query = " ".join(query_text.split())
        return any(marker in normalized_query for marker in self._GUIDANCE_QUERY_MARKERS)

    def _append_guidance_notice_if_needed(self, *, answer: str, original_query: str) -> str:
        """为治理/边界类问题补齐稳定的保守口径。"""
        if not self._query_is_guidance_or_boundary_request(original_query):
            return answer
        normalized_answer = answer
        notices: list[str] = []
        if (
            "依据" not in normalized_answer
            and "补充材料" not in normalized_answer
            and "不能编造" not in normalized_answer
        ):
            notices.append(self._GUIDANCE_CANONICAL_NOTICE)
        if "边界条件" in original_query and "适用对象" not in normalized_answer:
            notices.append(self._BOUNDARY_CANONICAL_NOTICE)
        if not notices:
            return answer
        return f"{answer}\n\n" + "\n".join(notices)

    def _normalize_answer_spacing(self, answer: str) -> str:
        """规整英文缩写与中文之间的黏连，降低字符串比对误差。"""
        return self._ASCII_CJK_BOUNDARY_PATTERN.sub(r"\g<ascii> \g<cjk>", answer)

    def _resolve_runtime_policy(
        self,
        messages: list[ChatMessage],
        *,
        requested_retrieval_size: int,
        requested_temperature: float | None,
    ) -> ChatRuntimePolicy:
        """根据后端策略决定本次聊天补全的实际检索与采样参数。"""
        del requested_temperature

        user_message_count = sum(1 for message in messages if message.role == "user")
        # 多轮追问通常更依赖跨轮上下文，因此给出更高的默认召回；若调用方显式
        # 请求更大的召回窗口，则在后端安全上限内放大，避免关键证据被过早裁掉。
        baseline_retrieval_size = (
            self._FOLLOWUP_RETRIEVAL_SIZE
            if user_message_count > 1
            else self._DEFAULT_RETRIEVAL_SIZE
        )
        retrieval_size = min(
            self._MAX_RETRIEVAL_SIZE,
            max(baseline_retrieval_size, requested_retrieval_size),
        )
        # RAG 问答默认走低温采样，避免前端通过温度扰动事实型回答的一致性。
        return ChatRuntimePolicy(
            retrieval_size=retrieval_size,
            temperature=self._DEFAULT_TEMPERATURE,
        )

    def _build_context_budget_config(self) -> ContextPackingBudgetConfig:
        """构造证据上下文预算配置。"""
        occupied_prompt_tokens = self._context_packing_service.estimate_token_count(
            self._build_system_prompt()
        ) + self._context_packing_service.estimate_token_count(self._build_render_contract_prompt())
        return ContextPackingBudgetConfig(
            max_prompt_tokens=self._DEFAULT_MAX_PROMPT_TOKENS,
            occupied_prompt_tokens=occupied_prompt_tokens,
            reserved_answer_tokens=self._DEFAULT_RESERVED_ANSWER_TOKENS,
            reserved_margin_tokens=self._DEFAULT_RESERVED_MARGIN_TOKENS,
            max_evidence_tokens=720,
            min_evidence_tokens=120,
        )

    def _normalize_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """清洗消息内容，避免把空白消息传给模型。"""
        normalized_messages: list[ChatMessage] = []
        for message in messages:
            content = message.content.strip()
            if not content:
                continue
            normalized_messages.append(ChatMessage(role=message.role, content=content))

        if not normalized_messages:
            msg = "messages 不能为空"
            raise ChatCompletionValidationError(msg)
        return normalized_messages

    def _resolve_retrieval_query(self, messages: list[ChatMessage]) -> str:
        """提取最后一条用户消息作为检索查询。"""
        for message in reversed(messages):
            if message.role == "user":
                return message.content

        msg = "至少需要一条 user 消息"
        raise ChatCompletionValidationError(msg)

    def _build_model_messages(
        self,
        *,
        messages: list[ChatMessage],
        context_prompt: str,
        evidence_assessment: EvidenceAssessment | None = None,
    ) -> list[ChatMessage]:
        """构造传给聊天模型的消息列表。"""
        prompt_messages = [
            ChatMessage(role="system", content=self._build_system_prompt()),
            ChatMessage(
                role="system",
                content=self._build_render_contract_prompt(),
            ),
            ChatMessage(
                role="system",
                # 把召回证据前置为 system 消息，尽量降低后续多轮对话对证据约束的稀释。
                content=context_prompt,
            ),
        ]
        evidence_prompt = self._context_packing_service.build_evidence_guidance_prompt(
            evidence_assessment=evidence_assessment,
        )
        if evidence_prompt:
            prompt_messages.append(ChatMessage(role="system", content=evidence_prompt))
        prompt_messages.extend(messages)
        return prompt_messages

    def _build_system_prompt(self) -> str:
        """返回配置注入的风控系统提示词。"""
        return self._system_prompt

    def _build_render_contract_prompt(self) -> str:
        """约束模型直接输出更稳定的 Markdown 正文。"""
        return "\n".join(
            [
                "输出格式要求：",
                "1. 正文请直接输出 Markdown，不要输出 HTML。",
                "2. 标题、列表、表格、引用段落请使用标准 Markdown 语法。",
                "3. 不要输出 JSON、字段名或额外解释。",
                "4. 保留文中的证据编号格式，例如 [1][2]。",
                "5. 除非内容确实需要代码示例，否则不要使用代码块围栏。",
            ]
        )

    def _build_citation(self, hit: ChunkSearchHit, *, index: int) -> ChatCitation:
        """把检索命中结果转换为聊天引用对象。"""
        return ChatCitation(
            citation_id=f"cit-{index}",
            chunk_id=hit.chunk_id,
            file_id=hit.file_id,
            chunk_type=hit.chunk_type,
            segment_id=hit.segment_id,
            source_filename=hit.source_filename,
            storage_key=hit.storage_key,
            chunk_index=hit.chunk_index,
            char_count=hit.char_count,
            content=hit.content,
            merged_terms=hit.merged_terms,
            score=hit.score,
            snippet=self._build_snippet(hit.content),
            heading_path=list(hit.heading_path),
            section_title=hit.section_title,
            content_type="table" if hit.content_type == "table" else "paragraph",
            page_number=hit.page_number,
            source_anchor=hit.source_anchor or f"chunk:{hit.chunk_index}",
            file_content_type=hit.file_content_type,
            extension=hit.file_extension or self._infer_file_extension(hit.source_filename),
            size=hit.file_size,
            image_assets=hit.image_assets,
        )

    def _serialize_citation(self, citation: ChatCitation) -> dict[str, object]:
        """把引用对象转换为可序列化结构。"""
        return {
            "id": citation.citation_id,
            "chunk_id": citation.chunk_id,
            "file_id": citation.file_id,
            "chunk_type": citation.chunk_type,
            "segment_id": citation.segment_id,
            "source_filename": citation.source_filename,
            "storage_key": citation.storage_key,
            "chunk_index": citation.chunk_index,
            "char_count": citation.char_count,
            "content": citation.content,
            "snippet": citation.snippet,
            "merged_terms": citation.merged_terms,
            "score": citation.score,
            "heading_path": citation.heading_path,
            "section_title": citation.section_title,
            "content_type": citation.content_type,
            "page_number": citation.page_number,
            "source_anchor": citation.source_anchor,
            "file_content_type": citation.file_content_type,
            "extension": citation.extension,
            "size": citation.size,
            "image_assets": [self._serialize_image_asset(item) for item in citation.image_assets],
        }

    def _serialize_content_block(self, block: ChatContentBlock) -> dict[str, object]:
        """把结构化正文块转换为可序列化结构。"""
        return {
            "block_id": block.block_id,
            "block_type": block.block_type,
            "text": block.text,
            "citation_ids": block.citation_ids,
            "sequence": block.sequence,
            "files_assets": [self._serialize_block_asset(item) for item in block.files_assets],
        }

    def _serialize_image_asset(self, asset: ChunkImageAsset) -> dict[str, object]:
        """把图片资产转换为可序列化结构。"""
        return {
            "segment_id": asset.segment_id,
            "asset_id": asset.asset_id,
            "source_anchor": asset.source_anchor,
            "storage_key": asset.storage_key,
            "thumbnail_storage_key": asset.thumbnail_storage_key,
            "image_type": asset.image_type,
            "summary": asset.summary,
            "ocr_text": asset.ocr_text,
        }

    def _serialize_block_asset(self, asset: ChatBlockAsset) -> dict[str, object]:
        """把正文块统一资产转换为可序列化结构。"""
        return {
            "asset_id": asset.asset_id,
            "display_name": asset.display_name,
            "storage_key": asset.storage_key,
            "content_type": asset.content_type,
            "extension": asset.extension,
            "size": asset.size,
            "preview_storage_key": asset.preview_storage_key,
            "source_anchor": asset.source_anchor,
            "summary": asset.summary,
            "ocr_text": asset.ocr_text,
        }

    def _serialize_evidence_assessment(
        self,
        assessment: EvidenceAssessment | None,
    ) -> dict[str, object] | None:
        """把证据充分性判断结果转换为可序列化结构。"""
        if assessment is None:
            return None
        return {
            "sufficient": assessment.sufficient,
            "reason_code": assessment.reason_code,
            "citation_count": assessment.citation_count,
            "top_score": assessment.top_score,
        }

    def _serialize_retrieval_trace(
        self,
        retrieval_trace: RetrievalTrace | None,
    ) -> dict[str, object] | None:
        """把检索 trace 转换为可序列化结构。"""
        if retrieval_trace is None:
            return None
        # retrieval_trace 可能随检索框架演进新增字段，这里统一按 dataclass 展开后透传。
        if is_dataclass(retrieval_trace):
            serialized_trace = asdict(retrieval_trace)
        else:
            serialized_trace = {
                "mode": retrieval_trace.mode,
                "query_intent": retrieval_trace.query_intent,
                "lane_count": retrieval_trace.lane_count,
                "final_hit_count": retrieval_trace.final_hit_count,
                "evidence_sufficient": retrieval_trace.evidence_sufficient,
                "evidence_reason": retrieval_trace.evidence_reason,
                "deep_rerank_triggered": retrieval_trace.deep_rerank_triggered,
                "lanes": [
                    {
                        "lane_id": lane.lane_id,
                        "query_text": lane.query_text,
                        "lane_weight": lane.lane_weight,
                        "result_count": lane.result_count,
                        "top_chunk_ids": list(lane.top_chunk_ids),
                    }
                    for lane in retrieval_trace.lanes
                ],
            }
        return self._sanitize_serialized_retrieval_trace(serialized_trace)

    def _sanitize_serialized_retrieval_trace(
        self,
        serialized_trace: dict[str, object],
    ) -> dict[str, object]:
        """清理公开 trace 中不应直接暴露的内部检索细节。"""
        raw_lanes = serialized_trace.get("lanes")
        if not isinstance(raw_lanes, list):
            serialized_trace["lanes"] = []
            return serialized_trace

        sanitized_lanes: list[object] = []
        for raw_lane in raw_lanes:
            if not isinstance(raw_lane, dict):
                sanitized_lanes.append(raw_lane)
                continue

            sanitized_lane = dict(raw_lane)
            raw_embedding_summary = sanitized_lane.pop("embedding_source_summary", None)
            sanitized_lane["embedding_source_changed"] = bool(raw_embedding_summary)
            sanitized_lanes.append(sanitized_lane)

        serialized_trace["lanes"] = sanitized_lanes
        return serialized_trace

    def _build_snippet(self, content: str) -> str:
        """为引用卡片构造简短摘要。"""
        normalized = " ".join(content.split())
        if len(normalized) <= self._MAX_SNIPPET_CHARS:
            return normalized
        return f"{normalized[: self._MAX_SNIPPET_CHARS]}..."

    def _infer_file_extension(self, filename: str) -> str | None:
        """从文件名中提取稳定扩展名。"""
        normalized_name = filename.rsplit("/", maxsplit=1)[-1].strip()
        if "." not in normalized_name:
            return None
        extension = normalized_name.rsplit(".", maxsplit=1)[-1].strip().lower()
        return extension or None

    def _build_render_content(
        self,
        answer: str,
        citations: list[ChatCitation],
        *,
        original_query: str,
        retrieval_query: str,
        finish_reason: str,
    ) -> tuple[str, list[ChatContentBlock]]:
        """把模型回答解析为可渲染正文块，并完成引用编号校验。"""
        cleaned_answer = answer.strip()
        if not cleaned_answer:
            return "", []

        if finish_reason != "stop":
            normalized_notice = self._normalize_markdown_text(cleaned_answer)
            return cleaned_answer, [
                ChatContentBlock(
                    block_id="blk-1",
                    block_type="notice",
                    text=normalized_notice,
                    citation_ids=[],
                    sequence=1,
                    files_assets=[],
                )
            ]

        if not citations:
            return self._build_uncited_render_content(cleaned_answer)

        raw_blocks = [
            block.strip()
            for block in self._BLOCK_SPLIT_PATTERN.split(cleaned_answer)
            if block.strip()
        ]
        markdown_blocks: list[ChatContentBlock] = []
        has_valid_reference = False

        for index, raw_block in enumerate(raw_blocks, start=1):
            markdown_block, has_reference = self._build_markdown_block_from_raw_block(
                raw_block,
                citations,
                block_id=f"blk-{index}",
                sequence=index,
            )
            if has_reference:
                has_valid_reference = True
            if markdown_block is None:
                continue
            markdown_blocks.append(markdown_block)

        if not markdown_blocks:
            stripped_answer = self._strip_citation_markers(cleaned_answer).strip() or cleaned_answer
            normalized_block_text = self._normalize_markdown_text(stripped_answer)
            return stripped_answer, [
                ChatContentBlock(
                    block_id="blk-1",
                    block_type="markdown",
                    text=normalized_block_text,
                    citation_ids=[],
                    sequence=1,
                    files_assets=[],
                )
            ]

        # 当模型未显式输出合法引用编号且仅命中一条证据时，把该证据附给全部正文块，
        # 避免前端无法建立正文与证据之间的最小联动关系。
        if not has_valid_reference and len(citations) == 1:
            only_citation_id = citations[0].citation_id
            markdown_blocks = [
                replace(
                    block,
                    citation_ids=[only_citation_id],
                )
                for block in markdown_blocks
            ]
        parsed_blocks = self._expand_markdown_blocks(
            markdown_blocks=markdown_blocks,
            citations=citations,
            original_query=original_query,
            retrieval_query=retrieval_query,
        )

        plain_text = self._build_plain_text_from_raw_blocks(raw_blocks)
        return plain_text or cleaned_answer, parsed_blocks

    def _build_uncited_render_content(
        self,
        answer: str,
    ) -> tuple[str, list[ChatContentBlock]]:
        """在没有引用证据时，仍按正常正文块输出模型回答。"""
        raw_blocks = [
            block.strip() for block in self._BLOCK_SPLIT_PATTERN.split(answer) if block.strip()
        ]
        parsed_blocks: list[ChatContentBlock] = []

        for sequence, raw_block in enumerate(raw_blocks, start=1):
            block_text = self._normalize_markdown_text(
                self._strip_citation_markers(raw_block).strip()
            )
            if not block_text:
                continue
            parsed_blocks.append(
                ChatContentBlock(
                    block_id=f"blk-{sequence}",
                    block_type="markdown",
                    text=block_text,
                    citation_ids=[],
                    sequence=sequence,
                    files_assets=[],
                )
            )

        if not parsed_blocks:
            stripped_answer = self._strip_citation_markers(answer).strip() or answer
            normalized_block_text = self._normalize_markdown_text(stripped_answer)
            return stripped_answer, [
                ChatContentBlock(
                    block_id="blk-1",
                    block_type="markdown",
                    text=normalized_block_text,
                    citation_ids=[],
                    sequence=1,
                    files_assets=[],
                )
            ]

        plain_text = self._build_plain_text_from_raw_blocks(raw_blocks)
        return plain_text or answer, parsed_blocks

    def _build_markdown_block_from_raw_block(
        self,
        raw_block: str,
        citations: list[ChatCitation],
        *,
        block_id: str,
        sequence: int,
    ) -> tuple[ChatContentBlock | None, bool]:
        """把单个已闭合正文块转换为 Markdown 正文块。"""
        citation_ids = self._resolve_block_citation_ids(raw_block, citations)
        has_valid_reference = bool(citation_ids)
        if not citation_ids and len(citations) == 1:
            citation_ids = [citations[0].citation_id]
        block_text = self._normalize_markdown_text(self._strip_citation_markers(raw_block).strip())
        if not block_text:
            return None, has_valid_reference

        return (
            ChatContentBlock(
                block_id=block_id,
                block_type="markdown",
                text=block_text,
                citation_ids=citation_ids,
                sequence=sequence,
                files_assets=[],
            ),
            has_valid_reference,
        )

    def _build_partial_stream_markdown_block(
        self,
        *,
        raw_block: str,
        citations: list[ChatCitation],
        block_id: str,
        sequence: int,
    ) -> ChatContentBlock | None:
        """从未闭合的流式正文块中提取当前可安全展示的 Markdown 文本。"""
        display_text = self._extract_stream_display_text(raw_block)
        if not display_text:
            return None

        citation_ids = self._resolve_block_citation_ids(raw_block, citations)
        if not citation_ids and len(citations) == 1:
            citation_ids = [citations[0].citation_id]
        return ChatContentBlock(
            block_id=block_id,
            block_type="markdown",
            text=display_text,
            citation_ids=citation_ids,
            sequence=sequence,
            files_assets=[],
        )

    def _extract_stream_display_text(self, raw_block: str) -> str:
        """剔除流式尾部未闭合的引用编号，避免前端出现回退式替换。"""
        if not raw_block.strip():
            return ""

        normalized_block = raw_block.replace("\r\n", "\n").replace("\r", "\n")
        trailing_marker_match = re.search(r"\[\d*$", normalized_block)
        if trailing_marker_match is not None:
            normalized_block = normalized_block[: trailing_marker_match.start()]
        elif normalized_block.endswith("["):
            normalized_block = normalized_block[:-1]
        return self._normalize_markdown_text(
            self._strip_citation_markers(normalized_block).rstrip()
        )

    def _build_stream_append_events(
        self,
        *,
        markdown_block: ChatContentBlock,
        emitted_text: str,
        after_block_id: str | None,
    ) -> list[ChatStreamEvent]:
        """构造 Markdown 增量追加事件。"""
        if not markdown_block.text.startswith(emitted_text):
            return []

        appended_text = markdown_block.text[len(emitted_text) :]
        if not appended_text:
            return []

        delta_block = ChatContentBlock(
            block_id=markdown_block.block_id,
            block_type="markdown",
            text=appended_text,
            citation_ids=list(markdown_block.citation_ids),
            sequence=markdown_block.sequence,
            files_assets=[],
        )
        return [
            ChatStreamEvent(
                event="delta",
                data={
                    "delta_type": "append",
                    "offset": len(emitted_text),
                    "after_block_id": after_block_id if not emitted_text else None,
                    "block": self._serialize_content_block(delta_block),
                },
            )
        ]

    def _build_stream_insert_events(
        self,
        *,
        markdown_block: ChatContentBlock,
        inserted_blocks: list[ChatContentBlock],
    ) -> list[ChatStreamEvent]:
        """构造图片块与文件块的插入事件。"""
        events: list[ChatStreamEvent] = []
        after_block_id = markdown_block.block_id
        for block in inserted_blocks:
            events.append(
                ChatStreamEvent(
                    event="delta",
                    data={
                        "delta_type": "insert",
                        "after_block_id": after_block_id,
                        "block": self._serialize_content_block(block),
                    },
                )
            )
            after_block_id = block.block_id
        return events

    def _expand_markdown_blocks(
        self,
        *,
        markdown_blocks: list[ChatContentBlock],
        citations: list[ChatCitation],
        original_query: str,
        retrieval_query: str,
    ) -> list[ChatContentBlock]:
        """把 Markdown 正文块扩展为最终可渲染的正文块列表。"""
        final_blocks: list[ChatContentBlock] = []
        next_sequence = 1
        for markdown_block in markdown_blocks:
            normalized_markdown_block = replace(
                markdown_block,
                sequence=next_sequence,
                files_assets=[],
            )
            final_blocks.append(normalized_markdown_block)
            next_sequence += 1
            support_blocks = self._build_support_blocks_for_markdown_block(
                markdown_block=normalized_markdown_block,
                citations=citations,
                original_query=original_query,
                retrieval_query=retrieval_query,
            )
            for block in support_blocks:
                final_blocks.append(block)
                next_sequence = block.sequence + 1
        return final_blocks

    def _build_support_blocks_for_markdown_block(
        self,
        *,
        markdown_block: ChatContentBlock,
        citations: list[ChatCitation],
        original_query: str,
        retrieval_query: str,
    ) -> list[ChatContentBlock]:
        """为 Markdown 正文块补充图片块和文件块。"""
        if markdown_block.block_type != "markdown":
            return []

        support_blocks: list[ChatContentBlock] = []
        next_sequence = markdown_block.sequence + 1
        image_assets = self._collect_image_assets_for_citation_ids(
            markdown_block.citation_ids, citations
        )
        if image_assets:
            image_assets = self._rerank_images_for_markdown_block(
                markdown_block=markdown_block,
                citations=citations,
                original_query=original_query,
                retrieval_query=retrieval_query,
                image_assets=image_assets,
            )
            support_blocks.append(
                ChatContentBlock(
                    block_id=f"{markdown_block.block_id}-imgs",
                    block_type="image_gallery",
                    text="",
                    citation_ids=list(markdown_block.citation_ids),
                    sequence=next_sequence,
                    files_assets=[
                        self._build_block_asset_from_image(item) for item in image_assets
                    ],
                )
            )
            next_sequence += 1

        file_assets = self._collect_file_assets_for_citation_ids(
            markdown_block.citation_ids, citations
        )
        if file_assets:
            support_blocks.append(
                ChatContentBlock(
                    block_id=f"{markdown_block.block_id}-files",
                    block_type="source_file",
                    text="",
                    citation_ids=list(markdown_block.citation_ids),
                    sequence=next_sequence,
                    files_assets=file_assets,
                )
            )
        return support_blocks

    def _normalize_markdown_text(self, text: str) -> str:
        """对模型直出的 Markdown 做最小必要规整，尽量保持原貌。"""
        stripped_text = text.strip()
        if not stripped_text:
            return ""

        normalized_newlines = stripped_text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in normalized_newlines.split("\n")]
        normalized_text = "\n".join(lines).strip()
        normalized_text = self._EXCESSIVE_BLANK_LINE_PATTERN.sub("\n\n", normalized_text)
        return normalized_text

    def _build_plain_text_from_raw_blocks(self, raw_blocks: list[str]) -> str:
        """从原始正文块构造纯文本返回值。"""
        plain_blocks: list[str] = []
        for raw_block in raw_blocks:
            normalized_block = self._strip_citation_markers(raw_block).strip()
            if not normalized_block:
                continue
            plain_blocks.append(normalized_block)
        return "\n\n".join(plain_blocks).strip()

    def _collect_image_assets_for_citation_ids(
        self,
        citation_ids: list[str],
        citations: list[ChatCitation],
    ) -> list[ChunkImageAsset]:
        """汇总正文块关联引用中的图片资产。"""
        image_assets: list[ChunkImageAsset] = []
        seen_asset_ids: set[str] = set()

        for citation in citations:
            if citation.citation_id not in citation_ids:
                continue
            for image_asset in citation.image_assets:
                unique_key = (
                    f"{image_asset.segment_id}::{image_asset.asset_id}"
                    if image_asset.segment_id
                    else image_asset.asset_id or image_asset.source_anchor
                )
                if unique_key in seen_asset_ids:
                    continue
                seen_asset_ids.add(unique_key)
                image_assets.append(image_asset)

        return image_assets

    def _collect_file_assets_for_citation_ids(
        self,
        citation_ids: list[str],
        citations: list[ChatCitation],
    ) -> list[ChatBlockAsset]:
        """汇总正文块关联引用中的文件资产。"""
        file_assets: list[ChatBlockAsset] = []
        seen_file_ids: set[str] = set()

        for citation in citations:
            if citation.citation_id not in citation_ids:
                continue
            if not citation.file_id or citation.file_id in seen_file_ids:
                continue
            seen_file_ids.add(citation.file_id)
            file_assets.append(self._build_block_asset_from_citation(citation))
        return file_assets

    def _build_block_asset_from_image(self, image_asset: ChunkImageAsset) -> ChatBlockAsset:
        """把检索图片资产转换为统一正文块资产。"""
        return ChatBlockAsset(
            asset_id=image_asset.asset_id,
            display_name=image_asset.image_type or image_asset.asset_id,
            storage_key=image_asset.storage_key,
            content_type=image_asset.content_type or None,
            extension=(image_asset.extension or "").removeprefix(".") or None,
            size=len(image_asset.image_bytes) if image_asset.image_bytes is not None else None,
            preview_storage_key=image_asset.thumbnail_storage_key,
            source_anchor=image_asset.source_anchor,
            summary=image_asset.summary or None,
            ocr_text=image_asset.ocr_text or None,
        )

    def _build_block_asset_from_citation(self, citation: ChatCitation) -> ChatBlockAsset:
        """把引用关联的源文件转换为统一正文块资产。"""
        return ChatBlockAsset(
            asset_id=citation.file_id,
            display_name=citation.source_filename,
            storage_key=citation.storage_key,
            content_type=citation.file_content_type,
            extension=citation.extension or self._infer_file_extension(citation.source_filename),
            size=citation.size,
            preview_storage_key=None,
            source_anchor=None,
            summary=None,
            ocr_text=None,
        )

    def _rerank_images_for_markdown_block(
        self,
        *,
        markdown_block: ChatContentBlock,
        citations: list[ChatCitation],
        original_query: str,
        retrieval_query: str,
        image_assets: list[ChunkImageAsset],
    ) -> list[ChunkImageAsset]:
        """按当前正文块上下文对图片资产做二次排序。"""
        if self._image_rerank_service is None:
            return image_assets

        context_sections = [
            citation.snippet or citation.content
            for citation in citations
            if citation.citation_id in markdown_block.citation_ids
        ]
        return self._image_rerank_service.rerank(
            user_query=original_query,
            retrieval_query=retrieval_query,
            block_text=markdown_block.text,
            context_sections=context_sections,
            image_assets=list(image_assets),
        )

    def _extract_completed_stream_blocks(self, buffer: str) -> tuple[list[str], str]:
        """从流式缓冲区中提取已闭合的正文块。"""
        completed_blocks: list[str] = []
        last_end = 0
        for match in self._BLOCK_SPLIT_PATTERN.finditer(buffer):
            raw_block = buffer[last_end : match.start()].strip()
            if raw_block:
                completed_blocks.append(raw_block)
            last_end = match.end()
        return completed_blocks, buffer[last_end:]

    def _resolve_block_citation_ids(
        self,
        block_text: str,
        citations: list[ChatCitation],
    ) -> list[str]:
        """从正文块中提取并校验引用编号。"""
        resolved_ids: list[str] = []
        seen_ids: set[str] = set()

        for raw_index in self._CITATION_PATTERN.findall(block_text):
            citation_position = int(raw_index)
            if citation_position <= 0 or citation_position > len(citations):
                continue
            citation_id = citations[citation_position - 1].citation_id
            if citation_id in seen_ids:
                continue
            seen_ids.add(citation_id)
            resolved_ids.append(citation_id)

        return resolved_ids

    def _strip_citation_markers(self, text: str) -> str:
        """移除正文中的引用编号标记，保留纯展示文本。"""
        stripped = self._CITATION_PATTERN.sub("", text)
        return re.sub(r"[ \t]{2,}", " ", stripped)
