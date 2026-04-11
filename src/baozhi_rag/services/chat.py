"""RAG 聊天编排服务。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime
from http import HTTPStatus
from typing import Protocol

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.chunk_search import ChunkSearchExecutionResult, ChunkSearchHit
from baozhi_rag.services.context_packing import ContextPackingService
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
    source_anchor: str | None = None
    image_assets: list[ChunkImageAsset] = field(default_factory=_empty_image_asset_list)


@dataclass(frozen=True, slots=True)
class ChatContentBlock:
    """聊天回答的结构化正文块。"""

    block_id: str
    block_type: str
    text: str
    citation_ids: list[str]
    sequence: int
    image_assets: list[ChunkImageAsset] = field(default_factory=_empty_image_asset_list)


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


@dataclass(frozen=True, slots=True)
class ChatStreamEvent:
    """聊天流式事件。"""

    event: str
    data: dict[str, object]


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


class ChatService:
    """负责检索增强、风控提示和模型对话编排。"""

    _CITATION_PATTERN = re.compile(r"\[(\d+)\]")
    _BLOCK_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
    _EXCESSIVE_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
    _MAX_CONTEXT_CHARS = 1200
    _MAX_SNIPPET_CHARS = 180
    _FALLBACK_ANSWER = (
        "当前知识库中未检索到足以支撑结论的材料，暂时不能直接给出确定答复。"
        "建议补充问题细节、上传相关文档，或转人工进一步核实。"
    )

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
            messages, retrieval_size, viewer_user_id=viewer_user_id
        )
        answer = self._chat_client.complete_chat(
            completion.model_messages,
            temperature=temperature,
        ).strip()
        finish_reason = "stop"
        if not answer:
            answer = self._FALLBACK_ANSWER
            if not completion.citations:
                # 即使已经放开“无命中仍调用模型”，也要为模型空输出保留明确兜底。
                finish_reason = "context_exhausted"
        elif (
            completion.evidence_assessment is not None
            and not completion.evidence_assessment.sufficient
        ):
            finish_reason = "evidence_insufficient"
        plain_text, content_blocks = self._build_render_content(
            answer,
            completion.citations,
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
            citations=completion.citations,
            finish_reason=finish_reason,
            rewrite_applied=completion.rewrite_applied,
            query_intent=completion.query_intent,
            retrieval_trace=completion.retrieval_trace,
            evidence_assessment=completion.evidence_assessment,
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
            messages, retrieval_size, viewer_user_id=viewer_user_id
        )
        citations_payload = [self._serialize_citation(item) for item in completion.citations]

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
            },
        )

        answer_parts: list[str] = []
        pending_buffer = ""
        emitted_block_count = 0
        for delta in self._chat_client.stream_chat(
            completion.model_messages,
            temperature=temperature,
        ):
            if not delta:
                continue
            answer_parts.append(delta)
            pending_buffer += delta
            completed_raw_blocks, pending_buffer = self._extract_completed_stream_blocks(
                pending_buffer
            )
            for raw_block in completed_raw_blocks:
                block_candidates, _ = self._build_stream_content_blocks_from_raw_block(
                    raw_block,
                    completion.citations,
                    start_sequence=emitted_block_count + 1,
                )
                content_blocks = self._inject_image_blocks(block_candidates, completion.citations)
                content_blocks = self._rerank_block_images(
                    blocks=content_blocks,
                    citations=completion.citations,
                    original_query=completion.original_query,
                    retrieval_query=completion.retrieval_query,
                )
                for block in content_blocks:
                    emitted_block_count += 1
                    yield ChatStreamEvent(
                        event="delta",
                        data={
                            "delta_type": "content_block",
                            "block": self._serialize_content_block(block),
                        },
                    )

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
        plain_text, content_blocks = self._build_render_content(
            answer,
            completion.citations,
            original_query=completion.original_query,
            retrieval_query=completion.retrieval_query,
            finish_reason=finish_reason,
        )
        for block in content_blocks[emitted_block_count:]:
            yield ChatStreamEvent(
                event="delta",
                data={
                    "delta_type": "content_block",
                    "block": self._serialize_content_block(block),
                },
            )
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
                    completion.evidence_assessment
                ),
            },
        )

    def _prepare_completion(
        self,
        messages: list[ChatMessage],
        retrieval_size: int,
        *,
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
            retrieval_size,
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
        citations = [
            self._build_citation(hit, index=index) for index, hit in enumerate(hits, start=1)
        ]
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
                retrieval_query=retrieval_query,
                citations=citations,
                evidence_assessment=search_result.evidence_assessment,
            ),
            rewrite_applied=rewrite_applied,
            query_intent=search_result.query_intent,
            retrieval_trace=retrieval_trace,
            evidence_assessment=search_result.evidence_assessment,
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
        retrieval_query: str,
        citations: list[ChatCitation],
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
                content=self._context_packing_service.build_context_prompt(
                    retrieval_query=retrieval_query,
                    citations=citations,
                ),
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
            source_anchor=f"chunk:{hit.chunk_index}",
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
            "source_anchor": citation.source_anchor,
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
            "image_assets": [self._serialize_image_asset(item) for item in block.image_assets],
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
        return {
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

    def _build_snippet(self, content: str) -> str:
        """为引用卡片构造简短摘要。"""
        normalized = " ".join(content.split())
        if len(normalized) <= self._MAX_SNIPPET_CHARS:
            return normalized
        return f"{normalized[: self._MAX_SNIPPET_CHARS]}..."

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
                )
            ]

        if not citations:
            return self._build_uncited_render_content(cleaned_answer)

        raw_blocks = [
            block.strip()
            for block in self._BLOCK_SPLIT_PATTERN.split(cleaned_answer)
            if block.strip()
        ]
        parsed_blocks: list[ChatContentBlock] = []
        has_valid_reference = False

        for raw_block in raw_blocks:
            blocks, has_reference = self._build_stream_content_blocks_from_raw_block(
                raw_block,
                citations,
                start_sequence=len(parsed_blocks) + 1,
            )
            if has_reference:
                has_valid_reference = True
            if not blocks:
                continue
            parsed_blocks.extend(blocks)

        if not parsed_blocks:
            stripped_answer = self._strip_citation_markers(cleaned_answer).strip() or cleaned_answer
            normalized_block_text = self._normalize_markdown_text(stripped_answer)
            return stripped_answer, [
                ChatContentBlock(
                    block_id="blk-1",
                    block_type="markdown",
                    text=normalized_block_text,
                    citation_ids=[],
                    sequence=1,
                )
            ]

        # 当模型未显式输出合法引用编号且仅命中一条证据时，把该证据附给全部正文块，
        # 避免前端无法建立正文与证据之间的最小联动关系。
        if not has_valid_reference and len(citations) == 1:
            only_citation_id = citations[0].citation_id
            parsed_blocks = [
                ChatContentBlock(
                    block_id=block.block_id,
                    block_type=block.block_type,
                    text=block.text,
                    citation_ids=[only_citation_id]
                    if block.block_type == "markdown"
                    else block.citation_ids,
                    sequence=block.sequence,
                    image_assets=block.image_assets,
                )
                for block in parsed_blocks
            ]
            parsed_blocks = self._inject_image_blocks(parsed_blocks, citations)
        else:
            parsed_blocks = self._inject_image_blocks(parsed_blocks, citations)
        parsed_blocks = self._rerank_block_images(
            blocks=parsed_blocks,
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
                )
            ]

        plain_text = self._build_plain_text_from_raw_blocks(raw_blocks)
        return plain_text or answer, parsed_blocks

    def _build_stream_content_blocks_from_raw_block(
        self,
        raw_block: str,
        citations: list[ChatCitation],
        *,
        start_sequence: int,
    ) -> tuple[list[ChatContentBlock], bool]:
        """把单个原始正文块转换为结构化文本块。"""
        citation_ids = self._resolve_block_citation_ids(raw_block, citations)
        has_valid_reference = bool(citation_ids)
        if not citation_ids and len(citations) == 1:
            citation_ids = [citations[0].citation_id]
        block_text = self._normalize_markdown_text(self._strip_citation_markers(raw_block).strip())
        if not block_text:
            return [], has_valid_reference

        return (
            [
                ChatContentBlock(
                    block_id=f"blk-{start_sequence}",
                    block_type="markdown",
                    text=block_text,
                    citation_ids=citation_ids,
                    sequence=start_sequence,
                )
            ],
            has_valid_reference,
        )

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

    def _inject_image_blocks(
        self,
        blocks: list[ChatContentBlock],
        citations: list[ChatCitation],
    ) -> list[ChatContentBlock]:
        """在正文块后插入对应的图片画廊块。"""
        if not blocks:
            return []

        final_blocks: list[ChatContentBlock] = []
        next_sequence = 1
        for block in blocks:
            final_blocks.append(
                replace(
                    block,
                    sequence=next_sequence,
                )
            )
            next_sequence += 1
            if block.block_type != "markdown":
                continue
            image_assets = self._collect_image_assets_for_citation_ids(
                block.citation_ids, citations
            )
            if not image_assets:
                continue
            final_blocks.append(
                ChatContentBlock(
                    block_id=f"{block.block_id}-imgs",
                    block_type="image_gallery",
                    text="",
                    citation_ids=list(block.citation_ids),
                    sequence=next_sequence,
                    image_assets=image_assets,
                )
            )
            next_sequence += 1
        return final_blocks

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

    def _rerank_block_images(
        self,
        *,
        blocks: list[ChatContentBlock],
        citations: list[ChatCitation],
        original_query: str,
        retrieval_query: str,
    ) -> list[ChatContentBlock]:
        """按当前正文块上下文对图片块做二次排序。"""
        if self._image_rerank_service is None:
            return blocks

        markdown_blocks_by_id = {
            block.block_id: block for block in blocks if block.block_type == "markdown"
        }
        reranked_blocks: list[ChatContentBlock] = []

        for block in blocks:
            if block.block_type != "image_gallery":
                reranked_blocks.append(block)
                continue

            owner_block = markdown_blocks_by_id.get(block.block_id.removesuffix("-imgs"))
            context_sections = [
                citation.snippet or citation.content
                for citation in citations
                if citation.citation_id in block.citation_ids
            ]
            reranked_assets = self._image_rerank_service.rerank(
                user_query=original_query,
                retrieval_query=retrieval_query,
                block_text=owner_block.text if owner_block is not None else "",
                context_sections=context_sections,
                image_assets=list(block.image_assets),
            )
            reranked_blocks.append(
                ChatContentBlock(
                    block_id=block.block_id,
                    block_type=block.block_type,
                    text=block.text,
                    citation_ids=list(block.citation_ids),
                    sequence=block.sequence,
                    image_assets=reranked_assets,
                )
            )

        return reranked_blocks

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
