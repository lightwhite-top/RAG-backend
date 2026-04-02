"""RAG 聊天编排服务。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.chunk_search import ChunkSearchHit
from baozhi_rag.services.llm import ChatMessage, ChatModelClient


class ChatCompletionValidationError(AppError):
    """聊天请求参数非法。"""

    default_message = "聊天请求参数非法"
    default_error_code = "chat_completion_validation_error"
    default_status_code = status.HTTP_400_BAD_REQUEST


@dataclass(frozen=True, slots=True)
class ChatCitation:
    """聊天回答引用的证据片段。"""

    chunk_id: str
    file_id: str
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


@dataclass(frozen=True, slots=True)
class ChatContentBlock:
    """聊天回答的结构化正文块。"""

    block_id: str
    block_type: str
    text: str
    citation_ids: list[str]
    sequence: int


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


@dataclass(frozen=True, slots=True)
class ChatStreamEvent:
    """聊天流式事件。"""

    event: str
    data: dict[str, object]


class ChatChunkSearcher(Protocol):
    """聊天服务依赖的检索协议。"""

    def search(self, query_text: str, size: int) -> list[ChunkSearchHit]:
        """按查询文本返回相关 chunk。"""
        ...


@dataclass(frozen=True, slots=True)
class _PreparedChatCompletion:
    """聊天补全前的预处理结果。"""

    original_query: str
    retrieval_query: str
    citations: list[ChatCitation]
    model_messages: list[ChatMessage]
    fallback_answer: str | None
    rewrite_applied: bool = False


class ChatService:
    """负责检索增强、风控提示和模型对话编排。"""

    _CITATION_PATTERN = re.compile(r"\[(\d+)\]")
    _BLOCK_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
    _MAX_CONTEXT_CHARS = 1200
    _MAX_SNIPPET_CHARS = 180
    _FALLBACK_ANSWER = (
        "当前知识库中未检索到足以支撑结论的材料，暂时不能直接给出确定答复。"
        "建议补充问题细节、上传相关条款，或转人工进一步核实。"
    )

    def __init__(
        self,
        chat_client: ChatModelClient,
        chunk_search_service: ChatChunkSearcher,
        system_prompt: str,
    ) -> None:
        """初始化聊天服务。

        参数:
            chat_client: 负责调用底层聊天模型的客户端。
            chunk_search_service: 负责执行检索增强的 chunk 检索服务。
            system_prompt: 传给模型的基础系统提示词，用于约束金融保险场景回答。

        返回:
            None。
        """
        self._chat_client = chat_client
        self._chunk_search_service = chunk_search_service
        self._system_prompt = system_prompt

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        retrieval_size: int,
        temperature: float | None = None,
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
        completion = self._prepare_completion(messages, retrieval_size)
        if completion.fallback_answer is not None:
            # 金融保险场景下证据为空时直接兜底，避免模型在无依据时继续生成。
            plain_text, content_blocks = self._build_render_content(
                completion.fallback_answer,
                completion.citations,
                finish_reason="context_exhausted",
            )
            return ChatCompletionResult(
                answer=completion.fallback_answer,
                plain_text=plain_text,
                content_blocks=content_blocks,
                original_query=completion.original_query,
                retrieval_query=completion.retrieval_query,
                citations=completion.citations,
                finish_reason="context_exhausted",
                rewrite_applied=completion.rewrite_applied,
            )

        answer = self._chat_client.complete_chat(
            completion.model_messages,
            temperature=temperature,
        ).strip()
        if not answer:
            answer = self._FALLBACK_ANSWER
        plain_text, content_blocks = self._build_render_content(
            answer,
            completion.citations,
            finish_reason="stop",
        )

        return ChatCompletionResult(
            answer=answer,
            plain_text=plain_text,
            content_blocks=content_blocks,
            original_query=completion.original_query,
            retrieval_query=completion.retrieval_query,
            citations=completion.citations,
            finish_reason="stop",
            rewrite_applied=completion.rewrite_applied,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        retrieval_size: int,
        temperature: float | None = None,
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
        completion = self._prepare_completion(messages, retrieval_size)
        citations_payload = [self._serialize_citation(item) for item in completion.citations]

        # 先把检索上下文透出给调用方，便于前端同步展示证据和审计信息。
        yield ChatStreamEvent(
            event="context",
            data={
                "original_query": completion.original_query,
                "retrieval_query": completion.retrieval_query,
                "rewrite_applied": completion.rewrite_applied,
                "citations": citations_payload,
            },
        )

        if completion.fallback_answer is not None:
            plain_text, content_blocks = self._build_render_content(
                completion.fallback_answer,
                completion.citations,
                finish_reason="context_exhausted",
            )
            yield ChatStreamEvent(
                event="delta",
                data={"content": completion.fallback_answer},
            )
            yield ChatStreamEvent(
                event="done",
                data={
                    "answer": completion.fallback_answer,
                    "plain_text": plain_text,
                    "content_blocks": [
                        self._serialize_content_block(item) for item in content_blocks
                    ],
                    "original_query": completion.original_query,
                    "retrieval_query": completion.retrieval_query,
                    "citations": citations_payload,
                    "finish_reason": "context_exhausted",
                    "rewrite_applied": completion.rewrite_applied,
                },
            )
            return

        answer_parts: list[str] = []
        for delta in self._chat_client.stream_chat(
            completion.model_messages,
            temperature=temperature,
        ):
            if not delta:
                continue
            answer_parts.append(delta)
            yield ChatStreamEvent(event="delta", data={"content": delta})

        answer = "".join(answer_parts).strip() or self._FALLBACK_ANSWER
        plain_text, content_blocks = self._build_render_content(
            answer,
            completion.citations,
            finish_reason="stop",
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
                "finish_reason": "stop",
                "rewrite_applied": completion.rewrite_applied,
            },
        )

    def _prepare_completion(
        self,
        messages: list[ChatMessage],
        retrieval_size: int,
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
        retrieval_query = original_query
        hits = self._chunk_search_service.search(retrieval_query, retrieval_size)
        citations = [self._build_citation(hit, index=index) for index, hit in enumerate(hits, start=1)]

        if not citations:
            return _PreparedChatCompletion(
                original_query=original_query,
                retrieval_query=retrieval_query,
                citations=[],
                model_messages=[],
                fallback_answer=self._FALLBACK_ANSWER,
                rewrite_applied=False,
            )

        return _PreparedChatCompletion(
            original_query=original_query,
            retrieval_query=retrieval_query,
            citations=citations,
            model_messages=self._build_model_messages(
                messages=normalized_messages,
                retrieval_query=retrieval_query,
                citations=citations,
            ),
            fallback_answer=None,
            rewrite_applied=False,
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
    ) -> list[ChatMessage]:
        """构造传给聊天模型的消息列表。"""
        prompt_messages = [
            ChatMessage(role="system", content=self._build_system_prompt()),
            ChatMessage(
                role="system",
                # 把召回证据前置为 system 消息，尽量降低后续多轮对话对证据约束的稀释。
                content=self._build_context_prompt(
                    retrieval_query=retrieval_query,
                    citations=citations,
                ),
            ),
        ]
        prompt_messages.extend(messages)
        return prompt_messages

    def _build_system_prompt(self) -> str:
        """返回配置注入的风控系统提示词。"""
        return self._system_prompt

    def _build_context_prompt(
        self,
        *,
        retrieval_query: str,
        citations: list[ChatCitation],
    ) -> str:
        """把检索结果格式化为模型可消费的证据上下文。"""
        sections = [f"用户当前问题：{retrieval_query}", "以下是可引用的知识库证据："]

        for index, citation in enumerate(citations, start=1):
            sections.append(
                "\n".join(
                    [
                        f"[{index}] 文件：{citation.source_filename}",
                        f"chunk_id：{citation.chunk_id}",
                        f"chunk_index：{citation.chunk_index}",
                        f"score：{citation.score if citation.score is not None else 'null'}",
                        f"内容：{self._truncate_content(citation.content)}",
                    ]
                )
            )

        sections.append("请只基于上述证据回答，不要引用未提供的外部知识。")
        return "\n\n".join(sections)

    def _build_citation(self, hit: ChunkSearchHit, *, index: int) -> ChatCitation:
        """把检索命中结果转换为聊天引用对象。"""
        return ChatCitation(
            citation_id=f"cit-{index}",
            chunk_id=hit.chunk_id,
            file_id=hit.file_id,
            source_filename=hit.source_filename,
            storage_key=hit.storage_key,
            chunk_index=hit.chunk_index,
            char_count=hit.char_count,
            content=hit.content,
            merged_terms=hit.merged_terms,
            score=hit.score,
            snippet=self._build_snippet(hit.content),
            heading_path=[],
            section_title=None,
            content_type="paragraph",
            source_anchor=f"chunk:{hit.chunk_index}",
        )

    def _serialize_citation(self, citation: ChatCitation) -> dict[str, object]:
        """把引用对象转换为可序列化结构。"""
        return {
            "id": citation.citation_id,
            "chunk_id": citation.chunk_id,
            "file_id": citation.file_id,
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
        }

    def _serialize_content_block(self, block: ChatContentBlock) -> dict[str, object]:
        """把结构化正文块转换为可序列化结构。"""
        return {
            "block_id": block.block_id,
            "block_type": block.block_type,
            "text": block.text,
            "citation_ids": block.citation_ids,
            "sequence": block.sequence,
        }

    def _truncate_content(self, content: str) -> str:
        """限制单条证据进入提示词的长度，避免上下文失控。"""
        # 证据要尽量保持原意，但也要控制 token，避免少量长条款挤掉其他召回结果。
        normalized = " ".join(content.split())
        if len(normalized) <= self._MAX_CONTEXT_CHARS:
            return normalized
        return f"{normalized[: self._MAX_CONTEXT_CHARS]}..."

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
        finish_reason: str,
    ) -> tuple[str, list[ChatContentBlock]]:
        """把模型回答解析为可渲染正文块，并完成引用编号校验。"""
        cleaned_answer = answer.strip()
        if not cleaned_answer:
            return "", []

        if finish_reason != "stop" or not citations:
            return cleaned_answer, [
                ChatContentBlock(
                    block_id="blk-1",
                    block_type="notice",
                    text=cleaned_answer,
                    citation_ids=[],
                    sequence=1,
                )
            ]

        raw_blocks = [
            block.strip()
            for block in self._BLOCK_SPLIT_PATTERN.split(cleaned_answer)
            if block.strip()
        ]
        parsed_blocks: list[ChatContentBlock] = []
        has_valid_reference = False

        for sequence, raw_block in enumerate(raw_blocks, start=1):
            citation_ids = self._resolve_block_citation_ids(raw_block, citations)
            if citation_ids:
                has_valid_reference = True
            block_text = self._strip_citation_markers(raw_block).strip()
            if not block_text:
                continue

            parsed_blocks.append(
                ChatContentBlock(
                    block_id=f"blk-{sequence}",
                    block_type="markdown",
                    text=block_text,
                    citation_ids=citation_ids,
                    sequence=sequence,
                )
            )

        if not parsed_blocks:
            stripped_answer = self._strip_citation_markers(cleaned_answer).strip() or cleaned_answer
            return stripped_answer, [
                ChatContentBlock(
                    block_id="blk-1",
                    block_type="markdown",
                    text=stripped_answer,
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
                    citation_ids=[only_citation_id],
                    sequence=block.sequence,
                )
                for block in parsed_blocks
            ]

        plain_text = "\n\n".join(block.text for block in parsed_blocks).strip()
        return plain_text or cleaned_answer, parsed_blocks

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
