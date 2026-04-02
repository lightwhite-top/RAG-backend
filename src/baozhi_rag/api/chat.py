"""聊天接口。"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from baozhi_rag.api.dependencies import get_chat_service
from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.core.exceptions import AppError
from baozhi_rag.core.request_context import REQUEST_ID_HEADER_NAME, ensure_request_id
from baozhi_rag.schemas.chat import (
    ChatAssistantMessage,
    ChatCitationItem,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatContentBlockItem,
    ChatTraceItem,
)
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.services.chat import ChatService, ChatStreamEvent
from baozhi_rag.services.llm import ChatMessage

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/completions",
    response_model=SuccessResponse[ChatCompletionResponse],
    summary="RAG 聊天补全",
)
def create_chat_completion(
    request: Request,
    payload: ChatCompletionRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SuccessResponse[ChatCompletionResponse] | StreamingResponse:
    """执行带检索增强的聊天补全，支持普通返回和 SSE 流式返回。

    参数:
        request: 当前 HTTP 请求对象，用于提取或生成请求 ID。
        payload: 聊天请求体，包含消息列表、检索条数和流式开关。
        service: 聊天服务实例，负责检索增强与模型调用。

    返回:
        当 `stream=false` 时返回统一成功响应；当 `stream=true` 时返回 `text/event-stream`。

    异常:
        ChatCompletionValidationError: 当请求参数非法时由服务层抛出。
        AlibabaModelStudioError: 当底层聊天模型调用失败时继续上抛。
        HybridChunkStoreError: 当检索链路失败时继续上抛。
    """
    request_id = ensure_request_id(request)
    started_at = time.perf_counter()
    # 在 API 边界先完成 schema 到领域消息的转换，避免服务层感知 HTTP 模型。
    messages = [
        ChatMessage(role=message.role, content=message.content) for message in payload.messages
    ]

    if payload.stream:
        message_id = uuid4().hex
        original_query = _resolve_original_query(messages)
        # 先预取首个事件，确保检索或参数错误在响应头发出前就能走统一异常处理。
        stream_iterator = iter(
            service.stream(
                messages,
                retrieval_size=payload.retrieval_size,
                temperature=payload.temperature,
            )
        )
        first_event = next(stream_iterator)
        return StreamingResponse(
            _stream_events(
                request_id=request_id,
                message_id=message_id,
                original_query=original_query,
                model_name=settings.bailian_chat_model,
                started_at=started_at,
                first_event=first_event,
                remaining_events=stream_iterator,
            ),
            media_type="text/event-stream",
            headers={
                REQUEST_ID_HEADER_NAME: request_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    result = service.complete(
        messages,
        retrieval_size=payload.retrieval_size,
        temperature=payload.temperature,
    )
    citations = _build_citation_items(result.citations)
    message_id = uuid4().hex
    assistant_message = _build_assistant_message(
        message_id=message_id,
        answer=result.answer,
        plain_text=result.plain_text,
        content_blocks=result.content_blocks,
        citations=citations,
        finish_reason=result.finish_reason,
    )
    trace = _build_trace_item(
        request_id=request_id,
        original_query=result.original_query or _resolve_original_query(messages),
        retrieval_query=result.retrieval_query,
        rewrite_applied=result.rewrite_applied,
        model_name=settings.bailian_chat_model,
        latency_ms=_calculate_latency_ms(started_at),
    )
    return SuccessResponse[ChatCompletionResponse].success(
        message="聊天完成",
        request_id=request_id,
        data=ChatCompletionResponse(
            assistant_message=assistant_message,
            trace=trace,
            answer=result.answer,
            retrieval_query=result.retrieval_query,
            citation_count=len(citations),
            citations=citations,
            finish_reason=result.finish_reason,
        ),
    )


def _stream_events(
    *,
    request_id: str,
    message_id: str,
    original_query: str,
    model_name: str | None,
    started_at: float,
    first_event: ChatStreamEvent,
    remaining_events: Iterator[ChatStreamEvent],
) -> Iterator[str]:
    """把聊天服务事件编码为 SSE 文本流。"""
    citations: list[ChatCitationItem] = []
    retrieval_query = original_query
    rewrite_applied = False
    delta_seq = 0
    offset = 0
    message_started = False
    emitted_citation_ids: set[str] = set()

    try:
        for event in _iterate_stream_events(first_event, remaining_events):
            if event.event == "context":
                retrieval_query = str(event.data.get("retrieval_query", original_query))
                rewrite_applied = bool(event.data.get("rewrite_applied", False))
                citations = _build_citation_items(event.data.get("citations", []))
                if not message_started:
                    yield _encode_sse_event(
                        "message.start",
                        {
                            "message_id": message_id,
                            "request_id": request_id,
                            "original_query": original_query,
                            "retrieval_query": retrieval_query,
                            "rewrite_applied": rewrite_applied,
                            "model": model_name,
                        },
                    )
                    message_started = True
                for citation_event in _encode_citation_add_events(
                    request_id=request_id,
                    message_id=message_id,
                    citations=citations,
                    emitted_citation_ids=emitted_citation_ids,
                ):
                    yield citation_event
                continue

            if event.event == "delta":
                text = str(event.data.get("content", ""))
                if not text:
                    continue
                if not message_started:
                    yield _encode_sse_event(
                        "message.start",
                        {
                            "message_id": message_id,
                            "request_id": request_id,
                            "original_query": original_query,
                            "retrieval_query": retrieval_query,
                            "rewrite_applied": rewrite_applied,
                            "model": model_name,
                        },
                    )
                    message_started = True
                delta_seq += 1
                yield _encode_sse_event(
                    "message.delta",
                    {
                        "message_id": message_id,
                        "request_id": request_id,
                        "seq": delta_seq,
                        "offset": offset,
                        "text": text,
                    },
                )
                offset += len(text)
                continue

            if event.event == "done":
                if not citations:
                    citations = _build_citation_items(event.data.get("citations", []))
                if not message_started:
                    yield _encode_sse_event(
                        "message.start",
                        {
                            "message_id": message_id,
                            "request_id": request_id,
                            "original_query": original_query,
                            "retrieval_query": retrieval_query,
                            "rewrite_applied": rewrite_applied,
                            "model": model_name,
                        },
                    )
                    message_started = True
                for citation_event in _encode_citation_add_events(
                    request_id=request_id,
                    message_id=message_id,
                    citations=citations,
                    emitted_citation_ids=emitted_citation_ids,
                ):
                    yield citation_event
                finish_reason = str(event.data.get("finish_reason", "stop"))
                assistant_message = _build_assistant_message(
                    message_id=message_id,
                    answer=str(event.data.get("answer", "")),
                    plain_text=_read_optional_str(event.data.get("plain_text")),
                    content_blocks=event.data.get("content_blocks"),
                    citations=citations,
                    finish_reason=finish_reason,
                )
                trace = _build_trace_item(
                    request_id=request_id,
                    original_query=str(event.data.get("original_query", original_query)),
                    retrieval_query=str(event.data.get("retrieval_query", retrieval_query)),
                    rewrite_applied=bool(event.data.get("rewrite_applied", rewrite_applied)),
                    model_name=model_name,
                    latency_ms=_calculate_latency_ms(started_at),
                )
                yield _encode_sse_event(
                    "message.end",
                    {
                        "message_id": message_id,
                        "request_id": request_id,
                        "assistant_message": assistant_message.model_dump(),
                        "trace": trace.model_dump(),
                    },
                )
    except AppError as exc:
        # SSE 已经开始输出后，不能再切回标准 JSON 错误体，只能继续发 error 事件。
        LOGGER.warning(
            "chat_stream_failed request_id=%s error_code=%s message=%s",
            request_id,
            exc.error_code,
            exc.message,
        )
        yield _encode_sse_event(
            "message.error",
            {
                "message_id": message_id,
                "code": exc.error_code,
                "message": exc.message,
                "request_id": request_id,
            },
        )
    except Exception:  # pragma: no cover - 兜底路径不稳定
        LOGGER.exception("chat_stream_failed request_id=%s", request_id)
        yield _encode_sse_event(
            "message.error",
            {
                "message_id": message_id,
                "code": "internal_server_error",
                "message": "服务内部错误",
                "request_id": request_id,
            },
        )


def _iterate_stream_events(
    first_event: ChatStreamEvent,
    remaining_events: Iterator[ChatStreamEvent],
) -> Iterator[ChatStreamEvent]:
    """按顺序遍历首个事件和剩余事件，避免一次性展开迭代器。"""
    yield first_event
    yield from remaining_events


def _encode_citation_add_events(
    *,
    request_id: str,
    message_id: str,
    citations: list[ChatCitationItem],
    emitted_citation_ids: set[str],
) -> Iterator[str]:
    """为尚未发出的引用卡片生成 `citation.add` 事件。"""
    for citation in citations:
        if citation.id in emitted_citation_ids:
            continue
        emitted_citation_ids.add(citation.id)
        yield _encode_sse_event(
            "citation.add",
            {
                "message_id": message_id,
                "request_id": request_id,
                "citation": citation.model_dump(),
            },
        )


def _build_citation_items(raw_citations: Any) -> list[ChatCitationItem]:
    """把服务层或事件中的引用对象统一转换为 schema。"""
    if not isinstance(raw_citations, list):
        return []

    citations: list[ChatCitationItem] = []
    for index, item in enumerate(raw_citations, start=1):
        if isinstance(item, dict):
            chunk_id = str(item.get("chunk_id", "")).strip()
            citations.append(
                ChatCitationItem(
                    id=_resolve_citation_id(item.get("id"), chunk_id=chunk_id, index=index),
                    chunk_id=chunk_id,
                    file_id=str(item.get("file_id", "")),
                    source_filename=str(item.get("source_filename", "")),
                    storage_key=str(item.get("storage_key", "")),
                    chunk_index=_coerce_int(item.get("chunk_index")),
                    char_count=_coerce_int(item.get("char_count")),
                    content=str(item.get("content", "")),
                    snippet=str(item.get("snippet", item.get("content", ""))),
                    merged_terms=_normalize_string_list(item.get("merged_terms")),
                    score=_read_optional_float(item.get("score")),
                    heading_path=_normalize_string_list(item.get("heading_path")),
                    section_title=_read_optional_str(item.get("section_title")),
                    content_type=_normalize_citation_content_type(item.get("content_type")),
                    source_anchor=_read_optional_str(item.get("source_anchor")),
                )
            )
            continue

        chunk_id = str(getattr(item, "chunk_id", "")).strip()
        citations.append(
            ChatCitationItem(
                id=_resolve_citation_id(
                    getattr(item, "citation_id", None),
                    chunk_id=chunk_id,
                    index=index,
                ),
                chunk_id=chunk_id,
                file_id=str(getattr(item, "file_id", "")),
                source_filename=str(getattr(item, "source_filename", "")),
                storage_key=str(getattr(item, "storage_key", "")),
                chunk_index=_coerce_int(getattr(item, "chunk_index", 0)),
                char_count=_coerce_int(getattr(item, "char_count", 0)),
                content=str(getattr(item, "content", "")),
                snippet=str(getattr(item, "snippet", getattr(item, "content", ""))),
                merged_terms=_normalize_string_list(getattr(item, "merged_terms", [])),
                score=_read_optional_float(getattr(item, "score", None)),
                heading_path=_normalize_string_list(getattr(item, "heading_path", [])),
                section_title=_read_optional_str(getattr(item, "section_title", None)),
                content_type=_normalize_citation_content_type(
                    getattr(item, "content_type", "paragraph")
                ),
                source_anchor=_read_optional_str(getattr(item, "source_anchor", None)),
            )
        )
    return citations


def _build_content_block_items(
    raw_blocks: Any,
    *,
    fallback_text: str,
    citations: list[ChatCitationItem],
) -> list[ChatContentBlockItem]:
    """把服务层正文块统一转换为 schema，并为兼容场景提供兜底块。"""
    blocks: list[ChatContentBlockItem] = []
    available_citation_ids = {item.id for item in citations if item.id}
    if isinstance(raw_blocks, list):
        for item in raw_blocks:
            if isinstance(item, dict):
                block_id = str(item.get("block_id", ""))
                block_type = _normalize_block_type(item.get("block_type"))
                text = str(item.get("text", "")).strip()
                citation_ids = _filter_known_citation_ids(
                    item.get("citation_ids"),
                    available_citation_ids,
                )
                sequence = _coerce_int(item.get("sequence"), default=len(blocks) + 1)
            else:
                block_id = str(getattr(item, "block_id", ""))
                block_type = _normalize_block_type(getattr(item, "block_type", "markdown"))
                text = str(getattr(item, "text", "")).strip()
                citation_ids = _filter_known_citation_ids(
                    getattr(item, "citation_ids", []),
                    available_citation_ids,
                )
                sequence = _coerce_int(
                    getattr(item, "sequence", len(blocks) + 1),
                    default=len(blocks) + 1,
                )

            if not text:
                continue
            blocks.append(
                ChatContentBlockItem(
                    block_id=block_id or f"blk-{len(blocks) + 1}",
                    block_type=block_type,
                    text=text,
                    citation_ids=citation_ids,
                    sequence=sequence,
                )
            )

    if blocks:
        return blocks

    fallback_block_type = _normalize_block_type("notice" if not citations else "markdown")
    return [
        ChatContentBlockItem(
            block_id="blk-1",
            block_type=fallback_block_type,
            text=fallback_text.strip(),
            citation_ids=[citations[0].id] if len(citations) == 1 and citations[0].id else [],
            sequence=1,
        )
    ]


def _build_assistant_message(
    *,
    message_id: str,
    answer: str,
    plain_text: str | None,
    content_blocks: Any,
    citations: list[ChatCitationItem],
    finish_reason: str,
) -> ChatAssistantMessage:
    """组装结构化助手消息。"""
    resolved_plain_text = (plain_text or "").strip() or answer.strip()
    block_items = _build_content_block_items(
        content_blocks,
        fallback_text=resolved_plain_text,
        citations=citations,
    )
    normalized_plain_text = (
        resolved_plain_text or "\n\n".join(block.text for block in block_items).strip()
    )
    return ChatAssistantMessage(
        message_id=message_id,
        plain_text=normalized_plain_text,
        content_blocks=block_items,
        citations=citations,
        finish_reason=finish_reason,
    )


def _build_trace_item(
    *,
    request_id: str,
    original_query: str,
    retrieval_query: str,
    rewrite_applied: bool,
    model_name: str | None,
    latency_ms: int | None,
) -> ChatTraceItem:
    """构造聊天链路追踪信息。"""
    return ChatTraceItem(
        request_id=request_id,
        original_query=original_query,
        retrieval_query=retrieval_query,
        rewrite_applied=rewrite_applied,
        model=model_name,
        usage=None,
        latency_ms=latency_ms,
    )


def _resolve_original_query(messages: list[ChatMessage]) -> str:
    """提取最后一条用户消息，作为原始问题。"""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _calculate_latency_ms(started_at: float) -> int:
    """计算毫秒级耗时。"""
    return int((time.perf_counter() - started_at) * 1000)


def _read_optional_str(value: object) -> str | None:
    """安全读取可选字符串。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_optional_float(value: object) -> float | None:
    """安全读取可选浮点数。"""
    if value is None:
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object, *, default: int = 0) -> int:
    """安全读取整数值。"""
    if not isinstance(value, int | float | str):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_string_list(value: object) -> list[str]:
    """把输入归一化为去空白的字符串列表。"""
    if not isinstance(value, list):
        return []
    normalized_items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized_items.append(text)
    return normalized_items


def _normalize_citation_content_type(value: object) -> Literal["paragraph", "table"]:
    """把证据类型归一化到协议允许值。"""
    return "table" if str(value).strip() == "table" else "paragraph"


def _normalize_block_type(value: object) -> Literal["markdown", "notice"]:
    """把正文块类型归一化到协议允许值。"""
    return "notice" if str(value).strip() == "notice" else "markdown"


def _resolve_citation_id(value: object, *, chunk_id: str, index: int) -> str:
    """为引用生成稳定的前端标识。"""
    return _read_optional_str(value) or chunk_id or f"cit-{index}"


def _filter_known_citation_ids(
    value: object,
    available_citation_ids: set[str],
) -> list[str]:
    """过滤正文块中的非法引用标识。"""
    if not isinstance(value, list):
        return []

    normalized_ids: list[str] = []
    seen_ids: set[str] = set()
    for item in value:
        citation_id = str(item).strip()
        if not citation_id or citation_id in seen_ids:
            continue
        if available_citation_ids and citation_id not in available_citation_ids:
            continue
        seen_ids.add(citation_id)
        normalized_ids.append(citation_id)
    return normalized_ids


def _encode_sse_event(event: str, data: dict[str, object]) -> str:
    """把单个事件编码为 SSE 文本块。"""
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {serialized}\n\n"
