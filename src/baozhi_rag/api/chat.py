"""聊天接口。"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from baozhi_rag.api.dependencies import (
    get_aliyun_oss_file_store,
    get_chat_service,
    get_conversation_chat_service,
    get_current_user,
)
from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.core.exceptions import AppError
from baozhi_rag.core.request_context import REQUEST_ID_HEADER_NAME, ensure_request_id
from baozhi_rag.domain.user import CurrentUser
from baozhi_rag.infra.storage.aliyun_oss_file_store import AliyunOssFileStore
from baozhi_rag.schemas.chat import (
    ChatAssistantMessage,
    ChatCitationItem,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatContentBlockItem,
    ChatImageAssetItem,
    ChatTraceItem,
)
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.services.chat import ChatCompletionResult, ChatService, ChatStreamEvent
from baozhi_rag.services.conversation_chat import ConversationChatService
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
    conversation_service: Annotated[
        ConversationChatService,
        Depends(get_conversation_chat_service),
    ],
    settings: Annotated[Settings, Depends(get_settings)],
    object_store: Annotated[AliyunOssFileStore, Depends(get_aliyun_oss_file_store)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[ChatCompletionResponse] | StreamingResponse:
    """执行带检索增强的聊天补全，支持普通返回和 SSE 流式返回。"""
    request_id = ensure_request_id(request)
    started_at = time.perf_counter()
    messages = [
        ChatMessage(role=message.role, content=message.content) for message in payload.messages
    ]

    if payload.stream:
        original_query = _resolve_original_query(messages)
        if payload.session_id is not None:
            stream_iterator = iter(
                conversation_service.stream(
                    session_id=payload.session_id,
                    messages=messages,
                    retrieval_size=payload.retrieval_size,
                    temperature=payload.temperature,
                    current_user=current_user,
                    request_id=request_id,
                )
            )
        else:
            stream_iterator = iter(
                service.stream(
                    messages,
                    retrieval_size=payload.retrieval_size,
                    temperature=payload.temperature,
                    viewer_user_id=current_user.id,
                )
            )
        first_event = next(stream_iterator)
        message_id = _read_optional_str(first_event.data.get("message_id")) or uuid4().hex
        return StreamingResponse(
            _stream_events(
                request_id=request_id,
                message_id=message_id,
                original_query=original_query,
                model_name=settings.llm_chat_model,
                file_url_builder=object_store,
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

    if payload.session_id is not None:
        result = conversation_service.complete(
            session_id=payload.session_id,
            messages=messages,
            retrieval_size=payload.retrieval_size,
            temperature=payload.temperature,
            current_user=current_user,
            request_id=request_id,
        )
    else:
        result = service.complete(
            messages,
            retrieval_size=payload.retrieval_size,
            temperature=payload.temperature,
            viewer_user_id=current_user.id,
        )

    return SuccessResponse[ChatCompletionResponse].success(
        message="聊天完成",
        request_id=request_id,
        data=_build_completion_response(
            result=result,
            request_id=request_id,
            fallback_original_query=_resolve_original_query(messages),
            model_name=settings.llm_chat_model,
            file_url_builder=object_store,
            latency_ms=_calculate_latency_ms(started_at),
        ),
    )


def _build_completion_response(
    *,
    result: ChatCompletionResult,
    request_id: str,
    fallback_original_query: str,
    model_name: str | None,
    file_url_builder: AliyunOssFileStore,
    latency_ms: int | None,
) -> ChatCompletionResponse:
    citations = _build_citation_items(result.citations, file_url_builder=file_url_builder)
    assistant_message = _build_assistant_message(
        message_id=result.message_id or uuid4().hex,
        session_id=result.session_id,
        sequence_no=result.sequence_no,
        answer=result.answer,
        plain_text=result.plain_text,
        content_blocks=result.content_blocks,
        citations=citations,
        file_url_builder=file_url_builder,
        finish_reason=result.finish_reason,
        created_at=result.created_at,
        completed_at=result.completed_at,
    )
    trace = _build_trace_item(
        request_id=request_id,
        original_query=result.original_query or fallback_original_query,
        retrieval_query=result.retrieval_query,
        rewrite_applied=result.rewrite_applied,
        model_name=model_name,
        latency_ms=latency_ms,
    )
    return ChatCompletionResponse(
        assistant_message=assistant_message,
        trace=trace,
        answer=result.answer,
        retrieval_query=result.retrieval_query,
        citation_count=len(citations),
        citations=citations,
        finish_reason=result.finish_reason,
    )


def _stream_events(
    *,
    request_id: str,
    message_id: str,
    original_query: str,
    model_name: str | None,
    file_url_builder: AliyunOssFileStore,
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
    session_id: str | None = None
    sequence_no: int | None = None

    try:
        for event in _iterate_stream_events(first_event, remaining_events):
            if event.event == "context":
                retrieval_query = str(event.data.get("retrieval_query", original_query))
                rewrite_applied = bool(event.data.get("rewrite_applied", False))
                citations = _build_citation_items(
                    event.data.get("citations", []),
                    file_url_builder=file_url_builder,
                )
                session_id = _read_optional_str(event.data.get("session_id"))
                sequence_no = _read_optional_int(event.data.get("sequence_no"))
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
                            "session_id": session_id,
                            "sequence_no": sequence_no,
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
                            "session_id": session_id,
                            "sequence_no": sequence_no,
                        },
                    )
                    message_started = True
                delta_type = _read_optional_str(event.data.get("delta_type"))
                block_payload = event.data.get("block")
                if delta_type == "content_block" and block_payload is not None:
                    block_items = _build_content_block_items(
                        [block_payload],
                        fallback_text="",
                        citations=citations,
                        file_url_builder=file_url_builder,
                    )
                    if not block_items:
                        continue
                    delta_seq += 1
                    yield _encode_sse_event(
                        "message.delta",
                        {
                            "message_id": message_id,
                            "request_id": request_id,
                            "seq": delta_seq,
                            "delta_type": "content_block",
                            "block": block_items[0].model_dump(),
                        },
                    )
                    continue

                text = str(event.data.get("content", ""))
                if not text:
                    continue
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
                    citations = _build_citation_items(
                        event.data.get("citations", []),
                        file_url_builder=file_url_builder,
                    )
                session_id = _read_optional_str(event.data.get("session_id")) or session_id
                sequence_no = _read_optional_int(event.data.get("sequence_no")) or sequence_no
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
                            "session_id": session_id,
                            "sequence_no": sequence_no,
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
                    session_id=session_id,
                    sequence_no=sequence_no,
                    answer=str(event.data.get("answer", "")),
                    plain_text=_read_optional_str(event.data.get("plain_text")),
                    content_blocks=event.data.get("content_blocks"),
                    citations=citations,
                    file_url_builder=file_url_builder,
                    finish_reason=finish_reason,
                    created_at=_read_optional_datetime(event.data.get("created_at")),
                    completed_at=_read_optional_datetime(event.data.get("completed_at")),
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
    except Exception:
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
    """按顺序遍历首个事件和剩余事件。"""
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


def _build_citation_items(
    raw_citations: Any,
    *,
    file_url_builder: AliyunOssFileStore,
) -> list[ChatCitationItem]:
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
                    image_assets=_build_chat_image_asset_items(
                        item.get("image_assets", []),
                        file_url_builder=file_url_builder,
                    ),
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
                image_assets=_build_chat_image_asset_items(
                    getattr(item, "image_assets", []),
                    file_url_builder=file_url_builder,
                ),
            )
        )
    return citations


def _build_content_block_items(
    raw_blocks: Any,
    *,
    fallback_text: str,
    citations: list[ChatCitationItem],
    file_url_builder: AliyunOssFileStore,
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
                image_assets = _build_chat_image_asset_items(
                    item.get("image_assets", []),
                    file_url_builder=file_url_builder,
                )
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
                image_assets = _build_chat_image_asset_items(
                    getattr(item, "image_assets", []),
                    file_url_builder=file_url_builder,
                )

            if not text and block_type != "image_gallery":
                continue
            blocks.append(
                ChatContentBlockItem(
                    block_id=block_id or f"blk-{len(blocks) + 1}",
                    block_type=block_type,
                    text=text,
                    citation_ids=citation_ids,
                    sequence=sequence,
                    image_assets=image_assets,
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
            image_assets=[],
        )
    ]


def _build_assistant_message(
    *,
    message_id: str,
    session_id: str | None,
    sequence_no: int | None,
    answer: str,
    plain_text: str | None,
    content_blocks: Any,
    citations: list[ChatCitationItem],
    file_url_builder: AliyunOssFileStore,
    finish_reason: str,
    created_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> ChatAssistantMessage:
    """组装结构化助手消息。"""
    resolved_plain_text = (plain_text or "").strip() or answer.strip()
    block_items = _build_content_block_items(
        content_blocks,
        fallback_text=resolved_plain_text,
        citations=citations,
        file_url_builder=file_url_builder,
    )
    normalized_plain_text = (
        resolved_plain_text or "\n\n".join(block.text for block in block_items).strip()
    )
    return ChatAssistantMessage(
        message_id=message_id,
        session_id=session_id,
        sequence_no=sequence_no,
        plain_text=normalized_plain_text,
        content_blocks=block_items,
        citations=citations,
        finish_reason=finish_reason,
        created_at=created_at,
        completed_at=completed_at,
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


def _read_optional_int(value: object) -> int | None:
    """安全读取可选整数。"""
    if value is None:
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_optional_datetime(value: object) -> datetime | None:
    """安全读取可选时间。"""
    if isinstance(value, datetime):
        return value
    text = _read_optional_str(value)
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
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


def _build_chat_image_asset_items(
    raw_assets: Any,
    *,
    file_url_builder: AliyunOssFileStore,
) -> list[ChatImageAssetItem]:
    """把图片资产统一转换为可渲染结构，并补充预签名 URL。"""
    if not isinstance(raw_assets, list):
        return []

    image_assets: list[ChatImageAssetItem] = []
    for item in raw_assets:
        if isinstance(item, dict):
            segment_id = _read_optional_str(item.get("segment_id"))
            storage_key = str(item.get("storage_key", "")).strip()
            thumbnail_storage_key = _read_optional_str(item.get("thumbnail_storage_key"))
            asset_id = str(item.get("asset_id", ""))
            source_anchor = _read_optional_str(item.get("source_anchor"))
            image_type = str(item.get("image_type", ""))
            summary = str(item.get("summary", ""))
            ocr_text = str(item.get("ocr_text", ""))
        else:
            segment_id = _read_optional_str(getattr(item, "segment_id", None))
            storage_key = str(getattr(item, "storage_key", "")).strip()
            thumbnail_storage_key = _read_optional_str(getattr(item, "thumbnail_storage_key", None))
            asset_id = str(getattr(item, "asset_id", ""))
            source_anchor = _read_optional_str(getattr(item, "source_anchor", None))
            image_type = str(getattr(item, "image_type", ""))
            summary = str(getattr(item, "summary", ""))
            ocr_text = str(getattr(item, "ocr_text", ""))
        image_assets.append(
            ChatImageAssetItem(
                segment_id=segment_id,
                asset_id=asset_id,
                source_anchor=source_anchor,
                storage_key=storage_key,
                thumbnail_storage_key=thumbnail_storage_key,
                image_url=file_url_builder.build_presigned_get_url(storage_key=storage_key)
                if storage_key
                else None,
                thumbnail_url=file_url_builder.build_presigned_get_url(
                    storage_key=thumbnail_storage_key
                )
                if thumbnail_storage_key
                else None,
                image_type=image_type,
                summary=summary,
                ocr_text=ocr_text,
            )
        )
    return image_assets


def _normalize_block_type(value: object) -> Literal["markdown", "notice", "image_gallery"]:
    """把正文块类型归一化到协议允许值。"""
    if str(value).strip() == "image_gallery":
        return "image_gallery"
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
