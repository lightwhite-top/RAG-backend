"""聊天接口。"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from baozhi_rag.api.dependencies import get_chat_service
from baozhi_rag.core.exceptions import AppError
from baozhi_rag.core.request_context import REQUEST_ID_HEADER_NAME, ensure_request_id
from baozhi_rag.schemas.chat import (
    ChatCitationItem,
    ChatCompletionRequest,
    ChatCompletionResponse,
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
    # 在 API 边界先完成 schema 到领域消息的转换，避免服务层感知 HTTP 模型。
    messages = [
        ChatMessage(role=message.role, content=message.content) for message in payload.messages
    ]

    if payload.stream:
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
    citations = [
        ChatCitationItem(
            chunk_id=item.chunk_id,
            file_id=item.file_id,
            source_filename=item.source_filename,
            storage_key=item.storage_key,
            chunk_index=item.chunk_index,
            char_count=item.char_count,
            content=item.content,
            merged_terms=item.merged_terms,
            score=item.score,
        )
        for item in result.citations
    ]
    return SuccessResponse[ChatCompletionResponse].success(
        message="聊天完成",
        request_id=request_id,
        data=ChatCompletionResponse(
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
    first_event: ChatStreamEvent,
    remaining_events: Iterator[ChatStreamEvent],
) -> Iterator[str]:
    """把聊天服务事件编码为 SSE 文本流。"""
    try:
        yield _encode_sse_event(first_event.event, first_event.data)
        for event in remaining_events:
            yield _encode_sse_event(event.event, event.data)
    except AppError as exc:
        # SSE 已经开始输出后，不能再切回标准 JSON 错误体，只能继续发 error 事件。
        LOGGER.warning(
            "chat_stream_failed request_id=%s error_code=%s message=%s",
            request_id,
            exc.error_code,
            exc.message,
        )
        yield _encode_sse_event(
            "error",
            {
                "code": exc.error_code,
                "message": exc.message,
                "request_id": request_id,
            },
        )
    except Exception:  # pragma: no cover - 兜底路径不稳定
        LOGGER.exception("chat_stream_failed request_id=%s", request_id)
        yield _encode_sse_event(
            "error",
            {
                "code": "internal_server_error",
                "message": "服务内部错误",
                "request_id": request_id,
            },
        )


def _encode_sse_event(event: str, data: dict[str, object]) -> str:
    """把单个事件编码为 SSE 文本块。"""
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {serialized}\n\n"
