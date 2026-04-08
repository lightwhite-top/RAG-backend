"""聊天会话接口。"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path, Query, Request

from baozhi_rag.api.dependencies import get_chat_session_service, get_current_user
from baozhi_rag.core.request_context import ensure_request_id
from baozhi_rag.domain.chat_message import (
    ChatMessageCitationRecord,
    ChatMessageRecord,
    ChatMessageRole,
)
from baozhi_rag.domain.chat_session import ChatSession, ChatSessionStatus
from baozhi_rag.domain.user import CurrentUser
from baozhi_rag.schemas.chat import (
    ChatAssistantMessage,
    ChatCitationItem,
    ChatContentBlockItem,
    ChatTraceItem,
)
from baozhi_rag.schemas.chat_sessions import (
    ChatHistoryMessageItem,
    ChatHistoryMessageListResponseData,
    ChatSessionDetailResponseData,
    ChatSessionItem,
    ChatSessionListResponseData,
    ChatSessionSummaryItem,
    CreateChatSessionRequest,
    UpdateChatSessionRequest,
)
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.services.chat_sessions import ChatSessionMessageHistoryResult, ChatSessionService

router = APIRouter(prefix="/chat/sessions", tags=["chat-sessions"])


@router.post(
    "",
    response_model=SuccessResponse[ChatSessionDetailResponseData],
    summary="创建聊天会话",
)
def create_chat_session(
    request: Request,
    payload: CreateChatSessionRequest,
    service: Annotated[ChatSessionService, Depends(get_chat_session_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[ChatSessionDetailResponseData]:
    session = service.create_session(current_user=current_user, title=payload.title)
    return SuccessResponse[ChatSessionDetailResponseData].success(
        message="创建会话成功",
        request_id=ensure_request_id(request),
        data=ChatSessionDetailResponseData(session=_build_chat_session_item(session)),
    )


@router.get(
    "",
    response_model=SuccessResponse[ChatSessionListResponseData],
    summary="分页查询聊天会话",
)
def list_chat_sessions(
    request: Request,
    service: Annotated[ChatSessionService, Depends(get_chat_session_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    page: Annotated[int, Query(ge=1, description="页码")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="每页数量")] = 20,
    status: Annotated[
        Literal["active", "archived", "all"] | None, Query(description="会话状态")
    ] = None,
) -> SuccessResponse[ChatSessionListResponseData]:
    normalized_status: ChatSessionStatus | None
    if status in (None, "all"):
        normalized_status = None
    elif status == "active":
        normalized_status = ChatSessionStatus.ACTIVE
    else:
        normalized_status = ChatSessionStatus.ARCHIVED
    result = service.list_sessions(
        current_user=current_user,
        page=page,
        page_size=page_size,
        status=normalized_status,
    )
    return SuccessResponse[ChatSessionListResponseData].success(
        message="获取会话列表成功",
        request_id=ensure_request_id(request),
        data=ChatSessionListResponseData(
            items=[_build_chat_session_item(item) for item in result.items]
        ),
        meta={
            "page": result.page,
            "page_size": result.page_size,
            "total": result.total,
        },
    )


@router.get(
    "/{session_id}",
    response_model=SuccessResponse[ChatSessionDetailResponseData],
    summary="获取聊天会话详情",
)
def get_chat_session(
    request: Request,
    session_id: Annotated[str, Path(description="会话ID")],
    service: Annotated[ChatSessionService, Depends(get_chat_session_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[ChatSessionDetailResponseData]:
    session = service.get_session(session_id=session_id, current_user=current_user)
    return SuccessResponse[ChatSessionDetailResponseData].success(
        message="获取会话详情成功",
        request_id=ensure_request_id(request),
        data=ChatSessionDetailResponseData(session=_build_chat_session_item(session)),
    )


@router.patch(
    "/{session_id}",
    response_model=SuccessResponse[ChatSessionDetailResponseData],
    summary="更新聊天会话",
)
def update_chat_session(
    request: Request,
    session_id: Annotated[str, Path(description="会话ID")],
    payload: UpdateChatSessionRequest,
    service: Annotated[ChatSessionService, Depends(get_chat_session_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[ChatSessionDetailResponseData]:
    session = service.update_session(
        session_id=session_id,
        current_user=current_user,
        title=payload.title,
        status=ChatSessionStatus(payload.status) if payload.status is not None else None,
    )
    return SuccessResponse[ChatSessionDetailResponseData].success(
        message="更新会话成功",
        request_id=ensure_request_id(request),
        data=ChatSessionDetailResponseData(session=_build_chat_session_item(session)),
    )


@router.delete(
    "/{session_id}",
    response_model=SuccessResponse[None],
    summary="删除聊天会话",
)
def delete_chat_session(
    request: Request,
    session_id: Annotated[str, Path(description="会话ID")],
    service: Annotated[ChatSessionService, Depends(get_chat_session_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> SuccessResponse[None]:
    service.delete_session(session_id=session_id, current_user=current_user)
    return SuccessResponse[None].success(
        message="删除会话成功",
        request_id=ensure_request_id(request),
    )


@router.get(
    "/{session_id}/messages",
    response_model=SuccessResponse[ChatHistoryMessageListResponseData],
    summary="查询聊天历史消息",
)
def list_chat_session_messages(
    request: Request,
    session_id: Annotated[str, Path(description="会话ID")],
    service: Annotated[ChatSessionService, Depends(get_chat_session_service)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    before_sequence_no: Annotated[int | None, Query(ge=1, description="向前翻页游标")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="返回条数")] = 50,
) -> SuccessResponse[ChatHistoryMessageListResponseData]:
    result = service.list_messages(
        session_id=session_id,
        current_user=current_user,
        before_sequence_no=before_sequence_no,
        limit=limit,
    )
    return SuccessResponse[ChatHistoryMessageListResponseData].success(
        message="获取会话历史成功",
        request_id=ensure_request_id(request),
        data=_build_message_history_response(result),
    )


def _build_chat_session_item(session: ChatSession) -> ChatSessionItem:
    return ChatSessionItem(
        session_id=session.id,
        title=session.title,
        status=session.status.value,
        message_count=session.message_count,
        summary_version=session.summary_version,
        last_message_at=session.last_message_at,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _build_message_history_response(
    result: ChatSessionMessageHistoryResult,
) -> ChatHistoryMessageListResponseData:
    return ChatHistoryMessageListResponseData(
        session=ChatSessionSummaryItem(
            session_id=result.session.id,
            title=result.session.title,
            status=result.session.status.value,
        ),
        items=[_build_history_message_item(item) for item in result.items],
        next_before_sequence_no=result.next_before_sequence_no,
    )


def _build_history_message_item(message: ChatMessageRecord) -> ChatHistoryMessageItem:
    assistant_message: ChatAssistantMessage | None = None
    trace: ChatTraceItem | None = None
    if message.role is ChatMessageRole.ASSISTANT:
        citations = [_build_citation_item(item) for item in message.citations]
        content_blocks = [
            ChatContentBlockItem.model_validate(block) for block in message.content_blocks
        ]
        assistant_message = ChatAssistantMessage(
            message_id=message.id,
            session_id=message.session_id,
            sequence_no=message.sequence_no,
            plain_text=message.plain_text,
            content_blocks=content_blocks,
            citations=citations,
            finish_reason=message.finish_reason or "",
            created_at=message.created_at,
            completed_at=message.completed_at,
        )
        if message.request_id is not None:
            trace = ChatTraceItem(
                request_id=message.request_id,
                original_query=message.original_query or "",
                retrieval_query=message.retrieval_query or "",
                rewrite_applied=message.rewrite_applied,
                model=message.model_name,
                usage=message.usage,
                latency_ms=message.latency_ms,
            )

    return ChatHistoryMessageItem(
        message_id=message.id,
        session_id=message.session_id,
        sequence_no=message.sequence_no,
        role=message.role.value,
        status=message.status.value,
        plain_text=message.plain_text,
        assistant_message=assistant_message,
        request_id=message.request_id,
        trace=trace,
        created_at=message.created_at,
        completed_at=message.completed_at,
    )


def _build_citation_item(message_citation: ChatMessageCitationRecord) -> ChatCitationItem:
    return ChatCitationItem(
        id=message_citation.id,
        chunk_id=message_citation.chunk_id,
        file_id=message_citation.file_id,
        source_filename=message_citation.source_filename,
        storage_key=message_citation.storage_key,
        chunk_index=message_citation.chunk_index,
        char_count=message_citation.char_count,
        content=message_citation.content,
        snippet=message_citation.snippet,
        merged_terms=message_citation.merged_terms,
        score=message_citation.score,
        heading_path=message_citation.heading_path,
        section_title=message_citation.section_title,
        content_type="table" if message_citation.content_type == "table" else "paragraph",
        source_anchor=message_citation.source_anchor,
    )
