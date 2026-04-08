"""有状态聊天编排服务。"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime

from baozhi_rag.domain.chat_errors import ChatSessionModeInvalidMessagesError
from baozhi_rag.domain.chat_message import (
    ChatMessageCitationRecord,
    ChatMessageRecord,
    ChatMessageRole,
    ChatMessageStatus,
)
from baozhi_rag.domain.chat_message_repository import ChatMessageRepository
from baozhi_rag.domain.user import CurrentUser
from baozhi_rag.services.chat import ChatCompletionResult, ChatService, ChatStreamEvent
from baozhi_rag.services.chat_sessions import ChatSessionService
from baozhi_rag.services.llm import ChatMessage


class ConversationChatService:
    """负责有状态聊天的会话读取与消息落库。"""

    _RECENT_CONTEXT_MESSAGE_LIMIT = 8

    def __init__(
        self,
        *,
        chat_service: ChatService,
        session_service: ChatSessionService,
        message_repository: ChatMessageRepository,
        model_name: str | None,
    ) -> None:
        self._chat_service = chat_service
        self._session_service = session_service
        self._message_repository = message_repository
        self._model_name = model_name

    def complete(
        self,
        *,
        session_id: str,
        messages: list[ChatMessage],
        retrieval_size: int,
        temperature: float | None,
        current_user: CurrentUser,
        request_id: str,
    ) -> ChatCompletionResult:
        self._session_service.get_session(session_id=session_id, current_user=current_user)
        current_message = self._extract_current_user_message(messages)
        user_record = self._message_repository.append_message(
            session_id=session_id,
            role=ChatMessageRole.USER,
            status=ChatMessageStatus.COMPLETED,
            plain_text=current_message.content,
            request_id=request_id,
        )

        model_messages = self._build_model_messages(
            session_id=session_id,
            current_user_message=user_record,
        )
        started_at = time.perf_counter()
        result = self._chat_service.complete(
            model_messages,
            retrieval_size=retrieval_size,
            temperature=temperature,
            viewer_user_id=current_user.id,
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        assistant_record = self._message_repository.append_message(
            session_id=session_id,
            role=ChatMessageRole.ASSISTANT,
            status=ChatMessageStatus.COMPLETED,
            plain_text=result.plain_text or result.answer,
            content_blocks=[self._serialize_content_block(item) for item in result.content_blocks],
            request_id=request_id,
            model_name=self._model_name,
            original_query=result.original_query,
            retrieval_query=result.retrieval_query,
            rewrite_applied=result.rewrite_applied,
            retrieval_size=retrieval_size,
            temperature=temperature,
            finish_reason=result.finish_reason,
            latency_ms=latency_ms,
        )
        self._message_repository.replace_citations(
            assistant_record.id,
            self._build_citation_records(assistant_record.id, result),
        )
        persisted_record = self._message_repository.update_message(
            assistant_record.id,
            retrieval_size=retrieval_size,
            temperature=temperature,
            completed=True,
        )
        resolved_record = persisted_record or assistant_record
        return ChatCompletionResult(
            answer=result.answer,
            retrieval_query=result.retrieval_query,
            citations=result.citations,
            finish_reason=result.finish_reason,
            plain_text=result.plain_text,
            content_blocks=result.content_blocks,
            original_query=result.original_query,
            rewrite_applied=result.rewrite_applied,
            message_id=resolved_record.id,
            session_id=session_id,
            sequence_no=resolved_record.sequence_no,
            created_at=resolved_record.created_at,
            completed_at=resolved_record.completed_at,
        )

    def stream(
        self,
        *,
        session_id: str,
        messages: list[ChatMessage],
        retrieval_size: int,
        temperature: float | None,
        current_user: CurrentUser,
        request_id: str,
    ) -> Iterator[ChatStreamEvent]:
        self._session_service.get_session(session_id=session_id, current_user=current_user)
        current_message = self._extract_current_user_message(messages)
        user_record = self._message_repository.append_message(
            session_id=session_id,
            role=ChatMessageRole.USER,
            status=ChatMessageStatus.COMPLETED,
            plain_text=current_message.content,
            request_id=request_id,
        )
        model_messages = self._build_model_messages(
            session_id=session_id,
            current_user_message=user_record,
        )
        assistant_record = self._message_repository.append_message(
            session_id=session_id,
            role=ChatMessageRole.ASSISTANT,
            status=ChatMessageStatus.STREAMING,
            plain_text="",
            request_id=request_id,
            model_name=self._model_name,
            retrieval_size=retrieval_size,
            temperature=temperature,
        )

        started_at = time.perf_counter()
        try:
            for event in self._chat_service.stream(
                model_messages,
                retrieval_size=retrieval_size,
                temperature=temperature,
                viewer_user_id=current_user.id,
            ):
                if event.event == "context":
                    data = dict(event.data)
                    data.setdefault("message_id", assistant_record.id)
                    data.setdefault("session_id", session_id)
                    data.setdefault("sequence_no", assistant_record.sequence_no)
                    yield ChatStreamEvent(event="context", data=data)
                    continue

                if event.event == "done":
                    latency_ms = int((time.perf_counter() - started_at) * 1000)
                    plain_text = str(event.data.get("plain_text", "")).strip()
                    answer = str(event.data.get("answer", "")).strip()
                    original_query = str(event.data.get("original_query", current_message.content))
                    retrieval_query = str(
                        event.data.get("retrieval_query", current_message.content)
                    )
                    rewrite_applied = bool(event.data.get("rewrite_applied", False))
                    finish_reason = str(event.data.get("finish_reason", "stop"))
                    content_blocks = event.data.get("content_blocks")
                    if not isinstance(content_blocks, list):
                        content_blocks = []
                    updated_record = self._message_repository.update_message(
                        assistant_record.id,
                        status=ChatMessageStatus.COMPLETED,
                        plain_text=plain_text or answer,
                        content_blocks=content_blocks,
                        original_query=original_query,
                        retrieval_query=retrieval_query,
                        rewrite_applied=rewrite_applied,
                        retrieval_size=retrieval_size,
                        temperature=temperature,
                        finish_reason=finish_reason,
                        latency_ms=latency_ms,
                        completed=True,
                    )
                    self._message_repository.replace_citations(
                        assistant_record.id,
                        self._build_citation_records_from_event(assistant_record.id, event),
                    )
                    resolved_record = updated_record or assistant_record
                    data = dict(event.data)
                    data.setdefault("message_id", assistant_record.id)
                    data.setdefault("session_id", session_id)
                    data.setdefault("sequence_no", assistant_record.sequence_no)
                    data.setdefault("created_at", resolved_record.created_at.isoformat())
                    data.setdefault(
                        "completed_at",
                        resolved_record.completed_at.isoformat()
                        if resolved_record.completed_at is not None
                        else None,
                    )
                    yield ChatStreamEvent(event="done", data=data)
                    continue

                yield event
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            error_code = getattr(exc, "error_code", "internal_server_error")
            error_message = getattr(exc, "message", str(exc))
            self._message_repository.update_message(
                assistant_record.id,
                status=ChatMessageStatus.FAILED,
                error_code=error_code,
                error_message=error_message,
                latency_ms=latency_ms,
                completed=True,
            )
            raise

    def _extract_current_user_message(self, messages: list[ChatMessage]) -> ChatMessage:
        normalized_messages = [message for message in messages if message.content.strip()]
        if len(normalized_messages) != 1 or normalized_messages[0].role != "user":
            raise ChatSessionModeInvalidMessagesError()
        return ChatMessage(role="user", content=normalized_messages[0].content.strip())

    def _build_model_messages(
        self,
        *,
        session_id: str,
        current_user_message: ChatMessageRecord,
    ) -> list[ChatMessage]:
        recent_messages = self._message_repository.get_recent_messages(
            session_id=session_id,
            limit=self._RECENT_CONTEXT_MESSAGE_LIMIT,
        )
        model_messages: list[ChatMessage] = []
        for message in recent_messages:
            if message.status is not ChatMessageStatus.COMPLETED:
                continue
            model_messages.append(ChatMessage(role=message.role.value, content=message.plain_text))

        if not model_messages or model_messages[-1].content != current_user_message.plain_text:
            model_messages.append(ChatMessage(role="user", content=current_user_message.plain_text))
        return model_messages

    def _build_citation_records(
        self,
        message_id: str,
        result: ChatCompletionResult,
    ) -> list[ChatMessageCitationRecord]:
        now = datetime.now(UTC)
        return [
            ChatMessageCitationRecord(
                id=citation.citation_id or f"{message_id}-cit-{index}",
                message_id=message_id,
                citation_index=index,
                chunk_id=citation.chunk_id,
                file_id=citation.file_id,
                source_filename=citation.source_filename,
                storage_key=citation.storage_key,
                chunk_index=citation.chunk_index,
                char_count=citation.char_count,
                content=citation.content,
                snippet=citation.snippet,
                merged_terms=list(citation.merged_terms),
                score=citation.score,
                heading_path=list(citation.heading_path),
                section_title=citation.section_title,
                content_type=citation.content_type,
                source_anchor=citation.source_anchor,
                created_at=now,
            )
            for index, citation in enumerate(result.citations, start=1)
        ]

    def _build_citation_records_from_event(
        self,
        message_id: str,
        event: ChatStreamEvent,
    ) -> list[ChatMessageCitationRecord]:
        raw_citations = event.data.get("citations", [])
        if not isinstance(raw_citations, list):
            return []

        now = datetime.now(UTC)
        citation_records: list[ChatMessageCitationRecord] = []
        for index, item in enumerate(raw_citations, start=1):
            if not isinstance(item, dict):
                continue
            raw_score = item.get("score")
            citation_records.append(
                ChatMessageCitationRecord(
                    id=str(item.get("id", f"{message_id}-cit-{index}")),
                    message_id=message_id,
                    citation_index=index,
                    chunk_id=str(item.get("chunk_id", "")),
                    file_id=str(item.get("file_id", "")),
                    source_filename=str(item.get("source_filename", "")),
                    storage_key=str(item.get("storage_key", "")),
                    chunk_index=int(item.get("chunk_index", 0)),
                    char_count=int(item.get("char_count", 0)),
                    content=str(item.get("content", "")),
                    snippet=str(item.get("snippet", "")),
                    merged_terms=[
                        str(term).strip()
                        for term in item.get("merged_terms", [])
                        if str(term).strip()
                    ],
                    score=float(raw_score) if raw_score is not None else None,
                    heading_path=[
                        str(term).strip()
                        for term in item.get("heading_path", [])
                        if str(term).strip()
                    ],
                    section_title=str(item["section_title"]).strip()
                    if item.get("section_title") is not None
                    else None,
                    content_type=str(item.get("content_type", "paragraph")),
                    source_anchor=str(item["source_anchor"]).strip()
                    if item.get("source_anchor") is not None
                    else None,
                    created_at=now,
                )
            )
        return citation_records

    def _serialize_content_block(self, block: object) -> dict[str, object]:
        return {
            "block_id": getattr(block, "block_id", ""),
            "block_type": getattr(block, "block_type", "markdown"),
            "text": getattr(block, "text", ""),
            "citation_ids": list(getattr(block, "citation_ids", [])),
            "sequence": int(getattr(block, "sequence", 0)),
        }
