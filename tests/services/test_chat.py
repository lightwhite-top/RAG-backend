from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from baozhi_rag.domain.chat_message import ChatMessageRecord, ChatMessageRole, ChatMessageStatus
from baozhi_rag.domain.user import CurrentUser, UserRole
from baozhi_rag.services.chat import ChatCompletionResult, ChatService, ChatStreamEvent
from baozhi_rag.services.chunk_search import ChunkSearchExecutionResult, ChunkSearchHit
from baozhi_rag.services.conversation_chat import ConversationChatService
from baozhi_rag.services.document_chunking import ChunkImageAsset
from baozhi_rag.services.evidence_sufficiency import EvidenceAssessment
from baozhi_rag.services.llm import ChatMessage
from baozhi_rag.services.query_rewrite import QueryRewriteResult
from baozhi_rag.services.retrieval_trace import RetrievalTrace, RetrievalTraceLane


class _StubChatClient:
    """Return deterministic chat outputs and record the applied temperature."""

    def __init__(self) -> None:
        self.last_complete_temperature: float | None = None
        self.last_stream_temperature: float | None = None

    def complete_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> str:
        del messages
        self.last_complete_temperature = temperature
        return "This is the answer.[1]"

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> Iterator[str]:
        del messages
        self.last_stream_temperature = temperature
        yield "This is the answer.[1]"


class _StubSearchService:
    """Record search inputs and return a single deterministic hit."""

    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_retrieval_mode: str | None = None
        self.last_size: int | None = None

    def search(
        self,
        query_text: str,
        size: int,
        *,
        viewer_user_id: str = "",
        retrieval_mode: str = "search",
    ) -> list[ChunkSearchHit]:
        del viewer_user_id
        self.last_query = query_text
        self.last_retrieval_mode = retrieval_mode
        self.last_size = size
        return [
            ChunkSearchHit(
                chunk_id="file-1-chunk-0",
                file_id="file-1",
                chunk_type="text",
                segment_id="seg-1",
                source_filename="guide.md",
                storage_key="knowledge-files/file-1/guide.md",
                page_number=None,
                source_anchor=None,
                chunk_index=0,
                char_count=20,
                content="Authentication error code explanation.",
                merged_terms=["auth", "error-code"],
                score=0.91,
            )
        ]

    def search_with_trace(
        self,
        query_text: str,
        size: int,
        *,
        viewer_user_id: str = "",
        retrieval_mode: str = "search",
    ) -> ChunkSearchExecutionResult:
        self.last_query = query_text
        self.last_retrieval_mode = retrieval_mode
        self.last_size = size
        hits = self.search(
            query_text,
            size,
            viewer_user_id=viewer_user_id,
            retrieval_mode=retrieval_mode,
        )
        return ChunkSearchExecutionResult(
            hits=hits,
            query_intent="general",
            retrieval_trace=RetrievalTrace(
                mode=retrieval_mode,
                query_intent="general",
                lane_count=1,
                final_hit_count=len(hits),
                evidence_sufficient=True,
                evidence_reason="sufficient",
                lanes=[],
            ),
            evidence_assessment=EvidenceAssessment(
                sufficient=True,
                reason_code="sufficient",
                citation_count=len(hits),
                top_score=hits[0].score,
            ),
        )


@dataclass(frozen=True, slots=True)
class _ExtendedRetrievalTraceLane:
    lane_id: str
    query_text: str
    lane_weight: float
    result_count: int
    top_chunk_ids: list[str]
    expansion_source: str


@dataclass(frozen=True, slots=True)
class _ExtendedRetrievalTrace:
    mode: str
    query_intent: str
    lane_count: int
    final_hit_count: int
    evidence_sufficient: bool
    evidence_reason: str
    deep_rerank_triggered: bool = False
    expansion_strategy: str = "none"
    expansion_queries: list[str] = field(default_factory=list)
    expansion_query_count: int = 0
    lanes: list[_ExtendedRetrievalTraceLane] = field(default_factory=list)


class _RecordingMessageRepository:
    """Capture append/update payloads so tests can assert persisted effective values."""

    def __init__(self) -> None:
        self.records: dict[str, ChatMessageRecord] = {}
        self.append_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []
        self.replace_citation_calls: list[dict[str, object]] = []
        self.fail_update_message = False
        self.fail_replace_citations = False
        self._next_sequence = 1
        self._now = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)

    def append_message(self, **kwargs: object) -> ChatMessageRecord:
        self.append_calls.append(dict(kwargs))
        message_id = f"msg-{self._next_sequence}"
        record = ChatMessageRecord(
            id=message_id,
            session_id=str(kwargs["session_id"]),
            sequence_no=self._next_sequence,
            role=ChatMessageRole(kwargs["role"]),
            status=ChatMessageStatus(kwargs["status"]),
            plain_text=str(kwargs["plain_text"]),
            content_blocks=list(kwargs.get("content_blocks") or []),
            request_id=kwargs.get("request_id"),  # type: ignore[arg-type]
            model_name=kwargs.get("model_name"),  # type: ignore[arg-type]
            original_query=kwargs.get("original_query"),  # type: ignore[arg-type]
            retrieval_query=kwargs.get("retrieval_query"),  # type: ignore[arg-type]
            rewrite_applied=bool(kwargs.get("rewrite_applied", False)),
            retrieval_size=kwargs.get("retrieval_size"),  # type: ignore[arg-type]
            temperature=kwargs.get("temperature"),  # type: ignore[arg-type]
            finish_reason=kwargs.get("finish_reason"),  # type: ignore[arg-type]
            latency_ms=kwargs.get("latency_ms"),  # type: ignore[arg-type]
            usage=kwargs.get("usage"),  # type: ignore[arg-type]
            error_code=kwargs.get("error_code"),  # type: ignore[arg-type]
            error_message=kwargs.get("error_message"),  # type: ignore[arg-type]
            created_at=self._now,
            updated_at=self._now,
            completed_at=self._now,
        )
        self.records[message_id] = record
        self._next_sequence += 1
        return record

    def update_message(self, message_id: str, **kwargs: object) -> ChatMessageRecord | None:
        self.update_calls.append({"message_id": message_id, **kwargs})
        if self.fail_update_message:
            raise RuntimeError("update failed")
        record = self.records.get(message_id)
        if record is None:
            return None
        updated_record = replace(
            record,
            status=kwargs.get("status", record.status),  # type: ignore[arg-type]
            plain_text=str(kwargs.get("plain_text", record.plain_text)),
            content_blocks=list(kwargs.get("content_blocks", record.content_blocks) or []),
            model_name=kwargs.get("model_name", record.model_name),  # type: ignore[arg-type]
            original_query=kwargs.get("original_query", record.original_query),  # type: ignore[arg-type]
            retrieval_query=kwargs.get("retrieval_query", record.retrieval_query),  # type: ignore[arg-type]
            rewrite_applied=bool(kwargs.get("rewrite_applied", record.rewrite_applied)),
            retrieval_size=kwargs.get("retrieval_size", record.retrieval_size),  # type: ignore[arg-type]
            temperature=kwargs.get("temperature", record.temperature),  # type: ignore[arg-type]
            finish_reason=kwargs.get("finish_reason", record.finish_reason),  # type: ignore[arg-type]
            latency_ms=kwargs.get("latency_ms", record.latency_ms),  # type: ignore[arg-type]
            usage=kwargs.get("usage", record.usage),  # type: ignore[arg-type]
            error_code=kwargs.get("error_code", record.error_code),  # type: ignore[arg-type]
            error_message=kwargs.get("error_message", record.error_message),  # type: ignore[arg-type]
            updated_at=self._now,
            completed_at=self._now if kwargs.get("completed") else record.completed_at,
        )
        self.records[message_id] = updated_record
        return updated_record

    def replace_citations(self, message_id: str, citations: list[object]) -> list[object]:
        self.replace_citation_calls.append(
            {"message_id": message_id, "citation_count": len(citations)}
        )
        if self.fail_replace_citations:
            raise RuntimeError("replace citations failed")
        return citations

    def get_recent_messages(self, *, session_id: str, limit: int) -> list[ChatMessageRecord]:
        del session_id, limit
        return []


class _StubSessionService:
    def get_session(self, *, session_id: str, current_user: CurrentUser) -> None:
        del session_id, current_user


def _build_current_user() -> CurrentUser:
    now = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
    return CurrentUser(
        id="user-1",
        email="user@example.com",
        username="Light",
        role=UserRole.USER,
        created_at=now,
        updated_at=now,
    )


def test_chat_service_applies_followup_runtime_policy() -> None:
    search_service = _StubSearchService()
    chat_client = _StubChatClient()
    service = ChatService(
        chat_client=chat_client,  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [
            ChatMessage(role="user", content="What does the rate-limit window mean?"),
            ChatMessage(role="assistant", content="Previous answer."),
            ChatMessage(role="user", content="Then what about auth error codes?"),
        ],
        retrieval_size=2,
        temperature=1.4,
    )

    assert isinstance(result, ChatCompletionResult)
    assert search_service.last_query == "Then what about auth error codes?"
    assert search_service.last_retrieval_mode == "chat"
    assert search_service.last_size == result.applied_retrieval_size == 7
    assert result.applied_temperature == 0.0
    assert chat_client.last_complete_temperature == 0.0
    assert result.rewrite_applied is False


def test_chat_service_ignores_client_generation_parameters() -> None:
    search_service = _StubSearchService()
    chat_client = _StubChatClient()
    service = ChatService(
        chat_client=chat_client,  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [ChatMessage(role="user", content="What is vector search?")],
        retrieval_size=1,
        temperature=1.9,
    )

    assert result.applied_retrieval_size == 5
    assert result.applied_temperature == 0.0
    assert search_service.last_size == 5
    assert chat_client.last_complete_temperature == 0.0


def test_chat_service_honors_larger_requested_retrieval_size_within_cap() -> None:
    search_service = _StubSearchService()
    chat_client = _StubChatClient()
    service = ChatService(
        chat_client=chat_client,  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [ChatMessage(role="user", content="What is vector search?")],
        retrieval_size=8,
        temperature=1.9,
    )

    assert result.applied_retrieval_size == 8
    assert search_service.last_size == 8
    assert chat_client.last_complete_temperature == 0.0


def test_chat_service_injects_markdown_render_contract_prompt() -> None:
    search_service = _StubSearchService()
    service = ChatService(
        chat_client=_StubChatClient(),  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
    )

    prepared = service._build_model_messages(  # pyright: ignore[reportPrivateUsage]
        messages=[ChatMessage(role="user", content="What is vector search?")],
        context_prompt="Knowledge context",
    )

    assert len(prepared) >= 2
    assert "Markdown" in prepared[1].content


def test_chat_service_stream_emits_applied_generation_params() -> None:
    class _StreamingChatClient(_StubChatClient):
        def stream_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> Iterator[str]:
            del messages
            self.last_stream_temperature = temperature
            yield "This is the answer"
            yield ".[1]"

    search_service = _StubSearchService()
    chat_client = _StreamingChatClient()
    service = ChatService(
        chat_client=chat_client,  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
    )

    events = list(
        service.stream(
            [ChatMessage(role="user", content="What is an auth error code?")],
            retrieval_size=1,
            temperature=1.7,
        )
    )

    context_event = events[0]
    delta_events = [event for event in events if event.event == "delta"]
    done_event = events[-1]

    assert context_event.event == "context"
    assert context_event.data["applied_retrieval_size"] == 5
    assert context_event.data["applied_temperature"] == 0.0
    assert [event.data["delta_type"] for event in delta_events] == ["append", "append", "insert"]
    assert chat_client.last_stream_temperature == 0.0
    assert done_event.event == "done"
    assert done_event.data["applied_retrieval_size"] == 5
    assert done_event.data["applied_temperature"] == 0.0


def test_chat_service_followup_runtime_policy_caps_requested_retrieval_size() -> None:
    search_service = _StubSearchService()
    chat_client = _StubChatClient()
    service = ChatService(
        chat_client=chat_client,  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [
            ChatMessage(role="user", content="What does the rate-limit window mean?"),
            ChatMessage(role="assistant", content="Previous answer."),
            ChatMessage(role="user", content="Then what about auth error codes?"),
        ],
        retrieval_size=20,
        temperature=1.4,
    )

    assert result.applied_retrieval_size == 10
    assert search_service.last_size == 10


def test_chat_service_uses_rewritten_query_before_retrieval() -> None:
    class _RewriteService:
        def rewrite(
            self,
            messages: list[ChatMessage],
            *,
            original_query: str,
        ) -> QueryRewriteResult:
            del messages
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=f"{original_query}（改写）",
                rewrite_applied=True,
            )

    search_service = _StubSearchService()
    service = ChatService(
        chat_client=_StubChatClient(),  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
        query_rewrite_service=_RewriteService(),  # type: ignore[arg-type]
    )

    result = service.complete(
        [ChatMessage(role="user", content="这个报错是什么意思？")],
        retrieval_size=1,
    )

    assert search_service.last_query == "这个报错是什么意思？（改写）"
    assert result.rewrite_applied is True
    assert result.retrieval_query == "这个报错是什么意思？（改写）"


def test_chat_service_stream_passes_extended_retrieval_trace_fields() -> None:
    class _ExtendedTraceSearchService(_StubSearchService):
        def search_with_trace(
            self,
            query_text: str,
            size: int,
            *,
            viewer_user_id: str = "",
            retrieval_mode: str = "search",
        ) -> ChunkSearchExecutionResult:
            hits = self.search(
                query_text,
                size,
                viewer_user_id=viewer_user_id,
                retrieval_mode=retrieval_mode,
            )
            return ChunkSearchExecutionResult(
                hits=hits,
                query_intent="general",
                retrieval_trace=_ExtendedRetrievalTrace(  # type: ignore[arg-type]
                    mode=retrieval_mode,
                    query_intent="general",
                    lane_count=2,
                    final_hit_count=len(hits),
                    evidence_sufficient=True,
                    evidence_reason="sufficient",
                    deep_rerank_triggered=False,
                    expansion_strategy="llm_query_expansion",
                    expansion_queries=["auth error code", "authentication error code"],
                    expansion_query_count=2,
                    lanes=[
                        _ExtendedRetrievalTraceLane(
                            lane_id="base-query",
                            query_text=query_text,
                            lane_weight=1.0,
                            result_count=len(hits),
                            top_chunk_ids=[hit.chunk_id for hit in hits],
                            expansion_source="original",
                        ),
                        _ExtendedRetrievalTraceLane(
                            lane_id="expanded-query",
                            query_text="authentication error code",
                            lane_weight=0.8,
                            result_count=len(hits),
                            top_chunk_ids=[hit.chunk_id for hit in hits],
                            expansion_source="llm",
                        ),
                    ],
                ),
                evidence_assessment=EvidenceAssessment(
                    sufficient=True,
                    reason_code="sufficient",
                    citation_count=len(hits),
                    top_score=hits[0].score if hits else None,
                ),
            )

    service = ChatService(
        chat_client=_StubChatClient(),  # type: ignore[arg-type]
        chunk_search_service=_ExtendedTraceSearchService(),
        system_prompt="You are a test assistant.",
    )

    events = list(
        service.stream(
            [ChatMessage(role="user", content="What is an auth error code?")],
            retrieval_size=1,
        )
    )

    context_trace = events[0].data["retrieval_trace"]
    done_trace = events[-1].data["retrieval_trace"]

    assert isinstance(context_trace, dict)
    assert isinstance(done_trace, dict)
    assert context_trace["expansion_strategy"] == "llm_query_expansion"
    assert context_trace["expansion_query_count"] == 2
    assert context_trace["expansion_queries"] == [
        "auth error code",
        "authentication error code",
    ]
    assert context_trace["lanes"][1]["expansion_source"] == "llm"
    assert done_trace["expansion_strategy"] == "llm_query_expansion"
    assert done_trace["lanes"][0]["expansion_source"] == "original"


def test_chat_service_stream_hides_embedding_source_summary_from_public_trace() -> None:
    class _TraceSanitizingSearchService(_StubSearchService):
        def search_with_trace(
            self,
            query_text: str,
            size: int,
            *,
            viewer_user_id: str = "",
            retrieval_mode: str = "search",
        ) -> ChunkSearchExecutionResult:
            hits = self.search(
                query_text,
                size,
                viewer_user_id=viewer_user_id,
                retrieval_mode=retrieval_mode,
            )
            return ChunkSearchExecutionResult(
                hits=hits,
                query_intent="general",
                retrieval_trace=RetrievalTrace(
                    mode=retrieval_mode,
                    query_intent="general",
                    lane_count=2,
                    final_hit_count=len(hits),
                    evidence_sufficient=True,
                    evidence_reason="sufficient",
                    lanes=[
                        RetrievalTraceLane(
                            lane_id="hyde-lane",
                            query_text=query_text,
                            lane_weight=0.8,
                            result_count=len(hits),
                            top_chunk_ids=[hit.chunk_id for hit in hits],
                            source_strategy="hyde",
                            embedding_source_summary="这是一段不应暴露给客户端的 HyDE 假设文本",
                        )
                    ],
                ),
                evidence_assessment=EvidenceAssessment(
                    sufficient=True,
                    reason_code="sufficient",
                    citation_count=len(hits),
                    top_score=hits[0].score if hits else None,
                ),
            )

    service = ChatService(
        chat_client=_StubChatClient(),  # type: ignore[arg-type]
        chunk_search_service=_TraceSanitizingSearchService(),
        system_prompt="You are a test assistant.",
    )

    events = list(
        service.stream(
            [ChatMessage(role="user", content="What is an auth error code?")],
            retrieval_size=1,
        )
    )

    context_lane = events[0].data["retrieval_trace"]["lanes"][0]
    done_lane = events[-1].data["retrieval_trace"]["lanes"][0]

    assert "embedding_source_summary" not in context_lane
    assert context_lane["embedding_source_changed"] is True
    assert "embedding_source_summary" not in done_lane
    assert done_lane["embedding_source_changed"] is True


def test_chat_service_complete_injects_image_and_source_file_blocks() -> None:
    search_service = _StubSearchService()
    search_hit = replace(
        search_service.search("", 1)[0],  # type: ignore[arg-type]
        image_assets=[
            ChunkImageAsset(
                segment_id="seg-1",
                asset_id="img-1",
                asset_index=1,
                source_anchor="p:1:image:1",
                content_type="image/png",
                extension=".png",
                image_bytes=None,
                storage_key="knowledge-files/file-1/assets/images/img-1.png",
                thumbnail_storage_key="knowledge-files/file-1/assets/thumbnails/img-1.png",
                summary="Process diagram",
                ocr_text="auth error code",
                image_type="diagram",
            )
        ],
        file_content_type="text/markdown",
        file_extension="md",
        file_size=128,
    )
    search_service.search = lambda *args, **kwargs: [search_hit]  # type: ignore[method-assign]
    search_service.search_with_trace = lambda *args, **kwargs: ChunkSearchExecutionResult(  # type: ignore[method-assign]
        hits=[search_hit],
        query_intent="general",
        retrieval_trace=RetrievalTrace(
            mode="chat",
            query_intent="general",
            lane_count=1,
            final_hit_count=1,
            evidence_sufficient=True,
            evidence_reason="sufficient",
            lanes=[],
        ),
        evidence_assessment=EvidenceAssessment(
            sufficient=True,
            reason_code="sufficient",
            citation_count=1,
            top_score=search_hit.score,
        ),
    )
    service = ChatService(
        chat_client=_StubChatClient(),  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [ChatMessage(role="user", content="What is an auth error code?")],
        retrieval_size=5,
    )

    assert [block.block_type for block in result.content_blocks] == [
        "markdown",
        "image_gallery",
        "source_file",
    ]
    assert result.content_blocks[1].files_assets[0].preview_storage_key is not None
    assert result.content_blocks[2].files_assets[0].extension == "md"


def test_conversation_chat_service_persists_applied_generation_params() -> None:
    repository = _RecordingMessageRepository()
    result = ChatCompletionResult(
        answer="Answer.[1]",
        retrieval_query="rewritten query",
        citations=[],
        finish_reason="stop",
        plain_text="Answer.",
        original_query="original query",
        applied_retrieval_size=5,
        applied_temperature=0.0,
    )

    class _StubConversationChatDelegate:
        def complete(
            self,
            messages: list[ChatMessage],
            *,
            retrieval_size: int,
            temperature: float | None = None,
            viewer_user_id: str = "",
        ) -> ChatCompletionResult:
            del messages, retrieval_size, temperature, viewer_user_id
            return result

    service = ConversationChatService(
        chat_service=_StubConversationChatDelegate(),  # type: ignore[arg-type]
        session_service=_StubSessionService(),  # type: ignore[arg-type]
        message_repository=repository,  # type: ignore[arg-type]
        model_name="test-model",
    )

    completed = service.complete(
        session_id="sess-1",
        messages=[ChatMessage(role="user", content="What is a deductible?")],
        retrieval_size=1,
        temperature=1.8,
        current_user=_build_current_user(),
        request_id="req-1",
    )

    assistant_append = repository.append_calls[1]
    assistant_update = repository.update_calls[0]
    assert assistant_append["retrieval_size"] == 5
    assert assistant_append["temperature"] == 0.0
    assert assistant_update["retrieval_size"] == 5
    assert assistant_update["temperature"] == 0.0
    assert completed.applied_retrieval_size == 5
    assert completed.applied_temperature == 0.0


def test_conversation_chat_service_stream_still_yields_done_when_persisting_final_message_fails() -> (
    None
):
    repository = _RecordingMessageRepository()
    repository.fail_replace_citations = True

    class _StubStreamingConversationDelegate:
        def stream(
            self,
            messages: list[ChatMessage],
            *,
            retrieval_size: int,
            temperature: float | None = None,
            viewer_user_id: str = "",
        ) -> Iterator[ChatStreamEvent]:
            del messages, retrieval_size, temperature, viewer_user_id
            yield ChatStreamEvent(
                event="context",
                data={
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "rewrite_applied": False,
                    "citations": [],
                    "applied_retrieval_size": 5,
                    "applied_temperature": 0.0,
                },
            )
            yield ChatStreamEvent(
                event="done",
                data={
                    "answer": "Answer.[1]",
                    "plain_text": "Answer.",
                    "content_blocks": [],
                    "original_query": "original query",
                    "retrieval_query": "rewritten query",
                    "citations": [],
                    "finish_reason": "stop",
                    "rewrite_applied": False,
                    "applied_retrieval_size": 5,
                    "applied_temperature": 0.0,
                },
            )

    service = ConversationChatService(
        chat_service=_StubStreamingConversationDelegate(),  # type: ignore[arg-type]
        session_service=_StubSessionService(),  # type: ignore[arg-type]
        message_repository=repository,  # type: ignore[arg-type]
        model_name="test-model",
    )

    events = list(
        service.stream(
            session_id="sess-1",
            messages=[ChatMessage(role="user", content="What is a deductible?")],
            retrieval_size=1,
            temperature=1.8,
            current_user=_build_current_user(),
            request_id="req-1",
        )
    )

    assert [event.event for event in events] == ["context", "done"]
    assert events[-1].data["message_id"] == "msg-2"
    assert events[-1].data["session_id"] == "sess-1"
    assert events[-1].data["sequence_no"] == 2
    assert events[-1].data["completed_at"] is not None
    assert repository.update_calls
    assert repository.replace_citation_calls


def test_chat_service_returns_deterministic_fallback_when_evidence_is_insufficient() -> None:
    class _ExplodingChatClient:
        def complete_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> str:
            raise AssertionError("证据不足时不应再调用模型补全")

        def stream_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> Iterator[str]:
            raise AssertionError("证据不足时不应再调用模型流式输出")

    class _NoEvidenceSearchService:
        def search(
            self,
            query_text: str,
            size: int,
            *,
            viewer_user_id: str = "",
            retrieval_mode: str = "search",
        ) -> list[ChunkSearchHit]:
            del query_text, size, viewer_user_id, retrieval_mode
            return []

        def search_with_trace(
            self,
            query_text: str,
            size: int,
            *,
            viewer_user_id: str = "",
            retrieval_mode: str = "search",
        ) -> ChunkSearchExecutionResult:
            del query_text, size, viewer_user_id
            return ChunkSearchExecutionResult(
                hits=[],
                query_intent="general",
                retrieval_trace=RetrievalTrace(
                    mode=retrieval_mode,
                    query_intent="general",
                    lane_count=1,
                    final_hit_count=0,
                    evidence_sufficient=False,
                    evidence_reason="no_evidence",
                    lanes=[],
                ),
                evidence_assessment=EvidenceAssessment(
                    sufficient=False,
                    reason_code="no_evidence",
                    citation_count=0,
                    top_score=None,
                ),
            )

    service = ChatService(
        chat_client=_ExplodingChatClient(),  # type: ignore[arg-type]
        chunk_search_service=_NoEvidenceSearchService(),  # type: ignore[arg-type]
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [ChatMessage(role="user", content="系统里有这个制度吗？")],
        retrieval_size=3,
    )

    assert result.answer == ChatService._FALLBACK_ANSWER  # pyright: ignore[reportPrivateUsage]
    assert result.finish_reason == "evidence_insufficient"
    assert result.citations == []


def test_chat_service_returns_fallback_when_context_budget_exhausted() -> None:
    class _ExplodingChatClient:
        def complete_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> str:
            raise AssertionError("上下文预算耗尽时不应继续调用模型")

        def stream_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> Iterator[str]:
            raise AssertionError("上下文预算耗尽时不应继续调用模型流式输出")

    class _BudgetExhaustedSearchService(_StubSearchService):
        def search_with_trace(
            self,
            query_text: str,
            size: int,
            *,
            viewer_user_id: str = "",
            retrieval_mode: str = "search",
        ) -> ChunkSearchExecutionResult:
            hits = self.search(
                query_text,
                size,
                viewer_user_id=viewer_user_id,
                retrieval_mode=retrieval_mode,
            )
            return ChunkSearchExecutionResult(
                hits=hits,
                query_intent="general",
                retrieval_trace=RetrievalTrace(
                    mode=retrieval_mode,
                    query_intent="general",
                    lane_count=1,
                    final_hit_count=len(hits),
                    evidence_sufficient=False,
                    evidence_reason="context_budget_exhausted",
                    lanes=[],
                ),
                evidence_assessment=EvidenceAssessment(
                    sufficient=False,
                    reason_code="context_budget_exhausted",
                    citation_count=0,
                    top_score=hits[0].score if hits else None,
                ),
            )

    service = ChatService(
        chat_client=_ExplodingChatClient(),  # type: ignore[arg-type]
        chunk_search_service=_BudgetExhaustedSearchService(),  # type: ignore[arg-type]
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [ChatMessage(role="user", content="这个制度到底怎么规定？")],
        retrieval_size=3,
    )

    assert result.answer == ChatService._FALLBACK_ANSWER  # pyright: ignore[reportPrivateUsage]
    assert result.finish_reason == "evidence_insufficient"
    assert result.citations == []


def test_chat_service_hides_low_score_citations_when_returning_fallback() -> None:
    class _ExplodingChatClient:
        def complete_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> str:
            raise AssertionError("低分证据场景不应继续调用模型")

        def stream_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> Iterator[str]:
            raise AssertionError("低分证据场景不应继续调用模型")

    low_score_hit = ChunkSearchHit(
        chunk_id="chunk-1",
        file_id="file-1",
        chunk_type="text",
        segment_id="seg-1",
        source_filename="guide.md",
        storage_key="guide.md",
        page_number=None,
        source_anchor=None,
        chunk_index=0,
        char_count=12,
        content="low score content",
        merged_terms=[],
        score=0.01,
    )

    class _LowScoreSearchService:
        def search(
            self,
            query_text: str,
            size: int,
            *,
            viewer_user_id: str = "",
            retrieval_mode: str = "search",
        ) -> list[ChunkSearchHit]:
            del query_text, size, viewer_user_id, retrieval_mode
            return [low_score_hit]

        def search_with_trace(
            self,
            query_text: str,
            size: int,
            *,
            viewer_user_id: str = "",
            retrieval_mode: str = "search",
        ) -> ChunkSearchExecutionResult:
            del query_text, size, viewer_user_id
            return ChunkSearchExecutionResult(
                hits=[low_score_hit],
                query_intent="general",
                retrieval_trace=RetrievalTrace(
                    mode=retrieval_mode,
                    query_intent="general",
                    lane_count=1,
                    final_hit_count=1,
                    evidence_sufficient=False,
                    evidence_reason="low_score",
                    lanes=[],
                ),
                evidence_assessment=EvidenceAssessment(
                    sufficient=False,
                    reason_code="low_score",
                    citation_count=1,
                    top_score=low_score_hit.score,
                ),
            )

    service = ChatService(
        chat_client=_ExplodingChatClient(),  # type: ignore[arg-type]
        chunk_search_service=_LowScoreSearchService(),  # type: ignore[arg-type]
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [ChatMessage(role="user", content="这条规则是什么？")],
        retrieval_size=3,
    )

    assert result.answer == ChatService._FALLBACK_ANSWER  # pyright: ignore[reportPrivateUsage]
    assert result.finish_reason == "evidence_insufficient"
    assert result.citations == []


def test_chat_service_suppresses_citations_when_model_answer_declares_missing_evidence() -> None:
    class _ConservativeChatClient(_StubChatClient):
        def complete_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> str:
            del messages, temperature
            return "当前可检索到的知识库材料未提及员工报销宠物托管费的相关规定。"

    search_service = _StubSearchService()
    service = ChatService(
        chat_client=_ConservativeChatClient(),  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [ChatMessage(role="user", content="知识库里有没有规定员工可以报销宠物托管费？")],
        retrieval_size=5,
    )

    assert result.citations == []
    assert result.evidence_assessment is not None
    assert result.evidence_assessment.sufficient is False
    assert result.evidence_assessment.reason_code == "answer_indicates_no_evidence"


def test_chat_service_suppresses_citations_when_answer_says_not_specified() -> None:
    class _ConservativeChatClient(_StubChatClient):
        def complete_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> str:
            del messages, temperature
            return "现有培训手册未规定新人考试满分必须达到99分，仅明确通过线为85分。"

    search_service = _StubSearchService()
    service = ChatService(
        chat_client=_ConservativeChatClient(),  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [ChatMessage(role="user", content="培训手册是否规定新人考试满分必须达到99分？")],
        retrieval_size=5,
    )

    assert result.citations == []
    assert result.evidence_assessment is not None
    assert result.evidence_assessment.reason_code == "answer_indicates_no_evidence"


def test_chat_service_keeps_citations_for_governance_style_missing_evidence_answer() -> None:
    class _GovernanceChatClient(_StubChatClient):
        def complete_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> str:
            del messages, temperature
            return (
                "01_财务报销制度_V2.docx未提及遇到证据不足时的具体表述规则，"
                "仅可提供一般性要求：若证据不足无法支撑问题结论，应明确说明"
                "该文档未涵盖对应问题的相关内容，不得编造未被该文档支持的结论。"
            )

    search_service = _StubSearchService()
    service = ChatService(
        chat_client=_GovernanceChatClient(),  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [
            ChatMessage(
                role="user",
                content="如果只基于 01_财务报销制度_V2.docx 回答，遇到证据不足时应该怎么说？",
            )
        ],
        retrieval_size=5,
    )

    assert result.citations
    assert result.evidence_assessment is not None
    assert result.evidence_assessment.reason_code == "sufficient"


def test_chat_service_keeps_citations_for_boundary_condition_question() -> None:
    class _BoundaryChatClient(_StubChatClient):
        def complete_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> str:
            del messages, temperature
            return (
                "现有检索到的知识库内容仅明确该手册中规定实操陪跑周期为5个工作日，"
                "未提及处理实操陪跑周期前需要预先确认的相关边界条件。"
            )

    search_service = _StubSearchService()
    service = ChatService(
        chat_client=_BoundaryChatClient(),  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [
            ChatMessage(
                role="user",
                content="处理实操陪跑周期前，需要先确认哪些边界条件？",
            )
        ],
        retrieval_size=5,
    )

    assert result.citations
    assert result.evidence_assessment is not None
    assert result.evidence_assessment.reason_code == "sufficient"


def test_chat_service_appends_guidance_notice_for_governance_queries() -> None:
    class _GovernanceChatClient(_StubChatClient):
        def complete_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> str:
            del messages, temperature
            return "当前在目标文档中未检索到该问题的相关内容，无法给出对应结论。"

    service = ChatService(
        chat_client=_GovernanceChatClient(),  # type: ignore[arg-type]
        chunk_search_service=_StubSearchService(),
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [
            ChatMessage(
                role="user",
                content="如果只基于 01_财务报销制度_V2.docx 回答，遇到证据不足时应该怎么说？",
            )
        ],
        retrieval_size=5,
    )

    assert "依据有限" in result.answer
    assert "补充材料" in result.answer
    assert "不能编造" in result.answer


def test_chat_service_appends_boundary_notice_for_boundary_condition_queries() -> None:
    class _BoundaryChatClient(_StubChatClient):
        def complete_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> str:
            del messages, temperature
            return "现有内容仅明确触发时点为HR状态变更后30分钟内。"

    service = ChatService(
        chat_client=_BoundaryChatClient(),  # type: ignore[arg-type]
        chunk_search_service=_StubSearchService(),
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [
            ChatMessage(
                role="user",
                content="处理离职账号冻结触发时点前应该先确认哪些边界条件？",
            )
        ],
        retrieval_size=5,
    )

    assert "适用对象" in result.answer
    assert "版本" in result.answer
    assert "材料完整度" in result.answer


def test_chat_service_normalizes_ascii_cjk_spacing_in_answer() -> None:
    class _BoundaryChatClient(_StubChatClient):
        def complete_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> str:
            del messages, temperature
            return "离职账号冻结触发时点为HR状态变更后30分钟内。"

    service = ChatService(
        chat_client=_BoundaryChatClient(),  # type: ignore[arg-type]
        chunk_search_service=_StubSearchService(),
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [
            ChatMessage(
                role="user",
                content="按照 14_FAQ_权限申请与账号异常问答.pdf 的流程，处理 离职账号冻结触发时点 前应该先确认哪些边界条件？",
            )
        ],
        retrieval_size=5,
    )

    assert "HR 状态变更后30分钟内" in result.answer


def test_chat_service_still_calls_model_for_conflicting_evidence() -> None:
    class _ConflictingEvidenceSearchService(_StubSearchService):
        def search_with_trace(
            self,
            query_text: str,
            size: int,
            *,
            viewer_user_id: str = "",
            retrieval_mode: str = "search",
        ) -> ChunkSearchExecutionResult:
            hits = self.search(
                query_text,
                size,
                viewer_user_id=viewer_user_id,
                retrieval_mode=retrieval_mode,
            )
            return ChunkSearchExecutionResult(
                hits=hits,
                query_intent="general",
                retrieval_trace=RetrievalTrace(
                    mode=retrieval_mode,
                    query_intent="general",
                    lane_count=1,
                    final_hit_count=len(hits),
                    evidence_sufficient=False,
                    evidence_reason="conflicting_evidence",
                    lanes=[],
                ),
                evidence_assessment=EvidenceAssessment(
                    sufficient=False,
                    reason_code="conflicting_evidence",
                    citation_count=len(hits),
                    top_score=hits[0].score if hits else None,
                ),
            )

    chat_client = _StubChatClient()
    service = ChatService(
        chat_client=chat_client,  # type: ignore[arg-type]
        chunk_search_service=_ConflictingEvidenceSearchService(),
        system_prompt="You are a test assistant.",
    )

    result = service.complete(
        [ChatMessage(role="user", content="这个制度到底是启用还是禁用？")],
        retrieval_size=3,
    )

    assert result.answer == "This is the answer.[1]"
    assert result.finish_reason == "evidence_insufficient"
    assert chat_client.last_complete_temperature == 0.0


def test_chat_service_stream_returns_fallback_without_calling_model_when_evidence_is_insufficient() -> (
    None
):
    class _ExplodingChatClient:
        def complete_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> str:
            raise AssertionError("证据不足时不应再调用模型补全")

        def stream_chat(
            self,
            messages: list[ChatMessage],
            *,
            temperature: float | None = None,
        ) -> Iterator[str]:
            raise AssertionError("证据不足时不应再调用模型流式输出")

    class _NoEvidenceSearchService:
        def search(
            self,
            query_text: str,
            size: int,
            *,
            viewer_user_id: str = "",
            retrieval_mode: str = "search",
        ) -> list[ChunkSearchHit]:
            del query_text, size, viewer_user_id, retrieval_mode
            return []

        def search_with_trace(
            self,
            query_text: str,
            size: int,
            *,
            viewer_user_id: str = "",
            retrieval_mode: str = "search",
        ) -> ChunkSearchExecutionResult:
            del query_text, size, viewer_user_id
            return ChunkSearchExecutionResult(
                hits=[],
                query_intent="general",
                retrieval_trace=RetrievalTrace(
                    mode=retrieval_mode,
                    query_intent="general",
                    lane_count=1,
                    final_hit_count=0,
                    evidence_sufficient=False,
                    evidence_reason="no_evidence",
                    lanes=[],
                ),
                evidence_assessment=EvidenceAssessment(
                    sufficient=False,
                    reason_code="no_evidence",
                    citation_count=0,
                    top_score=None,
                ),
            )

    service = ChatService(
        chat_client=_ExplodingChatClient(),  # type: ignore[arg-type]
        chunk_search_service=_NoEvidenceSearchService(),  # type: ignore[arg-type]
        system_prompt="You are a test assistant.",
    )

    events = list(
        service.stream(
            [ChatMessage(role="user", content="系统里有这个制度吗？")],
            retrieval_size=3,
        )
    )

    assert [event.event for event in events] == ["context", "delta", "done"]
    assert events[1].data["content"] == ChatService._FALLBACK_ANSWER  # pyright: ignore[reportPrivateUsage]
    assert events[2].data["finish_reason"] == "evidence_insufficient"
