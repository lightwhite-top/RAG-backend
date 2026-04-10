from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from baozhi_rag.api.dependencies import (
    get_aliyun_oss_file_store,
    get_auth_service,
    get_chat_service,
    get_chat_session_service,
    get_chunk_search_service,
    get_conversation_chat_service,
    get_current_user,
    get_knowledge_file_delete_service,
    get_knowledge_file_query_service,
    get_knowledge_upload_service,
    get_settings,
    get_user_admin_service,
)
from baozhi_rag.app.main import create_app
from baozhi_rag.core.config import Settings
from baozhi_rag.domain.chat_message import (
    ChatMessageCitationRecord,
    ChatMessageRecord,
    ChatMessageRole,
    ChatMessageStatus,
)
from baozhi_rag.domain.chat_session import ChatSession, ChatSessionListPage, ChatSessionStatus
from baozhi_rag.domain.knowledge_file import FileStorageProvider, FileVisibilityScope
from baozhi_rag.domain.knowledge_upload_task import (
    KnowledgeUploadTask,
    KnowledgeUploadTaskStage,
    KnowledgeUploadTaskStatus,
)
from baozhi_rag.domain.user import CurrentUser, UserRole
from baozhi_rag.services.auth import LoginResult, SendRegistrationCodeResult
from baozhi_rag.services.chat import (
    ChatCitation,
    ChatCompletionResult,
    ChatContentBlock,
    ChatStreamEvent,
)
from baozhi_rag.services.chat_sessions import ChatSessionMessageHistoryResult
from baozhi_rag.services.chunk_search import ChunkSearchExecutionResult, ChunkSearchHit
from baozhi_rag.services.document_chunking import ChunkImageAsset
from baozhi_rag.services.evidence_sufficiency import EvidenceAssessment
from baozhi_rag.services.knowledge_file_query import (
    KnowledgeFileListItemResult,
    KnowledgeFileListResult,
)
from baozhi_rag.services.retrieval_trace import RetrievalTrace, RetrievalTraceLane
from baozhi_rag.services.user_admin import UserListResult

NOW = datetime(2026, 4, 7, 10, 0, tzinfo=UTC)
_NO_OVERRIDE = object()


def _build_current_user(
    *,
    user_id: str,
    email: str,
    username: str,
    role: UserRole,
) -> CurrentUser:
    return CurrentUser(
        id=user_id,
        email=email,
        username=username,
        role=role,
        created_at=NOW,
        updated_at=NOW,
    )


def _build_send_registration_code_result() -> SendRegistrationCodeResult:
    return SendRegistrationCodeResult(
        expires_in_seconds=600,
        expires_at=NOW + timedelta(minutes=10),
        resend_interval_seconds=60,
    )


def _build_login_result() -> LoginResult:
    return LoginResult(
        access_token="token-123",
        token_type="Bearer",
        expires_in_seconds=604800,
        expires_at=NOW + timedelta(days=7),
        user=_build_current_user(
            user_id="user-1",
            email="user@example.com",
            username="Light",
            role=UserRole.USER,
        ),
    )


def _build_upload_task(
    *,
    task_id: str = "task-1",
    status: KnowledgeUploadTaskStatus = KnowledgeUploadTaskStatus.QUEUED,
    stage: KnowledgeUploadTaskStage = KnowledgeUploadTaskStage.UPLOADED,
    file_id: str | None = None,
) -> KnowledgeUploadTask:
    return KnowledgeUploadTask(
        id=task_id,
        request_id="req-upload",
        uploader_user_id="user-1",
        uploader_role=UserRole.USER.value,
        raw_sha256="raw-sha",
        source_storage_key="tmp/source.docx",
        requested_filename="policy.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        size=256,
        ingest_version="v1",
        status=status,
        stage=stage,
        content_sha256=None,
        file_id=file_id,
        chunk_count=0,
        deduplicated=False,
        replaced=False,
        title_updated=False,
        error_code=None,
        error_message=None,
        attempt_count=0,
        worker_id=None,
        lease_expires_at=None,
        last_heartbeat_at=None,
        created_at=NOW,
        updated_at=NOW,
        completed_at=None,
    )


def _build_file_list_result() -> KnowledgeFileListResult:
    return KnowledgeFileListResult(
        items=[
            KnowledgeFileListItemResult(
                file_id="file-1",
                uploader_user_id="user-1",
                original_filename="policy.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size=256,
                storage_key="knowledge-files/user-1/file-1/policy.docx",
                file_url="https://example.com/policy.docx",
                storage_provider=FileStorageProvider.ALIYUN_OSS.value,
                visibility_scope=FileVisibilityScope.OWNER_ONLY.value,
                chunk_count=2,
                uploaded_at=NOW,
                updated_at=NOW,
            )
        ],
        total=1,
        page=1,
        page_size=20,
    )


def _build_search_hit() -> ChunkSearchHit:
    return ChunkSearchHit(
        chunk_id="file-1-chunk-0",
        file_id="file-1",
        chunk_type="text",
        segment_id="seg-1",
        source_filename="policy.docx",
        storage_key="knowledge-files/user-1/file-1/policy.docx",
        chunk_index=0,
        char_count=42,
        heading_path=["产品说明", "免赔规则"],
        section_title="免赔规则",
        content_type="paragraph",
        content="Deductible details and coverage notes.",
        merged_terms=["deductible", "coverage"],
        score=0.91,
        uploader_user_id="user-1",
        visibility_scope=FileVisibilityScope.OWNER_ONLY.value,
    )


def _build_chat_result() -> ChatCompletionResult:
    citation = ChatCitation(
        citation_id="cit-1",
        chunk_id="file-1-chunk-0",
        file_id="file-1",
        chunk_type="text",
        segment_id="seg-1",
        source_filename="policy.docx",
        storage_key="knowledge-files/user-1/file-1/policy.docx",
        chunk_index=0,
        char_count=42,
        content="Deductible details and coverage notes.",
        merged_terms=["deductible"],
        score=0.91,
        snippet="Deductible details and coverage notes.",
        heading_path=[],
        section_title=None,
        content_type="paragraph",
        source_anchor="chunk:0",
    )
    return ChatCompletionResult(
        answer="A deductible is the amount paid by the insured first.[1]",
        plain_text="A deductible is the amount paid by the insured first.",
        content_blocks=[
            ChatContentBlock(
                block_id="blk-1",
                block_type="markdown",
                text="A deductible is the amount paid by the insured first.",
                citation_ids=["cit-1"],
                sequence=1,
            )
        ],
        original_query="What is a deductible?",
        retrieval_query="What is a deductible?",
        citations=[citation],
        finish_reason="stop",
        rewrite_applied=False,
        query_intent="definition",
        retrieval_trace=RetrievalTrace(
            mode="chat",
            query_intent="definition",
            lane_count=3,
            final_hit_count=1,
            evidence_sufficient=True,
            evidence_reason="sufficient",
            lanes=[
                RetrievalTraceLane(
                    lane_id="hybrid-default",
                    query_text="What is a deductible?",
                    lane_weight=1.0,
                    result_count=1,
                    top_chunk_ids=["file-1-chunk-0"],
                )
            ],
        ),
        evidence_assessment=EvidenceAssessment(
            sufficient=True,
            reason_code="sufficient",
            citation_count=1,
            top_score=0.91,
        ),
    )


def _build_image_asset() -> ChunkImageAsset:
    return ChunkImageAsset(
        segment_id="seg-1",
        asset_id="img-1",
        asset_index=1,
        source_anchor="p:1:image:1",
        content_type="image/png",
        extension=".png",
        image_bytes=None,
        storage_key="knowledge-files/user-1/file-1/assets/images/img-1.png",
        thumbnail_storage_key="knowledge-files/user-1/file-1/assets/thumbnails/img-1.png",
        summary="A simple process diagram",
        ocr_text="apply -> review -> approve",
        image_type="diagram",
    )


def _build_chat_session() -> ChatSession:
    return ChatSession(
        id="sess-1",
        owner_user_id="user-1",
        title="理赔等待期咨询",
        status=ChatSessionStatus.ACTIVE,
        message_count=2,
        summary_version=0,
        last_message_at=NOW,
        last_user_message_at=NOW,
        last_assistant_message_at=NOW,
        deleted_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _build_chat_session_list_page() -> ChatSessionListPage:
    return ChatSessionListPage(
        items=[_build_chat_session()],
        total=1,
        page=1,
        page_size=20,
    )


def _build_session_chat_result() -> ChatCompletionResult:
    result = _build_chat_result()
    return ChatCompletionResult(
        answer=result.answer,
        plain_text=result.plain_text,
        content_blocks=result.content_blocks,
        original_query=result.original_query,
        retrieval_query=result.retrieval_query,
        citations=result.citations,
        finish_reason=result.finish_reason,
        rewrite_applied=result.rewrite_applied,
        message_id="msg-session-2",
        session_id="sess-1",
        sequence_no=2,
        created_at=NOW,
        completed_at=NOW,
    )


@dataclass
class StubObjectStore:
    def build_presigned_get_url(
        self,
        *,
        storage_key: str,
        expires_seconds: int = 900,
    ) -> str:
        del expires_seconds
        return f"https://example.com/{storage_key}"


def _build_history_result() -> ChatSessionMessageHistoryResult:
    citation = ChatMessageCitationRecord(
        id="cit-1",
        message_id="msg-2",
        citation_index=1,
        chunk_id="file-1-chunk-0",
        file_id="file-1",
        source_filename="policy.docx",
        storage_key="knowledge-files/user-1/file-1/policy.docx",
        chunk_index=0,
        char_count=42,
        content="Deductible details and coverage notes.",
        snippet="Deductible details and coverage notes.",
        merged_terms=["deductible"],
        score=0.91,
        heading_path=[],
        section_title=None,
        content_type="paragraph",
        source_anchor="chunk:0",
        created_at=NOW,
    )
    return ChatSessionMessageHistoryResult(
        session=_build_chat_session(),
        items=[
            ChatMessageRecord(
                id="msg-1",
                session_id="sess-1",
                sequence_no=1,
                role=ChatMessageRole.USER,
                status=ChatMessageStatus.COMPLETED,
                plain_text="What is a deductible?",
                content_blocks=[],
                request_id="req-chat",
                model_name=None,
                original_query=None,
                retrieval_query=None,
                rewrite_applied=False,
                retrieval_size=None,
                temperature=None,
                finish_reason=None,
                latency_ms=None,
                usage=None,
                error_code=None,
                error_message=None,
                created_at=NOW,
                updated_at=NOW,
                completed_at=NOW,
            ),
            ChatMessageRecord(
                id="msg-2",
                session_id="sess-1",
                sequence_no=2,
                role=ChatMessageRole.ASSISTANT,
                status=ChatMessageStatus.COMPLETED,
                plain_text="A deductible is the amount paid by the insured first.",
                content_blocks=[
                    {
                        "block_id": "blk-1",
                        "block_type": "markdown",
                        "text": "A deductible is the amount paid by the insured first.",
                        "citation_ids": ["cit-1"],
                        "sequence": 1,
                    }
                ],
                request_id="req-chat",
                model_name="qwen-test",
                original_query="What is a deductible?",
                retrieval_query="What is a deductible?",
                rewrite_applied=False,
                retrieval_size=4,
                temperature=0.2,
                finish_reason="stop",
                latency_ms=123,
                usage=None,
                error_code=None,
                error_message=None,
                created_at=NOW,
                updated_at=NOW,
                completed_at=NOW,
                citations=[citation],
            ),
        ],
        next_before_sequence_no=None,
    )


def _build_user_list_result() -> UserListResult:
    return UserListResult(
        items=[
            _build_current_user(
                user_id="user-1",
                email="user@example.com",
                username="Light",
                role=UserRole.USER,
            )
        ],
        total=1,
        page=1,
        page_size=20,
    )


@dataclass
class StubAuthService:
    send_registration_code_result: SendRegistrationCodeResult = field(
        default_factory=_build_send_registration_code_result
    )
    send_registration_code_error: Exception | None = None
    register_result: CurrentUser = field(
        default_factory=lambda: _build_current_user(
            user_id="user-2",
            email="new-user@example.com",
            username="New User",
            role=UserRole.USER,
        )
    )
    register_error: Exception | None = None
    login_result: LoginResult = field(default_factory=_build_login_result)
    login_error: Exception | None = None
    token_user: CurrentUser = field(
        default_factory=lambda: _build_current_user(
            user_id="user-1",
            email="user@example.com",
            username="Light",
            role=UserRole.USER,
        )
    )
    token_error: Exception | None = None
    update_profile_result: CurrentUser = field(
        default_factory=lambda: _build_current_user(
            user_id="user-1",
            email="user@example.com",
            username="Updated Name",
            role=UserRole.USER,
        )
    )
    update_profile_error: Exception | None = None
    change_password_error: Exception | None = None

    def send_registration_code(self, *, email: str) -> SendRegistrationCodeResult:
        if self.send_registration_code_error is not None:
            raise self.send_registration_code_error
        return self.send_registration_code_result

    def register(
        self,
        *,
        email: str,
        password: str,
        username: str,
        verification_code: str,
    ) -> CurrentUser:
        if self.register_error is not None:
            raise self.register_error
        return self.register_result

    def login(self, *, email: str, password: str) -> LoginResult:
        if self.login_error is not None:
            raise self.login_error
        return self.login_result

    def get_current_user_from_token(self, token: str) -> CurrentUser:
        if self.token_error is not None:
            raise self.token_error
        return self.token_user

    def update_profile(self, *, user_id: str, username: str) -> CurrentUser:
        if self.update_profile_error is not None:
            raise self.update_profile_error
        return self.update_profile_result

    def change_password(
        self,
        *,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> None:
        if self.change_password_error is not None:
            raise self.change_password_error


@dataclass
class StubKnowledgeUploadService:
    submit_result: list[KnowledgeUploadTask] = field(default_factory=lambda: [_build_upload_task()])
    submit_error: Exception | None = None
    list_result: list[KnowledgeUploadTask] = field(default_factory=lambda: [_build_upload_task()])
    list_error: Exception | None = None
    task_result: KnowledgeUploadTask = field(default_factory=_build_upload_task)
    task_error: Exception | None = None
    retry_result: KnowledgeUploadTask = field(
        default_factory=lambda: _build_upload_task(
            task_id="task-1",
            status=KnowledgeUploadTaskStatus.QUEUED,
            stage=KnowledgeUploadTaskStage.UPLOADED,
        )
    )
    retry_error: Exception | None = None

    async def submit_files(
        self,
        files: list[Any],
        *,
        current_user: CurrentUser,
        request_id: str,
    ) -> list[KnowledgeUploadTask]:
        if self.submit_error is not None:
            raise self.submit_error
        return self.submit_result

    def list_tasks(self, *, current_user: CurrentUser) -> list[KnowledgeUploadTask]:
        if self.list_error is not None:
            raise self.list_error
        return self.list_result

    def get_task(self, *, task_id: str, current_user: CurrentUser) -> KnowledgeUploadTask:
        if self.task_error is not None:
            raise self.task_error
        return self.task_result

    def retry_task(self, *, task_id: str, current_user: CurrentUser) -> KnowledgeUploadTask:
        if self.retry_error is not None:
            raise self.retry_error
        return self.retry_result


@dataclass
class StubKnowledgeFileQueryService:
    global_result: KnowledgeFileListResult = field(default_factory=_build_file_list_result)
    my_result: KnowledgeFileListResult = field(default_factory=_build_file_list_result)
    global_error: Exception | None = None
    my_error: Exception | None = None

    def list_global_files(self, *, page: int, page_size: int) -> KnowledgeFileListResult:
        if self.global_error is not None:
            raise self.global_error
        return self.global_result

    def list_my_files(
        self,
        *,
        current_user: CurrentUser,
        page: int,
        page_size: int,
    ) -> KnowledgeFileListResult:
        if self.my_error is not None:
            raise self.my_error
        return self.my_result


@dataclass
class StubKnowledgeFileDeleteService:
    delete_error: Exception | None = None
    deleted_file_ids: list[str] = field(default_factory=list)

    def delete_file(self, *, file_id: str, current_user: CurrentUser) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted_file_ids.append(file_id)


@dataclass
class StubChunkSearchService:
    hits: list[ChunkSearchHit] = field(default_factory=lambda: [_build_search_hit()])
    error: Exception | None = None

    def search(
        self,
        query_text: str,
        size: int,
        *,
        viewer_user_id: str = "",
        retrieval_mode: str = "search",
    ) -> list[ChunkSearchHit]:
        if self.error is not None:
            raise self.error
        return self.hits

    def search_with_trace(
        self,
        query_text: str,
        size: int,
        *,
        viewer_user_id: str = "",
        retrieval_mode: str = "search",
    ) -> ChunkSearchExecutionResult:
        if self.error is not None:
            raise self.error
        return ChunkSearchExecutionResult(
            hits=self.hits,
            query_intent="general",
            retrieval_trace=RetrievalTrace(
                mode=retrieval_mode,
                query_intent="general",
                lane_count=1,
                final_hit_count=len(self.hits),
                evidence_sufficient=True,
                evidence_reason="sufficient",
                lanes=[],
            ),
            evidence_assessment=EvidenceAssessment(
                sufficient=True,
                reason_code="sufficient",
                citation_count=len(self.hits),
                top_score=self.hits[0].score if self.hits else None,
            ),
        )


@dataclass
class StubChatService:
    complete_result: ChatCompletionResult = field(default_factory=_build_chat_result)
    complete_error: Exception | None = None
    stream_events: list[ChatStreamEvent | Exception] = field(
        default_factory=lambda: [
            ChatStreamEvent(
                event="context",
                data={
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "rewrite_applied": False,
                    "citations": [],
                },
            ),
            ChatStreamEvent(event="delta", data={"content": "Hello"}),
            ChatStreamEvent(
                event="done",
                data={
                    "answer": "Hello",
                    "plain_text": "Hello",
                    "content_blocks": [],
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "citations": [],
                    "finish_reason": "stop",
                    "rewrite_applied": False,
                },
            ),
        ]
    )

    def complete(
        self,
        messages: list[Any],
        *,
        retrieval_size: int,
        temperature: float | None = None,
        viewer_user_id: str = "",
    ) -> ChatCompletionResult:
        if self.complete_error is not None:
            raise self.complete_error
        return self.complete_result

    def stream(
        self,
        messages: list[Any],
        *,
        retrieval_size: int,
        temperature: float | None = None,
        viewer_user_id: str = "",
    ) -> Iterator[ChatStreamEvent]:
        for item in self.stream_events:
            if isinstance(item, Exception):
                raise item
            yield item


@dataclass
class StubChatSessionService:
    create_result: ChatSession = field(default_factory=_build_chat_session)
    create_error: Exception | None = None
    list_result: ChatSessionListPage = field(default_factory=_build_chat_session_list_page)
    list_error: Exception | None = None
    get_result: ChatSession = field(default_factory=_build_chat_session)
    get_error: Exception | None = None
    update_result: ChatSession = field(default_factory=_build_chat_session)
    update_error: Exception | None = None
    delete_error: Exception | None = None
    history_result: ChatSessionMessageHistoryResult = field(default_factory=_build_history_result)
    history_error: Exception | None = None
    deleted_session_ids: list[str] = field(default_factory=list)

    def create_session(self, *, current_user: CurrentUser, title: str | None) -> ChatSession:
        if self.create_error is not None:
            raise self.create_error
        return self.create_result

    def list_sessions(
        self,
        *,
        current_user: CurrentUser,
        page: int,
        page_size: int,
        status: ChatSessionStatus | None,
    ) -> ChatSessionListPage:
        if self.list_error is not None:
            raise self.list_error
        return self.list_result

    def get_session(self, *, session_id: str, current_user: CurrentUser) -> ChatSession:
        if self.get_error is not None:
            raise self.get_error
        return self.get_result

    def update_session(
        self,
        *,
        session_id: str,
        current_user: CurrentUser,
        title: str | None = None,
        status: ChatSessionStatus | None = None,
    ) -> ChatSession:
        if self.update_error is not None:
            raise self.update_error
        return self.update_result

    def delete_session(self, *, session_id: str, current_user: CurrentUser) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted_session_ids.append(session_id)

    def list_messages(
        self,
        *,
        session_id: str,
        current_user: CurrentUser,
        before_sequence_no: int | None,
        limit: int,
    ) -> ChatSessionMessageHistoryResult:
        if self.history_error is not None:
            raise self.history_error
        return self.history_result


@dataclass
class StubConversationChatService:
    complete_result: ChatCompletionResult = field(default_factory=_build_session_chat_result)
    complete_error: Exception | None = None
    stream_events: list[ChatStreamEvent | Exception] = field(
        default_factory=lambda: [
            ChatStreamEvent(
                event="context",
                data={
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "rewrite_applied": False,
                    "citations": [],
                    "message_id": "msg-session-2",
                    "session_id": "sess-1",
                    "sequence_no": 2,
                },
            ),
            ChatStreamEvent(event="delta", data={"content": "Hello"}),
            ChatStreamEvent(
                event="done",
                data={
                    "answer": "Hello",
                    "plain_text": "Hello",
                    "content_blocks": [],
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "citations": [],
                    "finish_reason": "stop",
                    "rewrite_applied": False,
                    "message_id": "msg-session-2",
                    "session_id": "sess-1",
                    "sequence_no": 2,
                    "created_at": NOW.isoformat(),
                    "completed_at": NOW.isoformat(),
                },
            ),
        ]
    )

    def complete(
        self,
        *,
        session_id: str,
        messages: list[Any],
        retrieval_size: int,
        temperature: float | None,
        current_user: CurrentUser,
        request_id: str,
    ) -> ChatCompletionResult:
        if self.complete_error is not None:
            raise self.complete_error
        return self.complete_result

    def stream(
        self,
        *,
        session_id: str,
        messages: list[Any],
        retrieval_size: int,
        temperature: float | None,
        current_user: CurrentUser,
        request_id: str,
    ) -> Iterator[ChatStreamEvent]:
        for item in self.stream_events:
            if isinstance(item, Exception):
                raise item
            yield item


@dataclass
class StubUserAdminService:
    list_result: UserListResult = field(default_factory=_build_user_list_result)
    list_error: Exception | None = None
    get_result: CurrentUser = field(
        default_factory=lambda: _build_current_user(
            user_id="user-2",
            email="detail@example.com",
            username="Detail User",
            role=UserRole.USER,
        )
    )
    get_error: Exception | None = None
    create_result: CurrentUser = field(
        default_factory=lambda: _build_current_user(
            user_id="user-3",
            email="created@example.com",
            username="Created User",
            role=UserRole.USER,
        )
    )
    create_error: Exception | None = None
    update_result: CurrentUser = field(
        default_factory=lambda: _build_current_user(
            user_id="user-3",
            email="updated@example.com",
            username="Updated User",
            role=UserRole.ADMIN,
        )
    )
    update_error: Exception | None = None
    delete_error: Exception | None = None
    deleted_user_ids: list[str] = field(default_factory=list)

    def list_users(self, *, query_text: str | None, page: int, page_size: int) -> UserListResult:
        if self.list_error is not None:
            raise self.list_error
        return self.list_result

    def get_user(self, *, user_id: str) -> CurrentUser:
        if self.get_error is not None:
            raise self.get_error
        return self.get_result

    def create_user(
        self,
        *,
        email: str,
        password: str,
        username: str,
        role: UserRole,
    ) -> CurrentUser:
        if self.create_error is not None:
            raise self.create_error
        return self.create_result

    def update_user(
        self,
        *,
        user_id: str,
        email: str | None = None,
        username: str | None = None,
        role: UserRole | None = None,
        password: str | None = None,
    ) -> CurrentUser:
        if self.update_error is not None:
            raise self.update_error
        return self.update_result

    def delete_user(self, *, user_id: str) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted_user_ids.append(user_id)


@dataclass
class ApiHarness:
    client: TestClient
    settings: Settings
    auth_service: StubAuthService
    upload_service: StubKnowledgeUploadService
    file_query_service: StubKnowledgeFileQueryService
    file_delete_service: StubKnowledgeFileDeleteService
    search_service: StubChunkSearchService
    chat_service: StubChatService
    chat_session_service: StubChatSessionService
    conversation_chat_service: StubConversationChatService
    user_admin_service: StubUserAdminService
    object_store: StubObjectStore


@asynccontextmanager
async def _noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


def _build_settings() -> Settings:
    return Settings(
        _env_file=None,
        APP_NAME="RAG Test Service",
        APP_ENV="test",
        APP_DEBUG=False,
        APP_VERSION="test-1.0.0",
        SEARCH_DEFAULT_SIZE=8,
        LLM_CHAT_MODEL="qwen-test",
    )


@pytest.fixture
def standard_user() -> CurrentUser:
    return _build_current_user(
        user_id="user-1",
        email="user@example.com",
        username="Light",
        role=UserRole.USER,
    )


@pytest.fixture
def admin_user() -> CurrentUser:
    return _build_current_user(
        user_id="admin-1",
        email="admin@example.com",
        username="Admin",
        role=UserRole.ADMIN,
    )


@pytest.fixture
def build_harness() -> Iterator[Callable[..., Iterator[ApiHarness]]]:
    @contextmanager
    def _build(
        *,
        current_user: CurrentUser | object = _NO_OVERRIDE,
    ) -> Iterator[ApiHarness]:
        settings = _build_settings()
        auth_service = StubAuthService()
        upload_service = StubKnowledgeUploadService()
        file_query_service = StubKnowledgeFileQueryService()
        file_delete_service = StubKnowledgeFileDeleteService()
        search_service = StubChunkSearchService()
        chat_service = StubChatService()
        chat_session_service = StubChatSessionService()
        conversation_chat_service = StubConversationChatService()
        user_admin_service = StubUserAdminService()
        object_store = StubObjectStore()

        app = create_app(settings)
        app.router.lifespan_context = _noop_lifespan
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_auth_service] = lambda: auth_service
        app.dependency_overrides[get_knowledge_upload_service] = lambda: upload_service
        app.dependency_overrides[get_knowledge_file_query_service] = lambda: file_query_service
        app.dependency_overrides[get_knowledge_file_delete_service] = lambda: file_delete_service
        app.dependency_overrides[get_chunk_search_service] = lambda: search_service
        app.dependency_overrides[get_chat_service] = lambda: chat_service
        app.dependency_overrides[get_chat_session_service] = lambda: chat_session_service
        app.dependency_overrides[get_conversation_chat_service] = lambda: conversation_chat_service
        app.dependency_overrides[get_user_admin_service] = lambda: user_admin_service
        app.dependency_overrides[get_aliyun_oss_file_store] = lambda: object_store

        if current_user is not _NO_OVERRIDE:

            def _override_current_user() -> CurrentUser:
                return current_user  # type: ignore[return-value]

            app.dependency_overrides[get_current_user] = _override_current_user

        client = TestClient(app)
        try:
            yield ApiHarness(
                client=client,
                settings=settings,
                auth_service=auth_service,
                upload_service=upload_service,
                file_query_service=file_query_service,
                file_delete_service=file_delete_service,
                search_service=search_service,
                chat_service=chat_service,
                chat_session_service=chat_session_service,
                conversation_chat_service=conversation_chat_service,
                user_admin_service=user_admin_service,
                object_store=object_store,
            )
        finally:
            client.close()

    yield _build
