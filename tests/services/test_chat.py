from __future__ import annotations

from collections.abc import Iterator

from baozhi_rag.services.chat import ChatCompletionResult, ChatService
from baozhi_rag.services.chunk_search import ChunkSearchExecutionResult, ChunkSearchHit
from baozhi_rag.services.evidence_sufficiency import EvidenceAssessment
from baozhi_rag.services.llm import ChatMessage
from baozhi_rag.services.retrieval_trace import RetrievalTrace


class _StubChatClient:
    """返回固定聊天结果。"""

    def complete_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> str:
        return "这是回答。[1]"

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> Iterator[str]:
        yield "这是回答。[1]"


class _StubSearchService:
    """记录检索查询并返回固定命中。"""

    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_retrieval_mode: str | None = None

    def search(
        self,
        query_text: str,
        size: int,
        *,
        viewer_user_id: str = "",
        retrieval_mode: str = "search",
    ) -> list[ChunkSearchHit]:
        self.last_query = query_text
        self.last_retrieval_mode = retrieval_mode
        return [
            ChunkSearchHit(
                chunk_id="file-1-chunk-0",
                file_id="file-1",
                chunk_type="text",
                segment_id="seg-1",
                source_filename="guide.md",
                storage_key="knowledge-files/file-1/guide.md",
                chunk_index=0,
                char_count=20,
                content="认证相关错误码说明。",
                merged_terms=["认证", "错误码"],
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


def test_chat_service_rewrites_short_followup_query() -> None:
    search_service = _StubSearchService()
    service = ChatService(
        chat_client=_StubChatClient(),  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="你是测试助手。",
    )

    result = service.complete(
        [
            ChatMessage(role="user", content="接口限流窗口是什么意思"),
            ChatMessage(role="assistant", content="上一轮回答"),
            ChatMessage(role="user", content="那认证错误码呢"),
        ],
        retrieval_size=5,
    )

    assert isinstance(result, ChatCompletionResult)
    assert search_service.last_query == "接口限流窗口是什么意思。补充问题：那认证错误码呢"
    assert search_service.last_retrieval_mode == "chat"
    assert result.rewrite_applied is True


def test_chat_service_injects_markdown_render_contract_prompt() -> None:
    search_service = _StubSearchService()
    service = ChatService(
        chat_client=_StubChatClient(),  # type: ignore[arg-type]
        chunk_search_service=search_service,
        system_prompt="你是测试助手。",
    )

    prepared = service._build_model_messages(  # pyright: ignore[reportPrivateUsage]
        messages=[ChatMessage(role="user", content="什么是向量检索")],
        retrieval_query="什么是向量检索",
        citations=[],
    )

    assert len(prepared) >= 2
    assert "正文请直接输出 Markdown" in prepared[1].content
