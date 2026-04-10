from __future__ import annotations

from dataclasses import dataclass, field

from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest
from baozhi_rag.services.retrieval_pipeline import RetrievalPipelineService
from baozhi_rag.services.retrieval_plan import RetrievalLanePlan, RetrievalPlan
from baozhi_rag.services.term_matching import TermMatchResult


class _StubTermMatcher:
    def extract_terms(self, text: str) -> TermMatchResult:
        return TermMatchResult(merged_terms=["term-a"])


@dataclass
class _StubStore:
    requests: list[ChunkSearchRequest] = field(default_factory=list)

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        self.requests.append(request)
        return [
            ChunkSearchHit(
                chunk_id=f"{request.query_text}-{len(self.requests)}",
                file_id="file-1",
                chunk_type="text",
                segment_id="seg-1",
                source_filename="guide.md",
                storage_key="guide.md",
                chunk_index=0,
                char_count=10,
                content="content",
                merged_terms=[],
                score=1.0,
            )
        ]


@dataclass
class _StubEmbeddingService:
    calls: list[str] = field(default_factory=list)

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1, 0.2]


def test_retrieval_pipeline_executes_multiple_lanes_and_caches_embeddings() -> None:
    store = _StubStore()
    embedding_service = _StubEmbeddingService()
    pipeline = RetrievalPipelineService(
        term_matcher=_StubTermMatcher(),  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        chunk_embedding_service=embedding_service,  # type: ignore[arg-type]
    )

    plan = RetrievalPlan(
        mode="chat",
        query_text="什么是向量检索",
        query_intent="definition",
        result_size=3,
        lanes=[
            RetrievalLanePlan(
                lane_id="lane-1",
                query_text="什么是向量检索",
                lane_weight=1.0,
                result_size=3,
                lexical_candidate_size=10,
                vector_candidate_size=10,
                query_intent="definition",
            ),
            RetrievalLanePlan(
                lane_id="lane-2",
                query_text="什么是向量检索",
                lane_weight=0.8,
                result_size=3,
                lexical_candidate_size=10,
                vector_candidate_size=10,
                query_intent="definition",
            ),
        ],
    )

    execution_result = pipeline.execute(retrieval_plan=plan, viewer_user_id="user-1")

    assert len(store.requests) == 2
    assert embedding_service.calls == ["什么是向量检索"]
    assert len(execution_result.hits) == 2
    assert len(execution_result.lane_executions) == 2


def test_retrieval_pipeline_skips_embedding_when_vector_lane_disabled() -> None:
    store = _StubStore()
    embedding_service = _StubEmbeddingService()
    pipeline = RetrievalPipelineService(
        term_matcher=_StubTermMatcher(),  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        chunk_embedding_service=embedding_service,  # type: ignore[arg-type]
    )

    plan = RetrievalPlan(
        mode="search",
        query_text="这个内容在哪一章",
        query_intent="document_location",
        result_size=3,
        lanes=[
            RetrievalLanePlan(
                lane_id="lexical-only",
                query_text="这个内容在哪一章",
                lane_weight=1.0,
                result_size=3,
                lexical_candidate_size=10,
                vector_candidate_size=0,
                lexical_rrf_weight=1.0,
                vector_rrf_weight=0.0,
                query_intent="document_location",
            )
        ],
    )

    execution_result = pipeline.execute(retrieval_plan=plan, viewer_user_id="user-1")

    assert len(store.requests) == 1
    assert embedding_service.calls == []
    assert execution_result.plan.mode == "search"
