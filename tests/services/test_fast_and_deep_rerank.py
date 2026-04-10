from __future__ import annotations

from baozhi_rag.services.chunk_search import ChunkSearchHit
from baozhi_rag.services.deep_rerank import DeepRerankService
from baozhi_rag.services.fast_rerank import FastRerankService


class _StubDeepClient:
    def __init__(self) -> None:
        self.called = False

    def rerank(self, *, query_text: str, hits: list[ChunkSearchHit]) -> list[ChunkSearchHit]:
        self.called = True
        return list(reversed(hits))


def _build_hit(chunk_id: str, score: float, *, content_type: str = "paragraph") -> ChunkSearchHit:
    return ChunkSearchHit(
        chunk_id=chunk_id,
        file_id="file-1",
        chunk_type="text",
        segment_id=f"seg-{chunk_id}",
        source_filename="guide.md",
        storage_key="guide.md",
        chunk_index=0,
        char_count=10,
        heading_path=["配置说明"],
        section_title="配置说明",
        content_type=content_type,
        content="这里解释配置项的含义。",
        merged_terms=[],
        score=score,
    )


def test_fast_rerank_prefers_table_for_structured_queries() -> None:
    service = FastRerankService()
    paragraph_hit = _build_hit("a", 0.3, content_type="paragraph")
    table_hit = _build_hit("b", 0.3, content_type="table")

    reranked = service.rerank(
        query_text="表格里第二列是什么意思",
        query_intent="structured",
        hits=[paragraph_hit, table_hit],
    )

    assert reranked[0].chunk_id == "b"


def test_deep_rerank_triggers_when_top_scores_too_close() -> None:
    client = _StubDeepClient()
    service = DeepRerankService(
        client=client,  # type: ignore[arg-type]
        score_threshold=0.2,
        margin_threshold=0.05,
    )
    hits = [_build_hit("a", 0.4), _build_hit("b", 0.38)]

    reranked, triggered = service.rerank(query_text="test", hits=hits)

    assert triggered is True
    assert client.called is True
    assert reranked[0].chunk_id == "b"


def test_deep_rerank_skips_when_scores_are_confident() -> None:
    client = _StubDeepClient()
    service = DeepRerankService(
        client=client,  # type: ignore[arg-type]
        score_threshold=0.2,
        margin_threshold=0.05,
    )
    hits = [_build_hit("a", 0.8), _build_hit("b", 0.6)]

    reranked, triggered = service.rerank(query_text="test", hits=hits)

    assert triggered is False
    assert client.called is False
    assert reranked == hits
