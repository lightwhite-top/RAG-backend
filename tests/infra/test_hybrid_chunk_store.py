from __future__ import annotations

from dataclasses import dataclass

from baozhi_rag.infra.retrieval.hybrid_chunk_store import HybridChunkStore
from baozhi_rag.infra.retrieval.milvus_chunk_vector_store import MilvusVectorSearchHit
from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest


@dataclass
class _StubDocumentStore:
    lexical_hits: list[ChunkSearchHit]
    semantic_only_hits: list[ChunkSearchHit]
    last_request: ChunkSearchRequest | None = None

    def ensure_ready(self) -> None:  # pragma: no cover - 协议占位
        return None

    def ensure_index(self) -> None:  # pragma: no cover - 协议占位
        return None

    def index_chunks(self, chunks: list[object]) -> int:  # pragma: no cover - 协议占位
        return len(chunks)

    def delete_chunks_by_file_id(self, file_id: str) -> None:  # pragma: no cover - 协议占位
        return None

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        self.last_request = request
        return list(self.lexical_hits)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[ChunkSearchHit]:
        hit_map = {hit.chunk_id: hit for hit in self.semantic_only_hits}
        return [hit_map[chunk_id] for chunk_id in chunk_ids if chunk_id in hit_map]


@dataclass
class _StubVectorStore:
    semantic_hits: list[MilvusVectorSearchHit]
    last_size: int = 0

    def ensure_ready(self) -> None:  # pragma: no cover - 协议占位
        return None

    def ensure_collection(self) -> None:  # pragma: no cover - 协议占位
        return None

    def index_chunks(self, chunks: list[object]) -> int:  # pragma: no cover - 协议占位
        return len(chunks)

    def delete_chunks_by_file_id(self, file_id: str) -> None:  # pragma: no cover - 协议占位
        return None

    def search(
        self,
        query_embedding: list[float],
        size: int,
        *,
        viewer_user_id: str = "",
    ) -> list[MilvusVectorSearchHit]:
        self.last_size = size
        return list(self.semantic_hits)


def _build_hit(
    chunk_id: str,
    *,
    file_id: str = "file-1",
    source_filename: str = "guide.md",
) -> ChunkSearchHit:
    return ChunkSearchHit(
        chunk_id=chunk_id,
        file_id=file_id,
        chunk_type="text",
        segment_id=f"seg-{chunk_id}",
        source_filename=source_filename,
        storage_key=source_filename,
        page_number=None,
        source_anchor=None,
        chunk_index=0,
        char_count=10,
        content=f"content-{chunk_id}",
        merged_terms=[],
        score=1.0,
    )


def _build_vector_hit(chunk_id: str) -> MilvusVectorSearchHit:
    return MilvusVectorSearchHit(
        chunk_id=chunk_id,
        score=1.0,
    )


def test_hybrid_chunk_store_expands_candidate_size_before_fusion() -> None:
    document_store = _StubDocumentStore(
        lexical_hits=[_build_hit("a")],
        semantic_only_hits=[],
    )
    vector_store = _StubVectorStore(semantic_hits=[_build_vector_hit("a")])
    store = HybridChunkStore(
        document_store=document_store,
        vector_store=vector_store,
        lexical_candidate_size=20,
        vector_candidate_size=30,
        lexical_rrf_weight=1.0,
        vector_rrf_weight=1.0,
    )

    store.search(
        ChunkSearchRequest(
            query_text="test",
            size=5,
            merged_terms=[],
            query_embedding=[0.1],
            query_intent="general",
        )
    )

    assert document_store.last_request is not None
    assert document_store.last_request.size == 20
    assert vector_store.last_size == 30


def test_hybrid_chunk_store_uses_intent_based_weights_for_fusion() -> None:
    lexical_hits = [_build_hit("a"), _build_hit("b")]
    semantic_only_hits = [_build_hit("b")]
    semantic_hits = [_build_vector_hit("b"), _build_vector_hit("a")]
    document_store = _StubDocumentStore(
        lexical_hits=lexical_hits,
        semantic_only_hits=semantic_only_hits,
    )
    vector_store = _StubVectorStore(semantic_hits=semantic_hits)
    store = HybridChunkStore(
        document_store=document_store,
        vector_store=vector_store,
        lexical_candidate_size=2,
        vector_candidate_size=2,
        lexical_rrf_weight=1.0,
        vector_rrf_weight=1.0,
    )

    location_hits = store.search(
        ChunkSearchRequest(
            query_text="在哪一章",
            size=2,
            merged_terms=[],
            query_embedding=[0.1],
            query_intent="document_location",
        )
    )
    definition_hits = store.search(
        ChunkSearchRequest(
            query_text="什么是",
            size=2,
            merged_terms=[],
            query_embedding=[0.1],
            query_intent="definition",
        )
    )

    assert location_hits[0].chunk_id == "a"
    assert definition_hits[0].chunk_id == "b"


def test_hybrid_chunk_store_collapses_duplicate_chunks_from_same_file() -> None:
    document_store = _StubDocumentStore(
        lexical_hits=[
            _build_hit("a1", file_id="file-a", source_filename="A.docx"),
            _build_hit("a2", file_id="file-a", source_filename="A.docx"),
            _build_hit("b1", file_id="file-b", source_filename="B.docx"),
        ],
        semantic_only_hits=[],
    )
    vector_store = _StubVectorStore(semantic_hits=[])
    store = HybridChunkStore(
        document_store=document_store,
        vector_store=vector_store,
        lexical_candidate_size=10,
        vector_candidate_size=10,
        lexical_rrf_weight=1.0,
        vector_rrf_weight=1.0,
    )

    hits = store.search(
        ChunkSearchRequest(
            query_text="test",
            size=5,
            merged_terms=[],
            query_embedding=[0.1],
            query_intent="general",
        )
    )

    assert [hit.file_id for hit in hits] == ["file-a", "file-b"]


def test_hybrid_chunk_store_keeps_multiple_chunks_from_same_file_for_chat_mode() -> None:
    document_store = _StubDocumentStore(
        lexical_hits=[
            _build_hit("a1", file_id="file-a", source_filename="A.docx"),
            _build_hit("a2", file_id="file-a", source_filename="A.docx"),
            _build_hit("a3", file_id="file-a", source_filename="A.docx"),
            _build_hit("b1", file_id="file-b", source_filename="B.docx"),
        ],
        semantic_only_hits=[],
    )
    vector_store = _StubVectorStore(semantic_hits=[])
    store = HybridChunkStore(
        document_store=document_store,
        vector_store=vector_store,
        lexical_candidate_size=10,
        vector_candidate_size=10,
        lexical_rrf_weight=1.0,
        vector_rrf_weight=1.0,
    )

    hits = store.search(
        ChunkSearchRequest(
            query_text="test",
            size=5,
            merged_terms=[],
            query_embedding=[0.1],
            query_intent="general",
            retrieval_mode="chat",
        )
    )

    assert [hit.chunk_id for hit in hits] == ["a1", "a2", "b1"]


def test_hybrid_chunk_store_prefers_latest_version_chain_by_default() -> None:
    document_store = _StubDocumentStore(
        lexical_hits=[
            _build_hit("v4", file_id="file-v4", source_filename="149_退款规则_版本链_V4.docx"),
            _build_hit("v5", file_id="file-v5", source_filename="150_退款规则_版本链_V5.docx"),
        ],
        semantic_only_hits=[],
    )
    vector_store = _StubVectorStore(semantic_hits=[])
    store = HybridChunkStore(
        document_store=document_store,
        vector_store=vector_store,
        lexical_candidate_size=10,
        vector_candidate_size=10,
        lexical_rrf_weight=1.0,
        vector_rrf_weight=1.0,
    )

    hits = store.search(
        ChunkSearchRequest(
            query_text="普通商品签收后 7 天内可退款",
            size=5,
            merged_terms=[],
            query_embedding=[0.1],
            query_intent="general",
        )
    )

    assert len(hits) == 1
    assert hits[0].source_filename == "150_退款规则_版本链_V5.docx"


def test_hybrid_chunk_store_prefers_old_version_chain_when_query_mentions_old() -> None:
    document_store = _StubDocumentStore(
        lexical_hits=[
            _build_hit("v1", file_id="file-v1", source_filename="146_退款规则_版本链_V1.docx"),
            _build_hit("v5", file_id="file-v5", source_filename="150_退款规则_版本链_V5.docx"),
        ],
        semantic_only_hits=[],
    )
    vector_store = _StubVectorStore(semantic_hits=[])
    store = HybridChunkStore(
        document_store=document_store,
        vector_store=vector_store,
        lexical_candidate_size=10,
        vector_candidate_size=10,
        lexical_rrf_weight=1.0,
        vector_rrf_weight=1.0,
    )

    hits = store.search(
        ChunkSearchRequest(
            query_text="旧版退款规则是几天内可退款",
            size=5,
            merged_terms=[],
            query_embedding=[0.1],
            query_intent="general",
        )
    )

    assert len(hits) == 1
    assert hits[0].source_filename == "146_退款规则_版本链_V1.docx"


def test_hybrid_chunk_store_prefers_explicit_version_anchor_over_newer_default() -> None:
    document_store = _StubDocumentStore(
        lexical_hits=[
            _build_hit("v1", file_id="file-v1", source_filename="01_财务报销制度_V1.docx"),
            _build_hit("v2", file_id="file-v2", source_filename="01_财务报销制度_V2.docx"),
        ],
        semantic_only_hits=[],
    )
    vector_store = _StubVectorStore(semantic_hits=[])
    store = HybridChunkStore(
        document_store=document_store,
        vector_store=vector_store,
        lexical_candidate_size=10,
        vector_candidate_size=10,
        lexical_rrf_weight=1.0,
        vector_rrf_weight=1.0,
    )

    hits = store.search(
        ChunkSearchRequest(
            query_text="01_财务报销制度_V1.docx 中的单笔差旅自动审批上限是什么？",
            size=5,
            merged_terms=[],
            query_embedding=[0.1],
            query_intent="definition",
        )
    )

    assert len(hits) == 1
    assert hits[0].source_filename == "01_财务报销制度_V1.docx"
