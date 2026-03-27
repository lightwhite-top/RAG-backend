"""chunk 检索服务测试。"""

from __future__ import annotations

from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest, ChunkSearchService
from baozhi_rag.services.document_chunking import DocumentChunk
from baozhi_rag.services.term_matching import build_default_term_matcher


class RecordingChunkStore:
    """记录检索请求的测试替身。"""

    def __init__(self) -> None:
        self.request: ChunkSearchRequest | None = None

    def ensure_index(self) -> None:
        return

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        return len(chunks)

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        return

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        self.request = request
        return []


class FakeEmbeddingClient:
    """返回固定查询向量的测试替身。"""

    def ensure_ready(self) -> None:
        return

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_chunk_search_service_adds_query_embedding_when_enabled() -> None:
    """启用向量化服务时应把查询向量传给存储层。"""
    store = RecordingChunkStore()
    service = ChunkSearchService(
        term_matcher=build_default_term_matcher(),
        store=store,
        chunk_embedding_service=ChunkEmbeddingService(FakeEmbeddingClient()),
    )

    hits = service.search("免赔额责任", 5)

    assert hits == []
    assert store.request is not None
    assert store.request.query_embedding == [0.1, 0.2, 0.3]
    assert store.request.merged_terms
