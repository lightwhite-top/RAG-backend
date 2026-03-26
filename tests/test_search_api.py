"""chunk 检索接口测试。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from baozhi_rag.api.dependencies import get_chunk_search_service
from baozhi_rag.app.main import create_app
from baozhi_rag.core.config import Settings
from baozhi_rag.services.chunk_search import ChunkSearchHit


class FakeChunkSearchService:
    """搜索接口测试替身。"""

    def search(self, query_text: str, size: int) -> list[ChunkSearchHit]:
        assert query_text == "免赔额"
        assert size == 5
        return [
            ChunkSearchHit(
                chunk_id="chunk-1",
                file_id="file-1",
                source_filename="保险条款.docx",
                storage_key="2026/03/26/保险条款.docx",
                chunk_index=0,
                char_count=24,
                content="本条款包含免赔额和保险责任说明。",
                fmm_terms=["免赔额", "保险责任"],
                bmm_terms=["免赔额", "保险责任"],
                merged_terms=["免赔额", "保险责任"],
                score=12.5,
            )
        ]


def test_search_chunks_returns_hits_when_es_enabled(tmp_path: Path) -> None:
    """启用 ES 时应返回 chunk 检索结果。"""
    settings = Settings(
        app_name="Baozhi RAG Test",
        app_env="test",
        debug=False,
        version="0.1.0-test",
        log_level="INFO",
        upload_root_dir=tmp_path,
        doc_chunk_size=120,
        doc_chunk_overlap=20,
        doc_convert_temp_dir=tmp_path / "tmp",
        es_enabled=True,
        search_default_size=10,
    )
    app = create_app(settings)
    app.dependency_overrides[get_chunk_search_service] = lambda: FakeChunkSearchService()

    client = TestClient(app)
    response = client.get("/search/chunks", params={"q": "免赔额", "size": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "免赔额"
    assert payload["size"] == 1
    assert payload["hits"][0]["chunk_id"] == "chunk-1"
    assert payload["hits"][0]["merged_terms"] == ["免赔额", "保险责任"]


def test_search_chunks_returns_503_when_es_disabled(tmp_path: Path) -> None:
    """未启用 ES 时检索接口应直接返回 503。"""
    settings = Settings(
        app_name="Baozhi RAG Test",
        app_env="test",
        debug=False,
        version="0.1.0-test",
        log_level="INFO",
        upload_root_dir=tmp_path,
        doc_chunk_size=120,
        doc_chunk_overlap=20,
        doc_convert_temp_dir=tmp_path / "tmp",
        es_enabled=False,
    )
    client = TestClient(create_app(settings))

    response = client.get("/search/chunks", params={"q": "免赔额"})

    assert response.status_code == 503
    assert "ES 检索未启用" in response.json()["detail"]
