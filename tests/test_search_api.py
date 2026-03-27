"""chunk 检索接口测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from baozhi_rag.api.dependencies import get_chunk_search_service
from baozhi_rag.app.main import create_app
from baozhi_rag.core.config import Settings
from baozhi_rag.infra.retrieval.elasticsearch_chunk_store import (
    ElasticsearchDependencyError,
    ElasticsearchSearchError,
    ElasticsearchStoreError,
)
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


def build_test_client(tmp_path: Path, service: object) -> TestClient:
    """创建注入检索服务替身的测试客户端。"""
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
        search_default_size=10,
    )
    app = create_app(settings)
    app.dependency_overrides[get_chunk_search_service] = lambda: service
    return TestClient(app)


def test_search_chunks_returns_hits(tmp_path: Path) -> None:
    """检索服务可用时应返回 chunk 检索结果。"""
    client = build_test_client(tmp_path, FakeChunkSearchService())
    response = client.get("/search/chunks", params={"q": "免赔额", "size": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "免赔额"
    assert payload["size"] == 1
    assert payload["hits"][0]["chunk_id"] == "chunk-1"
    assert payload["hits"][0]["merged_terms"] == ["免赔额", "保险责任"]


def test_search_chunks_maps_value_error_to_400(tmp_path: Path) -> None:
    """检索参数错误应映射为 400。"""

    class ValueErrorService:
        def search(self, _: str, __: int) -> list[ChunkSearchHit]:
            raise ValueError("参数非法")

    client = build_test_client(tmp_path, ValueErrorService())

    response = client.get("/search/chunks", params={"q": "免赔额"})

    assert response.status_code == 400
    assert response.json()["detail"] == "参数非法"


def test_search_chunks_maps_dependency_error_to_500(tmp_path: Path) -> None:
    """检索依赖异常应映射为 500。"""

    class DependencyErrorService:
        def search(self, _: str, __: int) -> list[ChunkSearchHit]:
            raise ElasticsearchDependencyError("ES 依赖不可用")

    client = build_test_client(tmp_path, DependencyErrorService())

    response = client.get("/search/chunks", params={"q": "免赔额"})

    assert response.status_code == 500
    assert response.json()["detail"] == "ES 依赖不可用"


@pytest.mark.parametrize(
    "error_cls",
    [ElasticsearchSearchError, ElasticsearchStoreError],
)
def test_search_chunks_maps_search_related_error_to_502(
    tmp_path: Path,
    error_cls: type[Exception],
) -> None:
    """检索执行或存储异常应映射为 502。"""

    class SearchRelatedErrorService:
        def search(self, _: str, __: int) -> list[ChunkSearchHit]:
            raise error_cls("检索链路异常")

    client = build_test_client(tmp_path, SearchRelatedErrorService())

    response = client.get("/search/chunks", params={"q": "免赔额"})

    assert response.status_code == 502
    assert response.json()["detail"] == "检索链路异常"
