"""应用启动期 ES 就绪校验测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from baozhi_rag.app import main as app_main
from baozhi_rag.core.config import Settings


def _build_settings(tmp_path: Path) -> Settings:
    """构造测试用配置对象。"""
    return Settings(
        app_name="Baozhi RAG Test",
        app_env="test",
        debug=False,
        version="0.1.0-test",
        log_level="INFO",
        upload_root_dir=tmp_path,
        doc_chunk_size=120,
        doc_chunk_overlap=20,
        doc_convert_temp_dir=tmp_path / "tmp",
        chunk_embedding_enabled=False,
    )


def test_startup_success_app_starts_when_es_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当 ES readiness 成功时，应用应可正常启动。"""

    class _FakeStore:
        def ensure_ready(self) -> None:
            return

    class _FakeChunkStoreFactory:
        @classmethod
        def from_settings(cls, _: Settings) -> _FakeStore:
            return _FakeStore()

    monkeypatch.setattr(app_main, "ElasticsearchChunkStore", _FakeChunkStoreFactory)

    app = app_main.create_app(_build_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200


def test_startup_failure_es_unavailable_raises_on_lifespan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当 ES readiness 抛错时，应用启动应直接失败。"""

    class _FakeStore:
        def ensure_ready(self) -> None:
            msg = "es unavailable"
            raise RuntimeError(msg)

    class _FakeChunkStoreFactory:
        @classmethod
        def from_settings(cls, _: Settings) -> _FakeStore:
            return _FakeStore()

    monkeypatch.setattr(app_main, "ElasticsearchChunkStore", _FakeChunkStoreFactory)

    app = app_main.create_app(_build_settings(tmp_path))
    with pytest.raises(RuntimeError, match="es unavailable"), TestClient(app):
        pass


def test_startup_embedding_enabled_requires_bailian_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启用向量化时，应用启动应先校验百炼客户端配置。"""

    class _FakeStore:
        def ensure_ready(self) -> None:
            return

    class _FakeChunkStoreFactory:
        @classmethod
        def from_settings(cls, _: Settings) -> _FakeStore:
            return _FakeStore()

    class _FakeBailianClient:
        def ensure_ready(self) -> None:
            msg = "missing api key"
            raise RuntimeError(msg)

    class _FakeBailianClientFactory:
        @classmethod
        def from_settings(cls, _: Settings) -> _FakeBailianClient:
            return _FakeBailianClient()

    settings = _build_settings(tmp_path).model_copy(
        update={
            "chunk_embedding_enabled": True,
            "chunk_embedding_model": "text-embedding-v4",
            "chunk_embedding_dimensions": 1024,
        }
    )
    monkeypatch.setattr(app_main, "ElasticsearchChunkStore", _FakeChunkStoreFactory)
    monkeypatch.setattr(app_main, "AlibabaModelStudioClient", _FakeBailianClientFactory)

    app = app_main.create_app(settings)
    with pytest.raises(RuntimeError, match="missing api key"), TestClient(app):
        pass
