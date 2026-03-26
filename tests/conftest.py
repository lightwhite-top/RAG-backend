"""测试基础设施。"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from baozhi_rag.app.main import create_app
from baozhi_rag.core.config import Settings


@pytest.fixture
def client() -> Iterator[TestClient]:
    """创建测试客户端。"""
    settings = Settings(
        app_name="Baozhi RAG Test",
        app_env="test",
        debug=False,
        version="0.1.0-test",
        log_level="INFO",
    )
    app = create_app(settings)

    with TestClient(app) as test_client:
        yield test_client
