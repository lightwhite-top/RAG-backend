"""知识文件删除接口测试。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from baozhi_rag.api.dependencies import (
    get_current_user,
    get_knowledge_file_delete_service,
)
from baozhi_rag.app.main import create_app
from baozhi_rag.core.config import Settings
from baozhi_rag.core.request_context import REQUEST_ID_HEADER_NAME
from baozhi_rag.domain.knowledge_file_errors import KnowledgeFileNotFoundError
from baozhi_rag.domain.user import CurrentUser, UserRole


def _build_test_client(tmp_path: Path, *, delete_service: object) -> TestClient:
    """创建注入删除服务替身的测试客户端。

    参数:
        tmp_path: 测试临时目录。
        delete_service: 删除服务测试替身。

    返回:
        配置好依赖覆盖的测试客户端。
    """
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
        bailian_api_key="test-api-key",
    )
    app = create_app(settings)
    app.dependency_overrides[get_knowledge_file_delete_service] = lambda: delete_service
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        id="user-test-1",
        email="user@example.com",
        username="tester",
        role=UserRole.USER,
        created_at=datetime(2026, 4, 3, tzinfo=UTC),
        updated_at=datetime(2026, 4, 3, tzinfo=UTC),
    )
    return TestClient(app)


def test_delete_my_file_returns_success_response(tmp_path: Path) -> None:
    """删除自己的知识文件时应返回统一成功响应。"""

    class DeleteService:
        def delete_file(self, *, file_id: str, current_user: CurrentUser) -> None:
            assert file_id == "file-delete-1"
            assert current_user.id == "user-test-1"

    client = _build_test_client(tmp_path, delete_service=DeleteService())

    response = client.delete("/files/file-delete-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "success"
    assert payload["message"] == "删除文件成功"
    assert payload["request_id"] == response.headers[REQUEST_ID_HEADER_NAME]
    assert payload["data"] is None


def test_delete_my_file_maps_not_found_error(tmp_path: Path) -> None:
    """删除不存在或无权访问的文件时应返回统一错误体。"""

    class DeleteService:
        def delete_file(self, *, file_id: str, current_user: CurrentUser) -> None:
            del file_id, current_user
            raise KnowledgeFileNotFoundError()

    client = _build_test_client(tmp_path, delete_service=DeleteService())

    response = client.delete("/files/file-missing")

    assert response.status_code == 404
    assert response.json()["code"] == "knowledge_file_not_found"
