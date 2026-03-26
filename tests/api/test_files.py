"""文件上传接口测试。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from baozhi_rag.app.main import create_app
from baozhi_rag.core.config import Settings


def test_upload_files_returns_metadata_and_persists_files(tmp_path: Path) -> None:
    """批量上传应返回元数据并落盘。"""
    client = _create_upload_client(tmp_path)

    response = client.post(
        "/files/upload",
        files=[
            ("files", ("产品说明.txt", "保险责任说明".encode(), "text/plain")),
            ("files", ("claim form?.pdf", b"pdf-content", "application/pdf")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["files"]) == 2

    first_file = payload["files"][0]
    second_file = payload["files"][1]

    assert first_file["original_filename"] == "产品说明.txt"
    assert first_file["content_type"] == "text/plain"
    assert first_file["size"] == len("保险责任说明".encode())
    assert second_file["original_filename"] == "claim form?.pdf"
    assert second_file["content_type"] == "application/pdf"
    assert second_file["size"] == len(b"pdf-content")

    for file_item in payload["files"]:
        assert file_item["file_id"]
        assert file_item["storage_key"]
        assert "T" in file_item["uploaded_at"]
        assert not Path(file_item["storage_key"]).is_absolute()
        assert (tmp_path / file_item["storage_key"]).exists()


def test_upload_files_requires_files_field(tmp_path: Path) -> None:
    """未传 files 字段时应返回 422。"""
    client = _create_upload_client(tmp_path)

    response = client.post("/files/upload")

    assert response.status_code == 422


def _create_upload_client(upload_root_dir: Path) -> TestClient:
    """创建带临时上传目录的测试客户端。"""
    settings = Settings(
        app_name="Baozhi RAG Test",
        app_env="test",
        debug=False,
        version="0.1.0-test",
        log_level="INFO",
        upload_root_dir=upload_root_dir,
    )

    return TestClient(create_app(settings))
