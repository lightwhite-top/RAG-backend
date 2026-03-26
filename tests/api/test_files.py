"""文件上传接口测试。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from docx import Document
from fastapi.testclient import TestClient

from baozhi_rag.app.main import create_app
from baozhi_rag.core.config import Settings


def test_upload_files_returns_chunk_preview_and_persists_docx(tmp_path: Path) -> None:
    """上传 docx 应返回切块预览并落盘。"""
    client = _create_upload_client(tmp_path)
    docx_bytes = _build_docx_bytes(
        paragraphs=[
            ("标题一", "Heading 1"),
            ("这是第一段的内容。" * 40, None),
            ("这是第二段的内容。" * 40, None),
        ]
    )

    response = client.post(
        "/files/upload",
        files=[
            (
                "files",
                (
                    "产品说明.docx",
                    docx_bytes,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            )
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["files"]) == 1

    file_item = payload["files"][0]

    assert file_item["original_filename"] == "产品说明.docx"
    assert (
        file_item["content_type"]
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert file_item["size"] == len(docx_bytes)
    assert file_item["file_id"]
    assert file_item["storage_key"]
    assert "T" in file_item["uploaded_at"]
    assert not Path(file_item["storage_key"]).is_absolute()
    assert (tmp_path / file_item["storage_key"]).exists()
    assert file_item["chunk_status"] == "success"
    assert file_item["chunk_count"] > 0
    assert len(file_item["chunk_preview"]) >= 1
    assert file_item["chunk_preview"][0]["chunk_index"] == 0
    assert file_item["chunk_preview"][0]["char_count"] > 0
    assert file_item["chunk_preview"][0]["preview_text"]


def test_upload_files_rejects_unsupported_format_and_rolls_back_file(tmp_path: Path) -> None:
    """上传非 Word 文件应返回错误并删除已保存文件。"""
    client = _create_upload_client(tmp_path)

    response = client.post(
        "/files/upload",
        files=[("files", ("产品说明.txt", b"plain text", "text/plain"))],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "暂不支持的文件格式: .txt"
    assert list(tmp_path.rglob("*")) == []


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
        doc_chunk_size=120,
        doc_chunk_overlap=20,
        doc_convert_temp_dir=upload_root_dir / "tmp",
    )

    return TestClient(create_app(settings))


def _build_docx_bytes(paragraphs: list[tuple[str, str | None]]) -> bytes:
    """构造测试用 docx 二进制内容。"""
    document = Document()
    for text, style in paragraphs:
        paragraph = document.add_paragraph(text)
        if style is not None:
            paragraph.style = style

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
