"""Word 文档切块服务测试。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from docx import Document

from baozhi_rag.services.document_chunking import (
    DocumentChunkService,
    UnsupportedDocumentTypeError,
)


def test_chunk_docx_ignores_empty_paragraphs_and_preserves_heading_context(tmp_path: Path) -> None:
    """docx 切块应忽略空段落并保留标题上下文。"""
    file_path = tmp_path / "sample.docx"
    _write_docx(
        file_path,
        paragraphs=[
            ("保单说明", "Heading 1"),
            ("", None),
            ("这是正文第一段。", None),
            ("这是正文第二段。", None),
        ],
    )
    service = DocumentChunkService(chunk_size=200, chunk_overlap=20, convert_temp_dir=tmp_path)

    chunks = service.chunk_document(
        file_path=file_path,
        source_filename="sample.docx",
        storage_key="2026/03/26/sample.docx",
        file_id="file-1",
    )

    assert len(chunks) == 1
    assert "保单说明" in chunks[0].content
    assert "这是正文第一段。" in chunks[0].content
    assert "\n\n\n" not in chunks[0].content


def test_chunk_docx_applies_overlap_for_long_text(tmp_path: Path) -> None:
    """长文本切块应保留 overlap 上下文。"""
    file_path = tmp_path / "long.docx"
    long_text = "".join(str(index % 10) for index in range(260))
    _write_docx(file_path, paragraphs=[("长文本标题", "Heading 1"), (long_text, None)])
    service = DocumentChunkService(chunk_size=120, chunk_overlap=20, convert_temp_dir=tmp_path)

    chunks = service.chunk_document(
        file_path=file_path,
        source_filename="long.docx",
        storage_key="2026/03/26/long.docx",
        file_id="file-2",
    )

    assert len(chunks) >= 3
    assert chunks[0].char_count <= 120
    assert chunks[1].content[:20] == chunks[0].content[-20:]


def test_chunk_document_dispatches_doc_and_cleans_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """doc 分发应走转换逻辑并清理临时 docx。"""
    source_doc = tmp_path / "legacy.doc"
    source_doc.write_bytes(b"legacy")
    converted_docx = tmp_path / "converted.docx"
    _write_docx(converted_docx, paragraphs=[("转换后的内容", None)])
    service = DocumentChunkService(chunk_size=120, chunk_overlap=20, convert_temp_dir=tmp_path)

    def fake_convert(_: Path) -> Path:
        return converted_docx

    monkeypatch.setattr(service, "_convert_doc_to_docx", fake_convert)

    chunks = service.chunk_document(
        file_path=source_doc,
        source_filename="legacy.doc",
        storage_key="2026/03/26/legacy.doc",
        file_id="file-3",
    )

    assert len(chunks) == 1
    assert chunks[0].content == "转换后的内容"
    assert not converted_docx.exists()


def test_chunk_document_raises_for_unsupported_format(tmp_path: Path) -> None:
    """不支持的扩展名应直接报错。"""
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"pdf")
    service = DocumentChunkService(chunk_size=120, chunk_overlap=20, convert_temp_dir=tmp_path)

    with pytest.raises(UnsupportedDocumentTypeError, match=r"\.pdf"):
        service.chunk_document(
            file_path=file_path,
            source_filename="sample.pdf",
            storage_key="2026/03/26/sample.pdf",
            file_id="file-4",
        )


def test_chunk_document_logs_multiline_chunk_preview(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """切块日志应输出字段对齐的多行预览。"""
    file_path = tmp_path / "preview.docx"
    paragraphs: list[tuple[str, str | None]] = [(("第0段内容。" * 25) + "补充说明。", None)]
    _write_docx(file_path, paragraphs=paragraphs)
    service = DocumentChunkService(chunk_size=200, chunk_overlap=20, convert_temp_dir=tmp_path)

    with caplog.at_level("INFO"):
        service.chunk_document(
            file_path=file_path,
            source_filename="preview.docx",
            storage_key="2026/03/26/preview.docx",
            file_id="file-5",
        )

    assert "document_chunk_preview" in caplog.text
    assert "  file_id          : file-5" in caplog.text
    assert "  chunk_count      : 1" in caplog.text
    assert "  chunk_char_counts: [155]" in caplog.text
    assert "document_chunk_item" in caplog.text
    assert "  chunk_id         : file-5-chunk-0" in caplog.text
    assert "  char_count       : 155" in caplog.text
    assert "  content_preview  :" in caplog.text
    assert "document_chunk_item_full" not in caplog.text
    assert "document_chunk_es_preview" not in caplog.text
    assert "第0段内容。" in caplog.text


def _write_docx(file_path: Path, paragraphs: Sequence[tuple[str, str | None]]) -> None:
    """写入测试用 docx 文件。"""
    document = Document()
    for text, style in paragraphs:
        paragraph = document.add_paragraph(text)
        if style is not None:
            paragraph.style = style

    document.save(str(file_path))
