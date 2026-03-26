"""文档切块检索字段测试。"""

from __future__ import annotations

from pathlib import Path

from docx import Document

from baozhi_rag.services.document_chunking import DocumentChunkService


def test_chunk_includes_search_terms_and_preview_document(tmp_path: Path) -> None:
    """切块结果和 ES 预览文档都应包含领域词字段。"""
    file_path = tmp_path / "terms.docx"
    document = Document()
    document.add_paragraph("保险责任免除和免赔额说明。")
    document.save(str(file_path))

    service = DocumentChunkService(chunk_size=200, chunk_overlap=20, convert_temp_dir=tmp_path)
    chunks = service.chunk_document(
        file_path=file_path,
        source_filename="terms.docx",
        storage_key="2026/03/26/terms.docx",
        file_id="file-terms",
    )

    assert len(chunks) == 1
    assert chunks[0].fmm_terms == ["保险责任", "免赔额"]
    assert chunks[0].bmm_terms == ["保险", "责任免除", "免赔额"]
    assert chunks[0].merged_terms == ["保险责任", "免赔额", "保险", "责任免除"]

    preview_document = service._build_es_preview_document(chunks[0])

    assert preview_document["chunk_id"] == "file-terms-chunk-0"
    assert preview_document["fmm_terms"] == ["保险责任", "免赔额"]
    assert preview_document["bmm_terms"] == ["保险", "责任免除", "免赔额"]
    assert preview_document["merged_terms"] == ["保险责任", "免赔额", "保险", "责任免除"]
