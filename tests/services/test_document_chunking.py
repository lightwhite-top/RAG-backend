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


def test_parse_heading_level_by_name_custom_styles() -> None:
    """应支持通过自定义样式名识别标题（条款标题、章节名、章、节、条等）。"""
    service = DocumentChunkService(chunk_size=200, chunk_overlap=20, convert_temp_dir=Path("."))

    # 标准标题样式
    assert service._parse_heading_level_by_name("Heading 1") == 1
    assert service._parse_heading_level_by_name("heading 2") == 2
    assert service._parse_heading_level_by_name("HEADING 3") == 3
    assert service._parse_heading_level_by_name("标题 1") == 1
    assert service._parse_heading_level_by_name("标题 9") == 9

    # 自定义样式名
    assert service._parse_heading_level_by_name("条款标题") == 1
    assert service._parse_heading_level_by_name("章节名") == 1
    assert service._parse_heading_level_by_name("章") == 1
    assert service._parse_heading_level_by_name("节") == 1
    assert service._parse_heading_level_by_name("条") == 1
    assert service._parse_heading_level_by_name("款") == 1
    assert service._parse_heading_level_by_name("项") == 1

    # 非标题样式
    assert service._parse_heading_level_by_name("Normal") is None
    assert service._parse_heading_level_by_name("正文") is None
    assert service._parse_heading_level_by_name(None) is None
    assert service._parse_heading_level_by_name("") is None


def test_parse_heading_level_by_name_edge_cases() -> None:
    """应正确处理边界情况。"""
    service = DocumentChunkService(chunk_size=200, chunk_overlap=20, convert_temp_dir=Path("."))

    # 带空格的样式名
    assert service._parse_heading_level_by_name("Heading  1") == 1
    assert service._parse_heading_level_by_name("标题   2") == 2

    # 混合样式名
    assert service._parse_heading_level_by_name("Heading 1 Custom") == 1
    assert service._parse_heading_level_by_name("Custom 标题 2") == 2

    # 无效样式名
    assert service._parse_heading_level_by_name("Heading") is None
    assert service._parse_heading_level_by_name("标题") is None
    assert service._parse_heading_level_by_name("Heading10") == 10  # 允许超过 9 的级别
    assert service._parse_heading_level_by_name("条款标题自定义") is None  # 必须精确匹配


def test_parse_heading_level_by_content_chapter_patterns() -> None:
    """应支持通过正文内容识别章节编号模式。"""
    service = DocumentChunkService(chunk_size=200, chunk_overlap=20, convert_temp_dir=Path("."))

    # 第 X 章 - 一级标题
    assert service._parse_heading_level_by_content("第一章 总则") == 1
    assert service._parse_heading_level_by_content("第 1 章 总则") == 1
    assert service._parse_heading_level_by_content("第 二 章 总则") == 1

    # 第 X 节 - 二级标题
    assert service._parse_heading_level_by_content("第一节 适用范围") == 2
    assert service._parse_heading_level_by_content("第 1 节 适用范围") == 2

    # 第 X 条 - 根据编号类型判断级别
    assert service._parse_heading_level_by_content("第一条 保险责任") == 2  # 中文数字
    assert service._parse_heading_level_by_content("第 1 条 保险责任") == 3  # 阿拉伯数字


def test_parse_heading_level_by_content_number_patterns() -> None:
    """应支持通过正文内容识别中文数字和括号编号模式。"""
    service = DocumentChunkService(chunk_size=200, chunk_overlap=20, convert_temp_dir=Path("."))

    # （一）、(一) - 中文括号 + 中文数字，二级标题
    assert service._parse_heading_level_by_content("（一）适用范围") == 2
    assert service._parse_heading_level_by_content("(一) 适用范围") == 2

    # 一、二、三、 - 中文数字 + 顿号，一级标题
    assert service._parse_heading_level_by_content("一、总则") == 1
    assert service._parse_heading_level_by_content("二、分则") == 1
    assert service._parse_heading_level_by_content("十、附则") == 1

    # 1.1、1.1.1 - 小数点分隔，根据段数判断级别
    # 注意：1.1 有 2 个数字段，所以是级别 2
    assert service._parse_heading_level_by_content("1.1 概述") == 2
    assert service._parse_heading_level_by_content("1.1.1 详细说明") == 3
    assert service._parse_heading_level_by_content("1.1.1.1 补充内容") == 4


def test_parse_heading_level_by_content_edge_cases() -> None:
    """应正确处理正文内容匹配的边界情况。"""
    service = DocumentChunkService(chunk_size=200, chunk_overlap=20, convert_temp_dir=Path("."))

    # 非标题内容
    assert service._parse_heading_level_by_content("这是普通正文") is None
    assert service._parse_heading_level_by_content("正文内容没有任何编号") is None
    assert service._parse_heading_level_by_content("") is None

    # 边界情况：第一条 vs 第 1 条
    assert service._parse_heading_level_by_content("第一条") == 2
    assert service._parse_heading_level_by_content("第 1 条") == 3

    # 级别上限 - 最多支持 9 段数字
    assert service._parse_heading_level_by_content("1.1.1.1.1.1.1.1.1") == 9  # 9 段数字
    assert service._parse_heading_level_by_content("1.1.1.1.1.1.1.1.1.1") == 9  # 10 段也返回 9


def _write_docx(file_path: Path, paragraphs: Sequence[tuple[str, str | None]]) -> None:
    """写入测试用 docx 文件（仅设置样式名）。

    参数:
        file_path: 要写入的 docx 文件路径。
        paragraphs: (文本，样式名) 元组序列，样式名为 None 表示正文段落。
    """
    document = Document()
    for text, style in paragraphs:
        paragraph = document.add_paragraph(text)
        if style is not None:
            paragraph.style = style

    document.save(str(file_path))
