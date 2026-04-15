"""Word 文档切块服务。"""

from __future__ import annotations

import logging
import mimetypes
import re
import subprocess
import textwrap
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, cast

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.document_parsers.pdf_parser import (
    PdfDocumentParser,
    PdfParsedImageAsset,
    PdfParsedSegment,
    PdfParserError,
)
from baozhi_rag.services.term_matching import MaximumMatchingTermMatcher, build_default_term_matcher

LOGGER = logging.getLogger(__name__)
_SOFFICE_CONVERT_LOCK = Lock()
# Word 批注相关的 OOXML 标签与属性名，用于从底层 XML 中定位 comment 锚点。
_COMMENT_RANGE_START_TAG = qn("w:commentRangeStart")
_COMMENT_RANGE_END_TAG = qn("w:commentRangeEnd")
_COMMENT_REFERENCE_TAG = qn("w:commentReference")
_COMMENT_ID_ATTR = qn("w:id")
_DRAWING_BLIP_TAG = qn("a:blip")
_REL_EMBED_ATTR = qn("r:embed")


def _empty_str_list() -> list[str]:
    """返回空字符串列表，避免类型检查器把 `list` 推断为 `list[Unknown]`。"""
    return []


def _empty_image_asset_list() -> list[ChunkImageAsset]:
    """返回空图片资产列表，避免类型检查器推断为未知列表类型。"""
    return []


def _empty_image_asset_ref_list() -> list[ChunkImageAssetRef]:
    """返回空图片引用列表，避免类型检查器推断为未知列表类型。"""
    return []


class SegmentType(Enum):
    """文档片段类型。"""

    PARAGRAPH = "paragraph"
    TABLE = "table"


class ChunkType(Enum):
    """检索 chunk 类型。"""

    TEXT = "text"
    IMAGE_SEMANTIC = "image_semantic"


@dataclass(frozen=True, slots=True)
class ChunkImageAssetRef:
    """chunk 挂载的图片引用键。"""

    segment_id: str
    asset_id: str

    def to_search_document(self) -> dict[str, str]:
        """构造可写入 ES 的图片引用投影。"""
        return {
            "segment_id": self.segment_id,
            "asset_id": self.asset_id,
        }


@dataclass(frozen=True, slots=True)
class ChunkImageAsset:
    """切块阶段使用的图片资产结构。"""

    segment_id: str
    asset_id: str
    asset_index: int
    source_anchor: str
    content_type: str
    extension: str
    image_bytes: bytes | None = None
    image_sha256: str = ""
    normalized_image_sha256: str = ""
    storage_key: str = ""
    thumbnail_storage_key: str | None = None
    width: int | None = None
    height: int | None = None
    ocr_text: str = ""
    summary: str = ""
    image_type: str = ""
    recognition_model: str = ""

    def to_search_document(self) -> dict[str, object]:
        """构造可写入 ES 的图片资产投影。"""
        return {
            "segment_id": self.segment_id,
            "asset_id": self.asset_id,
            "source_anchor": self.source_anchor,
            "storage_key": self.storage_key,
            "thumbnail_storage_key": self.thumbnail_storage_key,
            "image_type": self.image_type,
            "summary": self.summary,
            "ocr_text": self.ocr_text,
        }


@dataclass
class DocumentSegment:
    """文档片段，可以是段落或表格。"""

    segment_id: str
    content: str
    segment_type: SegmentType
    heading_context: str
    page_number: int | None = None
    source_anchor: str | None = None
    # 锚定到当前片段的批注文本，会在切块前拼接到可检索内容中。
    comment_texts: list[str] = field(default_factory=_empty_str_list)
    # 锚定到当前片段的图片资产，会在切块阶段进一步绑定到 chunk。
    image_assets: list[ChunkImageAsset] = field(default_factory=_empty_image_asset_list)


class DocumentChunkingError(AppError):
    """文档切块失败。"""

    default_message = "文档切块失败"
    default_error_code = "document_chunking_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class UnsupportedDocumentTypeError(DocumentChunkingError):
    """不支持的文档格式。"""

    default_message = "不支持的文档格式"
    default_error_code = "unsupported_document_type"
    default_status_code = status.HTTP_400_BAD_REQUEST


class DocumentConversionError(DocumentChunkingError):
    """文档格式转换失败。"""

    default_message = "文档格式转换失败"
    default_error_code = "document_conversion_error"
    default_status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class DocumentParseError(DocumentChunkingError):
    """文档内容解析失败。"""

    default_message = "文档内容解析失败"
    default_error_code = "document_parse_error"
    default_status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """标准化切块结果。"""

    # 原始文件维度的稳定标识
    file_id: str
    # 切块的唯一标识，格式约定为 `{file_id}-chunk-{chunk_index}`。
    chunk_id: str
    # 文件内的切块顺序号
    chunk_index: int
    # chunk 类型，区分正文和图片语义。
    chunk_type: str
    # 原始解析片段标识，用于图片去重和跨 chunk 归并。
    segment_id: str
    # 切块后的正文内容
    content: str
    # 切块的字符数
    char_count: int
    # 上传时的原始文件名
    source_filename: str
    # 文件在对象存储中的对象键
    storage_key: str
    page_number: int | None = None
    source_anchor: str | None = None
    # 上传该文件的用户 ID
    uploader_user_id: str = ""
    # 文件可见性范围
    visibility_scope: str = ""
    # chunk 所属标题路径
    heading_path: list[str] = field(default_factory=_empty_str_list)
    # chunk 所属末级标题
    section_title: str | None = None
    # chunk 内容类型，当前主要区分 paragraph / table
    content_type: str = "paragraph"
    # 基于领域词词典抽取出的去重词项，用于检索时显式提权。
    merged_terms: list[str] = field(default_factory=_empty_str_list)
    # 与当前 chunk 关联的图片引用键。
    image_asset_refs: list[ChunkImageAssetRef] = field(default_factory=_empty_image_asset_ref_list)
    # 与当前 chunk 绑定的图片资产元数据。
    image_assets: list[ChunkImageAsset] = field(default_factory=_empty_image_asset_list)
    # 向量化
    content_embedding: list[float] | None = None

    def to_search_document(self) -> dict[str, object]:
        """构造 ES 文本检索入库文档。"""
        return {
            "chunk_id": self.chunk_id,
            "file_id": self.file_id,
            "source_filename": self.source_filename,
            "storage_key": self.storage_key,
            "page_number": self.page_number,
            "source_anchor": self.source_anchor,
            "uploader_user_id": self.uploader_user_id,
            "visibility_scope": self.visibility_scope,
            "chunk_type": self.chunk_type,
            "segment_id": self.segment_id,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "heading_path": self.heading_path,
            "section_title": self.section_title,
            "content_type": self.content_type,
            "content": self.content,
            "merged_terms": self.merged_terms,
            "image_asset_refs": [item.to_search_document() for item in self.image_asset_refs],
            "image_assets": [asset.to_search_document() for asset in self.image_assets],
        }


class DocumentChunkService:
    """负责 Word 文档解析、切块与预览日志输出。"""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        convert_temp_dir: Path,
        doc_convert_timeout_seconds: int = 120,
        term_matcher: MaximumMatchingTermMatcher | None = None,
        pdf_parser: PdfDocumentParser | None = None,
    ) -> None:
        """初始化切块服务。

        参数:
            chunk_size: 单个切块的目标最大字符数。
            chunk_overlap: 相邻切块之间保留的重叠字符数。
            convert_temp_dir: `.doc` 转 `.docx` 时使用的临时目录。
            doc_convert_timeout_seconds: `soffice` 转换超时时间，单位为秒。
            term_matcher: 领域词匹配器；未传时使用默认领域词典。

        返回:
            None。

        异常:
            ValueError: 当切块大小或 overlap 配置不合法时抛出。
        """
        if chunk_size <= 0:
            msg = "doc_chunk_size 必须大于 0"
            raise ValueError(msg)
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            msg = "doc_chunk_overlap 必须在 0 和 chunk_size 之间"
            raise ValueError(msg)
        if doc_convert_timeout_seconds <= 0:
            msg = "doc_convert_timeout_seconds 必须大于 0"
            raise ValueError(msg)

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._convert_temp_dir = convert_temp_dir
        self._doc_convert_timeout_seconds = doc_convert_timeout_seconds
        self._term_matcher = term_matcher or build_default_term_matcher()
        self._pdf_parser = pdf_parser

    def chunk_document(
        self,
        file_path: Path,
        source_filename: str,
        storage_key: str,
        file_id: str,
    ) -> list[DocumentChunk]:
        """根据文件扩展名分发到对应切块函数。

        参数:
            file_path: 已落盘文件的绝对路径。
            source_filename: 上传时的原始文件名，用于日志和异常信息。
            storage_key: 文件在本地存储中的相对路径。
            file_id: 文件唯一标识，用于生成 chunk 标识。

        返回:
            标准化的文档切块列表。

        异常:
            UnsupportedDocumentTypeError: 当文件扩展名未被支持时抛出。
            DocumentChunkingError: 当解析、转换或切块失败时抛出。
        """
        # 获取文件拓展名并转换为小写
        suffix = file_path.suffix.lower()

        match suffix:
            case ".docx":
                chunks = self.chunk_docx(
                    file_path=file_path,
                    source_filename=source_filename,
                    storage_key=storage_key,
                    file_id=file_id,
                )
            case ".doc":
                chunks = self.chunk_doc(
                    file_path=file_path,
                    source_filename=source_filename,
                    storage_key=storage_key,
                    file_id=file_id,
                )
            case ".pdf":
                chunks = self.chunk_pdf(
                    file_path=file_path,
                    source_filename=source_filename,
                    storage_key=storage_key,
                    file_id=file_id,
                )
            case _:
                msg = f"暂不支持的文件格式: {suffix or 'unknown'}"
                raise UnsupportedDocumentTypeError(msg)

        self._log_chunk_preview(
            file_id=file_id,
            source_filename=source_filename,
            storage_key=storage_key,
            chunks=chunks,
        )
        return chunks

    def chunk_docx(
        self,
        file_path: Path,
        source_filename: str,
        storage_key: str,
        file_id: str,
    ) -> list[DocumentChunk]:
        """解析 docx 文件并生成切块。

        参数:
            file_path: docx 文件的绝对路径。
            source_filename: 上传时的原始文件名。
            storage_key: 文件在本地存储中的相对路径。
            file_id: 文件唯一标识。

        返回:
            基于 docx 正文与标题上下文生成的切块列表。

        异常:
            DocumentParseError: 当文档无法解析或正文为空时抛出。
        """
        try:
            document = Document(str(file_path))
        except Exception as exc:  # pragma: no cover - 第三方库异常类型不稳定
            msg = f"解析 docx 文件失败: {source_filename}"
            raise DocumentParseError(msg) from exc

        # 将标题与正文进行合并，形成带上下文的文本片段列表
        segments = self._extract_docx_segments(document)
        if not segments:
            msg = f"Word 文档内容为空，无法切块: {source_filename}"
            raise DocumentParseError(msg)

        chunks: list[DocumentChunk] = []
        paragraph_buffer: list[str] = []
        paragraph_buffer_segment_ids: list[str] = []
        paragraph_buffer_heading_context = ""

        for segment in segments:
            if segment.segment_type is SegmentType.PARAGRAPH:
                if paragraph_buffer and paragraph_buffer_heading_context != segment.heading_context:
                    buffered_segment_id = self._build_buffer_segment_id(
                        paragraph_buffer_segment_ids
                    )
                    chunks.extend(
                        self._build_chunks(
                            text="\n\n".join(paragraph_buffer),
                            source_filename=source_filename,
                            storage_key=storage_key,
                            file_id=file_id,
                            segment_id=buffered_segment_id,
                            start_index=len(chunks),
                            heading_context=paragraph_buffer_heading_context,
                            content_type="paragraph",
                        )
                    )
                    paragraph_buffer.clear()
                    paragraph_buffer_segment_ids.clear()
                    paragraph_buffer_heading_context = ""

                if not segment.image_assets:
                    paragraph_buffer.append(segment.content)
                    paragraph_buffer_segment_ids.append(segment.segment_id)
                    paragraph_buffer_heading_context = segment.heading_context
                    continue

                # 带图片的段落单独切块，避免图片和无关段落被混装到同一个 chunk。
                if paragraph_buffer:
                    buffered_segment_id = self._build_buffer_segment_id(
                        paragraph_buffer_segment_ids
                    )
                    chunks.extend(
                        self._build_chunks(
                            text="\n\n".join(paragraph_buffer),
                            source_filename=source_filename,
                            storage_key=storage_key,
                            file_id=file_id,
                            segment_id=buffered_segment_id,
                            start_index=len(chunks),
                            heading_context=paragraph_buffer_heading_context,
                            content_type="paragraph",
                        )
                    )
                    paragraph_buffer.clear()
                    paragraph_buffer_segment_ids.clear()
                    paragraph_buffer_heading_context = ""

                chunks.extend(
                    self._build_chunks(
                        text=segment.content,
                        source_filename=source_filename,
                        storage_key=storage_key,
                        file_id=file_id,
                        segment_id=segment.segment_id,
                        start_index=len(chunks),
                        heading_context=segment.heading_context,
                        content_type="paragraph",
                        image_assets=segment.image_assets,
                    )
                )
                continue

            if paragraph_buffer:
                buffered_segment_id = self._build_buffer_segment_id(paragraph_buffer_segment_ids)
                chunks.extend(
                    self._build_chunks(
                        text="\n\n".join(paragraph_buffer),
                        source_filename=source_filename,
                        storage_key=storage_key,
                        file_id=file_id,
                        segment_id=buffered_segment_id,
                        start_index=len(chunks),
                        heading_context=paragraph_buffer_heading_context,
                        content_type="paragraph",
                    )
                )
                paragraph_buffer.clear()
                paragraph_buffer_segment_ids.clear()
                paragraph_buffer_heading_context = ""

            chunks.extend(
                self._build_table_chunks(
                    table_markdown=segment.content,
                    source_filename=source_filename,
                    storage_key=storage_key,
                    file_id=file_id,
                    segment_id=segment.segment_id,
                    start_index=len(chunks),
                    heading_context=segment.heading_context,
                    comment_texts=segment.comment_texts,
                    image_assets=segment.image_assets,
                )
            )

        if paragraph_buffer:
            buffered_segment_id = self._build_buffer_segment_id(paragraph_buffer_segment_ids)
            chunks.extend(
                self._build_chunks(
                    text="\n\n".join(paragraph_buffer),
                    source_filename=source_filename,
                    storage_key=storage_key,
                    file_id=file_id,
                    segment_id=buffered_segment_id,
                    start_index=len(chunks),
                    heading_context=paragraph_buffer_heading_context,
                    content_type="paragraph",
                )
            )

        return chunks

    def chunk_doc(
        self,
        file_path: Path,
        source_filename: str,
        storage_key: str,
        file_id: str,
    ) -> list[DocumentChunk]:
        """将 doc 转换为 docx 后复用 docx 切块逻辑。

        参数:
            file_path: doc 文件的绝对路径。
            source_filename: 上传时的原始文件名。
            storage_key: 文件在本地存储中的相对路径。
            file_id: 文件唯一标识。

        返回:
            复用 `chunk_docx` 逻辑生成的切块列表。

        异常:
            DocumentConversionError: 当 `.doc` 无法转换为 `.docx` 时抛出。
            DocumentParseError: 当转换后的 docx 无法解析或正文为空时抛出。
        """
        converted_path = self._convert_doc_to_docx(file_path)

        try:
            return self.chunk_docx(
                file_path=converted_path,
                source_filename=source_filename,
                storage_key=storage_key,
                file_id=file_id,
            )
        finally:
            self._cleanup_converted_file(converted_path)

    def chunk_pdf(
        self,
        file_path: Path,
        source_filename: str,
        storage_key: str,
        file_id: str,
    ) -> list[DocumentChunk]:
        """解析 PDF 文件并生成切块。"""
        if self._pdf_parser is None:
            msg = "未配置 PDF 解析器，无法处理 PDF 文件"
            raise DocumentParseError(msg)

        try:
            parsed_segments = self._pdf_parser.parse(
                file_path=str(file_path),
                source_filename=source_filename,
            )
        except PdfParserError as exc:
            raise DocumentParseError(str(exc)) from exc

        segments = [self._convert_pdf_segment(item) for item in parsed_segments]
        return self._chunk_segments(
            segments=segments,
            source_filename=source_filename,
            storage_key=storage_key,
            file_id=file_id,
        )

    def _convert_pdf_segment(self, segment: PdfParsedSegment) -> DocumentSegment:
        """把 PDF parser 结果转换为统一 DocumentSegment。"""
        return DocumentSegment(
            segment_id=segment.segment_id,
            content=segment.content,
            segment_type=SegmentType.TABLE
            if segment.segment_type == "table"
            else SegmentType.PARAGRAPH,
            heading_context=segment.heading_context,
            page_number=segment.page_number,
            source_anchor=segment.source_anchor,
            image_assets=[self._convert_pdf_image_asset(item) for item in segment.image_assets],
        )

    def _convert_pdf_image_asset(self, asset: PdfParsedImageAsset) -> ChunkImageAsset:
        """把 PDF 图片资产转换为统一图片资产。"""
        return ChunkImageAsset(
            segment_id=asset.segment_id,
            asset_id=asset.asset_id,
            asset_index=asset.asset_index,
            source_anchor=asset.source_anchor,
            content_type=asset.content_type,
            extension=asset.extension,
            image_bytes=asset.image_bytes,
            width=asset.width,
            height=asset.height,
        )

    def _chunk_segments(
        self,
        *,
        segments: list[DocumentSegment],
        source_filename: str,
        storage_key: str,
        file_id: str,
    ) -> list[DocumentChunk]:
        """把标准化片段转换为统一切块结果。"""
        if not segments:
            msg = f"文档内容为空，无法切块: {source_filename}"
            raise DocumentParseError(msg)

        chunks: list[DocumentChunk] = []
        paragraph_buffer: list[str] = []
        paragraph_buffer_segment_ids: list[str] = []
        paragraph_buffer_heading_context = ""
        paragraph_buffer_page_number: int | None = None
        paragraph_buffer_source_anchor: str | None = None

        for segment in segments:
            if segment.segment_type is SegmentType.PARAGRAPH:
                if paragraph_buffer and paragraph_buffer_heading_context != segment.heading_context:
                    buffered_segment_id = self._build_buffer_segment_id(
                        paragraph_buffer_segment_ids
                    )
                    chunks.extend(
                        self._build_chunks(
                            text="\n\n".join(paragraph_buffer),
                            source_filename=source_filename,
                            storage_key=storage_key,
                            file_id=file_id,
                            segment_id=buffered_segment_id,
                            start_index=len(chunks),
                            heading_context=paragraph_buffer_heading_context,
                            content_type="paragraph",
                            page_number=paragraph_buffer_page_number,
                            source_anchor=paragraph_buffer_source_anchor,
                        )
                    )
                    paragraph_buffer.clear()
                    paragraph_buffer_segment_ids.clear()
                    paragraph_buffer_heading_context = ""
                    paragraph_buffer_page_number = None
                    paragraph_buffer_source_anchor = None

                if not segment.image_assets:
                    paragraph_buffer.append(segment.content)
                    paragraph_buffer_segment_ids.append(segment.segment_id)
                    paragraph_buffer_heading_context = segment.heading_context
                    paragraph_buffer_page_number = (
                        paragraph_buffer_page_number
                        if paragraph_buffer_page_number is not None
                        else segment.page_number
                    )
                    paragraph_buffer_source_anchor = (
                        paragraph_buffer_source_anchor or segment.source_anchor
                    )
                    continue

                if paragraph_buffer:
                    buffered_segment_id = self._build_buffer_segment_id(
                        paragraph_buffer_segment_ids
                    )
                    chunks.extend(
                        self._build_chunks(
                            text="\n\n".join(paragraph_buffer),
                            source_filename=source_filename,
                            storage_key=storage_key,
                            file_id=file_id,
                            segment_id=buffered_segment_id,
                            start_index=len(chunks),
                            heading_context=paragraph_buffer_heading_context,
                            content_type="paragraph",
                            page_number=paragraph_buffer_page_number,
                            source_anchor=paragraph_buffer_source_anchor,
                        )
                    )
                    paragraph_buffer.clear()
                    paragraph_buffer_segment_ids.clear()
                    paragraph_buffer_heading_context = ""
                    paragraph_buffer_page_number = None
                    paragraph_buffer_source_anchor = None

                chunks.extend(
                    self._build_chunks(
                        text=segment.content,
                        source_filename=source_filename,
                        storage_key=storage_key,
                        file_id=file_id,
                        segment_id=segment.segment_id,
                        start_index=len(chunks),
                        heading_context=segment.heading_context,
                        content_type="paragraph",
                        image_assets=segment.image_assets,
                        page_number=segment.page_number,
                        source_anchor=segment.source_anchor,
                    )
                )
                continue

            if paragraph_buffer:
                buffered_segment_id = self._build_buffer_segment_id(paragraph_buffer_segment_ids)
                chunks.extend(
                    self._build_chunks(
                        text="\n\n".join(paragraph_buffer),
                        source_filename=source_filename,
                        storage_key=storage_key,
                        file_id=file_id,
                        segment_id=buffered_segment_id,
                        start_index=len(chunks),
                        heading_context=paragraph_buffer_heading_context,
                        content_type="paragraph",
                        page_number=paragraph_buffer_page_number,
                        source_anchor=paragraph_buffer_source_anchor,
                    )
                )
                paragraph_buffer.clear()
                paragraph_buffer_segment_ids.clear()
                paragraph_buffer_heading_context = ""
                paragraph_buffer_page_number = None
                paragraph_buffer_source_anchor = None

            chunks.extend(
                self._build_table_chunks(
                    table_markdown=segment.content,
                    source_filename=source_filename,
                    storage_key=storage_key,
                    file_id=file_id,
                    segment_id=segment.segment_id,
                    start_index=len(chunks),
                    heading_context=segment.heading_context,
                    comment_texts=segment.comment_texts,
                    image_assets=segment.image_assets,
                    page_number=segment.page_number,
                    source_anchor=segment.source_anchor,
                )
            )

        if paragraph_buffer:
            buffered_segment_id = self._build_buffer_segment_id(paragraph_buffer_segment_ids)
            chunks.extend(
                self._build_chunks(
                    text="\n\n".join(paragraph_buffer),
                    source_filename=source_filename,
                    storage_key=storage_key,
                    file_id=file_id,
                    segment_id=buffered_segment_id,
                    start_index=len(chunks),
                    heading_context=paragraph_buffer_heading_context,
                    content_type="paragraph",
                    page_number=paragraph_buffer_page_number,
                    source_anchor=paragraph_buffer_source_anchor,
                )
            )

        return chunks

    def _convert_doc_to_docx(self, file_path: Path) -> Path:
        """调用 soffice 将 doc 转换为 docx。

        参数:
            file_path: 需要转换的 `.doc` 文件绝对路径。

        返回:
            转换产物 `.docx` 的绝对路径。

        异常:
            DocumentConversionError: 当未安装 soffice 或转换命令失败时抛出。
        """
        output_dir = self._convert_temp_dir / file_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with _SOFFICE_CONVERT_LOCK:
                result = subprocess.run(
                    [
                        "soffice",
                        "--headless",
                        "--convert-to",
                        "docx",
                        "--outdir",
                        str(output_dir),
                        str(file_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self._doc_convert_timeout_seconds,
                )
        except FileNotFoundError as exc:
            msg = "未找到 soffice，无法解析 .doc 文件"
            raise DocumentConversionError(msg) from exc
        except subprocess.TimeoutExpired as exc:
            msg = "文档格式转换超时，请检查文件是否损坏或尝试手动转为 .docx 后重新上传"
            raise DocumentConversionError(msg) from exc

        converted_path = output_dir / f"{file_path.stem}.docx"
        if result.returncode != 0 or not converted_path.exists():
            error_message = (result.stderr or result.stdout).strip() or "未知错误"
            msg = f".doc 转换失败: {error_message}"
            raise DocumentConversionError(msg)

        return converted_path

    def _cleanup_converted_file(self, converted_path: Path) -> None:
        """清理 doc 临时转换文件。

        参数:
            converted_path: 临时生成的 `.docx` 文件绝对路径。

        返回:
            None。清理失败时静默忽略，避免覆盖主异常。
        """
        with suppress(OSError):
            if converted_path.exists():
                converted_path.unlink()
        with suppress(OSError):
            if converted_path.parent.exists():
                converted_path.parent.rmdir()

    def _is_vertically_merged_continuation(self, cell: object) -> bool:
        """检查单元格是否是垂直合并的延续格（需要跳过）。

        参数:
            cell: 表格单元格对象。

        返回:
            True 表示该单元格是合并延续格，应跳过；False 表示正常单元格或合并起始格。
        """
        # python-docx 这里没有公开类型声明，只能通过底层 OOXML 属性读取合并标记。
        cell_element = getattr(cast(Any, cell), "_element", None)
        tc_pr = getattr(cell_element, "tcPr", None)
        if tc_pr is None:
            return False

        vmerge = tc_pr.vMerge
        if vmerge is None:
            return False

        # restart 表示合并开始，continue 或无 val 表示延续
        val = vmerge.val
        return val is None or val == "continue"

    def _table_to_markdown(self, table: Table) -> str:
        """将表格转换为 Markdown 格式，处理合并单元格。

        参数:
            table: python-docx 的 Table 对象。

        返回:
            Markdown 格式的表格字符串；空表格返回空字符串。
        """
        if not table.rows:
            return ""

        lines: list[str] = []
        previous_row_texts: list[str] | None = None

        for row_index, row in enumerate(table.rows):
            row_values: list[str] = []
            current_row_texts: list[str] = []

            for column_index, cell in enumerate(row.cells):
                text = cell.text.strip().replace("\n", " ").replace("|", "\\|")
                current_row_texts.append(text)

                # 读取底层 tcPr，识别竖向合并场景，避免把续格内容重复写入 Markdown。
                cell_element = getattr(cast(Any, cell), "_element", None)
                tc_pr = getattr(cell_element, "tcPr", None)
                has_vertical_merge_marker = tc_pr is not None and tc_pr.vMerge is not None
                is_repeated_vertical_merge = (
                    has_vertical_merge_marker
                    and previous_row_texts is not None
                    and column_index < len(previous_row_texts)
                    and text
                    and text == previous_row_texts[column_index]
                )
                if self._is_vertically_merged_continuation(cell) or is_repeated_vertical_merge:
                    row_values.append("")
                else:
                    row_values.append(text)

            lines.append("| " + " | ".join(row_values) + " |")
            if row_index == 0:
                lines.append("|" + "|".join(["---"] * len(row_values)) + "|")
            previous_row_texts = current_row_texts

        return "\n".join(lines)

    def _process_paragraph(
        self,
        paragraph: Paragraph,
        headings: list[str],
        comment_texts: list[str],
        image_assets: list[ChunkImageAsset],
        segment_id: str,
    ) -> tuple[DocumentSegment | None, list[str]]:
        """处理单个段落，返回 DocumentSegment 和更新后的 headings。

        参数:
            paragraph: python-docx 的 Paragraph 对象。
            headings: 当前的标题层级列表。

        返回:
            元组 (DocumentSegment | None, 更新后的 headings 列表)。
            空段落返回 (None, headings)。
        """
        text = paragraph.text.strip()
        if not text:
            return None, headings

        heading_level = self._get_heading_level(paragraph)
        if heading_level is not None:
            # 更新标题层级
            updated_headings = headings[: heading_level - 1] + [text]
            content = self._append_comment_block(" / ".join(updated_headings), comment_texts)
            segment = DocumentSegment(
                segment_id=segment_id,
                content=content,
                segment_type=SegmentType.PARAGRAPH,
                heading_context=" / ".join(updated_headings),
                comment_texts=comment_texts,
                image_assets=image_assets,
            )
            return segment, updated_headings

        # 普通段落
        context = " / ".join(headings)
        content = f"{context}\n{text}" if context else text
        content = self._append_comment_block(content, comment_texts)
        segment = DocumentSegment(
            segment_id=segment_id,
            content=content,
            segment_type=SegmentType.PARAGRAPH,
            heading_context=" / ".join(headings),
            comment_texts=comment_texts,
            image_assets=image_assets,
        )
        return segment, headings

    def _process_table(
        self,
        table: Table,
        headings: list[str],
        comment_texts: list[str],
        image_assets: list[ChunkImageAsset],
        segment_id: str,
    ) -> DocumentSegment | None:
        """处理单个表格，转换为 Markdown 格式。

        参数:
            table: python-docx 的 Table 对象。
            headings: 当前的标题层级列表。

        返回:
            包含 Markdown 表格的 DocumentSegment；空表格返回 None。
        """
        markdown = self._table_to_markdown(table)
        if not markdown:
            return None

        return DocumentSegment(
            segment_id=segment_id,
            content=markdown,
            segment_type=SegmentType.TABLE,
            heading_context=" / ".join(headings),
            comment_texts=comment_texts,
            image_assets=image_assets,
        )

    def _extract_docx_segments(self, document: DocxDocument) -> list[DocumentSegment]:
        """抽取段落和表格，保持原始顺序并保留标题上下文。

        参数:
            document: `python-docx` 解析后的文档对象。

        返回:
            包含段落和表格的 DocumentSegment 列表，按文档原始顺序排列。
        """
        headings: list[str] = []
        segments: list[DocumentSegment] = []
        # Pylance 对 `document.element.body` 的类型推断较弱，这里先显式转成可遍历列表。
        body_elements = list(cast(Iterable[Any], cast(Any, document.element).body))
        body_comment_ids = self._collect_body_element_comment_ids(document)
        comment_lookup = self._build_comment_lookup(document)
        # 先按顶层 body 元素收集批注锚点，确保后续与 python-docx 的遍历顺序一致。
        for element_index, element in enumerate(body_elements):
            segment_id = f"seg-{element_index + 1}"
            # 当前元素可能命中多个批注，这里统一去重后再交给段落/表格处理逻辑。
            comment_texts = self._resolve_comment_texts(
                body_comment_ids.get(element_index, []),
                comment_lookup,
            )
            image_assets = self._extract_body_element_images(
                element=element,
                document=document,
                element_index=element_index,
            )
            if isinstance(element, CT_P):
                # 处理段落
                paragraph = Paragraph(element, document)
                segment, headings = self._process_paragraph(
                    paragraph,
                    headings,
                    comment_texts,
                    image_assets,
                    segment_id,
                )
                if segment:
                    segments.append(segment)
            elif isinstance(element, CT_Tbl):
                # 处理表格
                table = Table(element, document)
                segment = self._process_table(
                    table,
                    headings,
                    comment_texts,
                    image_assets,
                    segment_id,
                )
                if segment:
                    segments.append(segment)

        return segments

    def _build_comment_lookup(self, document: DocxDocument) -> dict[int, str]:
        """构建 comment id 到归一化批注文本的映射。"""
        comment_lookup: dict[int, str] = {}
        for comment in document.comments:
            # 批注正文会参与检索与 content_sha256，先做稳定归一化。
            normalized_text = self._normalize_comment_text(comment.text)
            if not normalized_text:
                continue
            comment_lookup[int(comment.comment_id)] = normalized_text
        return comment_lookup

    def _collect_body_element_comment_ids(self, document: DocxDocument) -> dict[int, list[int]]:
        """按顶层 body 元素收集命中的批注 id。"""
        body_comment_ids: dict[int, list[int]] = {}
        # active_comment_ids 表示跨元素仍然处于生效范围内的批注。
        active_comment_ids: set[int] = set()
        body_elements = list(cast(Iterable[Any], cast(Any, document.element).body))

        for element_index, element in enumerate(body_elements):
            # 先继承跨元素延续中的批注，再补当前元素内新出现的批注。
            element_comment_ids: set[int] = set(active_comment_ids)

            for descendant in cast(Iterable[Any], element.iter()):
                if descendant.tag == _COMMENT_RANGE_START_TAG:
                    # commentRangeStart 表示批注从当前元素开始生效。
                    comment_id = self._read_comment_id(descendant)
                    if comment_id is None:
                        continue
                    active_comment_ids.add(comment_id)
                    element_comment_ids.add(comment_id)
                    continue

                if descendant.tag == _COMMENT_REFERENCE_TAG:
                    # commentReference 通常出现在结束处，这里也视为当前元素命中。
                    comment_id = self._read_comment_id(descendant)
                    if comment_id is not None:
                        element_comment_ids.add(comment_id)
                    continue

                if descendant.tag == _COMMENT_RANGE_END_TAG:
                    # commentRangeEnd 命中当前元素后，需要把批注从活动集合中移除。
                    comment_id = self._read_comment_id(descendant)
                    if comment_id is None:
                        continue
                    element_comment_ids.add(comment_id)
                    active_comment_ids.discard(comment_id)

            if element_comment_ids:
                body_comment_ids[element_index] = sorted(element_comment_ids)

        return body_comment_ids

    def _extract_body_element_images(
        self,
        *,
        element: object,
        document: DocxDocument,
        element_index: int,
    ) -> list[ChunkImageAsset]:
        """从顶层 body 元素中提取图片并构造预览图片资产。"""
        image_assets: list[ChunkImageAsset] = []
        related_parts = cast(dict[str, Any], getattr(document.part, "related_parts", {}))

        element_iter = getattr(element, "iter", None)
        if not callable(element_iter):
            return image_assets

        for asset_index, descendant in enumerate(cast(Iterable[Any], element_iter()), start=1):
            if getattr(descendant, "tag", None) != _DRAWING_BLIP_TAG:
                continue

            relation_id = getattr(descendant, "get", lambda *_args, **_kwargs: None)(
                _REL_EMBED_ATTR
            )
            if relation_id in (None, ""):
                continue

            related_part = related_parts.get(str(relation_id))
            if related_part is None:
                continue

            image_bytes = cast(bytes | None, getattr(related_part, "blob", None))
            if not image_bytes:
                continue

            content_type = str(getattr(related_part, "content_type", "")).strip()
            extension = self._guess_image_extension(content_type)
            segment_id = f"seg-{element_index + 1}"
            source_anchor = f"{segment_id}:image:{asset_index}"
            image_assets.append(
                ChunkImageAsset(
                    segment_id=segment_id,
                    asset_id=f"preview-{segment_id}-{asset_index}",
                    asset_index=asset_index,
                    source_anchor=source_anchor,
                    content_type=content_type,
                    extension=extension,
                    image_bytes=image_bytes,
                )
            )

        return image_assets

    def _guess_image_extension(self, content_type: str) -> str:
        """根据图片 MIME 类型推断扩展名。"""
        guessed_extension = mimetypes.guess_extension(content_type, strict=False)
        if guessed_extension:
            return guessed_extension
        return ".bin"

    def _read_comment_id(self, element: object) -> int | None:
        """从 OOXML 批注锚点元素中读取 comment id。"""
        element_getter = getattr(element, "get", None)
        if not callable(element_getter):
            return None

        raw_value = element_getter(_COMMENT_ID_ATTR)
        if raw_value in (None, ""):
            return None

        try:
            return int(str(raw_value))
        except (TypeError, ValueError):
            return None

    def _resolve_comment_texts(
        self,
        comment_ids: Iterable[int],
        comment_lookup: dict[int, str],
    ) -> list[str]:
        """把批注 id 列表解析为去重后的批注文本列表。"""
        resolved_comment_texts: list[str] = []
        seen_texts: set[str] = set()

        for comment_id in comment_ids:
            comment_text = comment_lookup.get(comment_id)
            if comment_text is None or comment_text in seen_texts:
                continue
            resolved_comment_texts.append(comment_text)
            seen_texts.add(comment_text)

        return resolved_comment_texts

    def _normalize_comment_text(self, text: str) -> str:
        """归一化批注文本，降低换行和空白对哈希与检索的干扰。"""
        return " / ".join(
            normalized_line
            for normalized_line in (" ".join(part.split()) for part in text.splitlines())
            if normalized_line
        )

    def _append_comment_block(self, content: str, comment_texts: list[str]) -> str:
        """把批注块附加到正文末尾，使批注文本能随正文一起被检索。"""
        comment_suffix = self._build_comment_suffix(comment_texts)
        if not comment_suffix:
            return content
        return f"{content}{comment_suffix}"

    def _build_comment_suffix(self, comment_texts: list[str]) -> str:
        """构建稳定的批注补充块。"""
        if not comment_texts:
            return ""

        if len(comment_texts) == 1:
            return f"\n\n批注: {comment_texts[0]}"

        comment_lines = "\n".join(f"- {comment_text}" for comment_text in comment_texts)
        return f"\n\n批注:\n{comment_lines}"

    def _get_heading_level(self, paragraph: Paragraph) -> int | None:
        """获取段落的大纲级别。

        采用三级优先级识别策略：
        1. 标准样式（Heading n / 标题 n） > 自定义样式名
        2. XML 样式 ID 兜底（某些自定义样式可能未暴露到 style.name）
        3. 正文内容正则匹配（第 X 章、第 X 条、（一）、一、等）

        参数:
            paragraph: `python-docx` 解析后的段落对象。

        返回:
            若段落为标题则返回其级别 (1-9)，否则返回 None。
        """
        # 优先级 1：从样式名识别（包含标准样式和自定义样式名）
        style_name = paragraph.style.name if paragraph.style else ""
        level = self._parse_heading_level_by_name(style_name)
        if level is not None:
            return level

        # 优先级 2：从底层 XML 读取样式 ID（某些自定义样式可能未暴露到 style.name）
        # python-docx 未公开这段 XML 结构的完整类型，这里只在局部用 Any 读取样式 ID。
        xml_element = getattr(cast(Any, paragraph), "_element", None)
        p_pr = getattr(xml_element, "pPr", None)
        if p_pr is not None:
            p_style = p_pr.pStyle
            if p_style is not None:
                style_val = p_style.val
                if style_val is not None:
                    level = self._parse_heading_level_by_name(str(style_val))
                    if level is not None:
                        return level

        # 优先级 3：从正文内容正则匹配识别（第 X 章、第 X 条、（一）、一、等）
        text = paragraph.text.strip()
        if text:
            level = self._parse_heading_level_by_content(text)
            if level is not None:
                return level

        return None

    def _parse_heading_level_by_name(self, style_name: str | None) -> int | None:
        """基于样式名称解析标题级别（兜底逻辑）。

        参数:
            style_name: 段落样式名称，例如 `Heading 1`、`标题 1`、`条款标题` 等。

        返回:
            若样式表示标题则返回其层级，否则返回 None。
        """
        if style_name is None:
            return None

        # 匹配标准标题样式：Heading 1-9、标题 1-9
        matched = re.search(r"(Heading|标题)\s*(\d+)", style_name, flags=re.IGNORECASE)
        if matched is not None:
            return int(matched.group(2))

        # 匹配自定义样式名：条款标题、章节名、一级标题等视为一级标题
        # 这些样式通常没有数字后缀，统一视为级别 1
        custom_heading_patterns = [
            r"^条款标题$",
            r"^章节名$",
            r"^章$",
            r"^节$",
            r"^条$",
            r"^款$",
            r"^项$",
        ]
        for pattern in custom_heading_patterns:
            if re.match(pattern, style_name, flags=re.IGNORECASE):
                return 1

        return None

    def _parse_heading_level_by_content(self, text: str) -> int | None:
        """基于段落正文内容正则匹配识别标题级别。

        支持以下正文编号模式（优先级 3，作为样式名识别的兜底）：
        - 第 X 章 / 第 X 节 / 第 X 条：根据编号数字推断级别
        - （一）、(一)：中文括号 + 中文数字，视为二级标题
        - 一、二、三、：中文数字 + 顿号，视为一级标题
        - 1.1、1.1.1：小数点分隔的数字，根据段数推断级别

        参数:
            text: 段落正文文本。

        返回:
            若匹配到标题模式则返回其级别 (1-9)，否则返回 None。
        """
        # 模式 1：第 X 章、第 X 节、第 X 条
        chapter_match = re.match(r"^第\s*([一二三四五六七八九十\d]+)\s*(章|节|条)\b", text)
        if chapter_match:
            num_str = chapter_match.group(1)
            unit = chapter_match.group(2)
            # 章/节视为一级，条视为二级
            if unit == "章":
                return 1
            if unit == "节":
                return 2
            if unit == "条":
                # 如果编号是中文数字，视为二级；如果是阿拉伯数字，视为三级
                if re.match(r"^[一二三四五六七八九十]+$", num_str):
                    return 2
                return 3

        # 模式 2：（一）、(一) —— 中文括号 + 中文数字
        paren_match = re.match(r"^[（(]([一二三四五六七八九十]+)[)）]", text)
        if paren_match:
            return 2  # 视为二级标题

        # 模式 3：一、二、三、 —— 中文数字 + 顿号
        chinese_num_match = re.match(r"^([一二三四五六七八九十]+)[、．.]", text)
        if chinese_num_match:
            return 1  # 视为一级标题

        # 模式 4：1.1、1.1.1 —— 小数点分隔的数字
        # 使用更灵活的正则，支持任意段数
        dotted_match = re.match(
            r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?",
            text,
        )
        if dotted_match:
            # 根据非空捕获组数判断级别
            levels = sum(1 for g in dotted_match.groups() if g is not None)
            return min(levels, 9)  # 最多支持 9 级

        return None

    def _parse_heading_level(self, style_name: str | None) -> int | None:
        """解析标题样式级别（已废弃，保留以便向后兼容）。

        参数:
            style_name: 段落样式名称。

        返回:
            若样式表示标题则返回其层级，否则返回 None。
        """
        return self._parse_heading_level_by_name(style_name)

    def _create_chunk(
        self,
        content: str,
        chunk_index: int,
        source_filename: str,
        storage_key: str,
        file_id: str,
        chunk_type: ChunkType,
        segment_id: str,
        heading_context: str = "",
        content_type: str = "paragraph",
        image_assets: list[ChunkImageAsset] | None = None,
        page_number: int | None = None,
        source_anchor: str | None = None,
    ) -> DocumentChunk:
        """基于统一元数据创建单个 chunk。

        参数:
            content: 当前 chunk 的正文内容。
            chunk_index: 当前 chunk 在文件中的顺序号。
            source_filename: 上传时的原始文件名。
            storage_key: 文件在本地存储中的相对路径。
            file_id: 文件唯一标识。

        返回:
            已补齐 `chunk_id`、字符数和领域词抽取结果的标准化 chunk。
        """
        term_match_result = self._term_matcher.extract_terms(content)
        heading_path = self._parse_heading_path(heading_context)
        return DocumentChunk(
            file_id=file_id,
            chunk_id=f"{file_id}-chunk-{chunk_index}",
            chunk_index=chunk_index,
            chunk_type=chunk_type.value,
            segment_id=segment_id,
            content=content,
            char_count=len(content),
            source_filename=source_filename,
            storage_key=storage_key,
            page_number=page_number,
            source_anchor=source_anchor,
            heading_path=heading_path,
            section_title=heading_path[-1] if heading_path else None,
            content_type=content_type,
            merged_terms=term_match_result.merged_terms,
            image_asset_refs=self._build_image_asset_refs(image_assets or []),
            image_assets=list(image_assets or []),
        )

    def _parse_heading_path(self, heading_context: str) -> list[str]:
        """把标题上下文字符串解析为层级标题路径。"""
        if not heading_context.strip():
            return []
        return [item.strip() for item in heading_context.split(" / ") if item.strip()]

    def _build_image_asset_refs(
        self,
        image_assets: list[ChunkImageAsset],
    ) -> list[ChunkImageAssetRef]:
        """根据图片资产构造 chunk 引用键列表。"""
        return [
            ChunkImageAssetRef(
                segment_id=asset.segment_id,
                asset_id=asset.asset_id,
            )
            for asset in image_assets
        ]

    def _build_buffer_segment_id(
        self,
        segment_ids: list[str],
    ) -> str:
        """为缓冲合并后的纯文本片段构造稳定 segment_id。"""
        if not segment_ids:
            return "seg-buffer-empty"
        if len(segment_ids) == 1:
            return segment_ids[0]
        return f"{segment_ids[0]}__{segment_ids[-1]}"

    def _build_chunks(
        self,
        text: str,
        source_filename: str,
        storage_key: str,
        file_id: str,
        segment_id: str,
        start_index: int = 0,
        heading_context: str = "",
        content_type: str = "paragraph",
        image_assets: list[ChunkImageAsset] | None = None,
        page_number: int | None = None,
        source_anchor: str | None = None,
    ) -> list[DocumentChunk]:
        """按固定窗口与 overlap 生成切块。

        参数:
            text: 已完成段落拼接和标题补全的正文文本。
            source_filename: 上传时的原始文件名。
            storage_key: 文件在本地存储中的相对路径。
            file_id: 文件唯一标识。
            start_index: 当前批次切块写入前的起始序号。

        返回:
            基于字符窗口切分后的标准化 chunk 列表。
            每个 chunk 都保留 file_id，以便后续 ES/Milvus 接入。

        异常:
            DocumentParseError: 当正文为空无法切块时抛出。
        """
        normalized_text = text.strip()
        if not normalized_text:
            msg = f"Word 文档内容为空，无法切块: {source_filename}"
            raise DocumentParseError(msg)

        chunks: list[DocumentChunk] = []
        start = 0

        while start < len(normalized_text):
            end = min(start + self._chunk_size, len(normalized_text))
            chunk_content = normalized_text[start:end].strip()
            if chunk_content:
                chunks.append(
                    self._create_chunk(
                        content=chunk_content,
                        chunk_index=start_index + len(chunks),
                        source_filename=source_filename,
                        storage_key=storage_key,
                        file_id=file_id,
                        chunk_type=ChunkType.TEXT,
                        segment_id=segment_id,
                        heading_context=heading_context,
                        content_type=content_type,
                        image_assets=image_assets,
                        page_number=page_number,
                        source_anchor=source_anchor,
                    )
                )

            if end >= len(normalized_text):
                break

            next_start = end - self._chunk_overlap
            start = next_start if next_start > start else end

        return chunks

    def _build_table_chunks(
        self,
        table_markdown: str,
        source_filename: str,
        storage_key: str,
        file_id: str,
        segment_id: str,
        start_index: int = 0,
        heading_context: str = "",
        comment_texts: list[str] | None = None,
        image_assets: list[ChunkImageAsset] | None = None,
        page_number: int | None = None,
        source_anchor: str | None = None,
    ) -> list[DocumentChunk]:
        """为表格生成独立的 chunks。

        小表格整体作为一个 chunk；
        大表格按行分组切分，每组补充表头。

        参数:
            table_markdown: 表格的 Markdown 字符串。
            source_filename: 上传时的原始文件名。
            storage_key: 文件在本地存储中的相对路径。
            file_id: 文件唯一标识。
            start_index: 当前批次切块写入前的起始序号。
            heading_context: 表格所在标题上下文，会附加到每个表格 chunk 前部。

        返回:
            表格切块列表。
        """
        normalized_heading = heading_context.strip()
        heading_prefix = f"{normalized_heading}\n" if normalized_heading else ""
        # 表格批注作为补充上下文追加到每个表格 chunk 尾部，避免破坏表格主体结构。
        comment_suffix = self._build_comment_suffix(comment_texts or [])
        content_with_context = (
            f"{heading_prefix}{table_markdown}{comment_suffix}"
            if heading_prefix
            else f"{table_markdown}{comment_suffix}"
        )

        if len(content_with_context) <= self._chunk_size:
            # 小表格，整体作为一个 chunk
            return [
                self._create_chunk(
                    content=content_with_context,
                    chunk_index=start_index,
                    source_filename=source_filename,
                    storage_key=storage_key,
                    file_id=file_id,
                    chunk_type=ChunkType.TEXT,
                    segment_id=segment_id,
                    heading_context=heading_context,
                    content_type="table",
                    image_assets=image_assets,
                    page_number=page_number,
                    source_anchor=source_anchor,
                )
            ]

        # 大表格，需要切分
        lines = table_markdown.split("\n")
        if len(lines) < 3:
            # 格式异常，当作小表格
            return [
                self._create_chunk(
                    content=content_with_context,
                    chunk_index=start_index,
                    source_filename=source_filename,
                    storage_key=storage_key,
                    file_id=file_id,
                    chunk_type=ChunkType.TEXT,
                    segment_id=segment_id,
                    heading_context=heading_context,
                    content_type="table",
                    image_assets=image_assets,
                    page_number=page_number,
                    source_anchor=source_anchor,
                )
            ]

        header_line = lines[0]
        separator_line = lines[1]
        split_tables = self._split_large_table(
            table_markdown,
            header_line,
            separator_line,
            heading_context=heading_context,
            comment_suffix=comment_suffix,
        )

        chunks: list[DocumentChunk] = []
        for offset, table_part in enumerate(split_tables):
            chunks.append(
                self._create_chunk(
                    content=table_part,
                    chunk_index=start_index + offset,
                    source_filename=source_filename,
                    storage_key=storage_key,
                    file_id=file_id,
                    chunk_type=ChunkType.TEXT,
                    segment_id=segment_id,
                    heading_context=heading_context,
                    content_type="table",
                    image_assets=image_assets,
                    page_number=page_number,
                    source_anchor=source_anchor,
                )
            )

        return chunks

    def _split_large_table(
        self,
        markdown: str,
        header_line: str,
        separator_line: str,
        heading_context: str = "",
        comment_suffix: str = "",
    ) -> list[str]:
        """将超大表格按行分组切分，每组补充表头。

        参数:
            markdown: 完整的表格 Markdown 字符串。
            header_line: 表头行（如 "| 列1 | 列2 |"）。
            separator_line: 分隔行（如 "|---|---|"）。
            heading_context: 表格所在标题上下文，会附加到每个切分结果前部。

        返回:
            切分后的表格 Markdown 列表，每个都包含表头。
        """
        lines = markdown.split("\n")
        if len(lines) <= 2:
            normalized_heading = heading_context.strip()
            if not normalized_heading:
                return [f"{markdown}{comment_suffix}"]
            return [f"{normalized_heading}\n{markdown}{comment_suffix}"]

        normalized_heading = heading_context.strip()
        heading_prefix = f"{normalized_heading}\n" if normalized_heading else ""
        header = f"{header_line}\n{separator_line}"
        chunk_prefix = f"{heading_prefix}{header}" if heading_prefix else header
        # 预留批注补充块的长度，避免拆分后 chunk 明显超过目标大小。
        header_size = len(chunk_prefix) + len(comment_suffix)
        data_rows = lines[2:]

        chunks: list[str] = []
        current_rows: list[str] = []
        current_size = header_size

        for row in data_rows:
            row_size = len(row) + 1  # +1 for newline

            if current_size + row_size > self._chunk_size and current_rows:
                chunk_content = chunk_prefix + "\n" + "\n".join(current_rows) + comment_suffix
                chunks.append(chunk_content)
                current_rows = []
                current_size = header_size

            current_rows.append(row)
            current_size += row_size

        if current_rows:
            chunk_content = chunk_prefix + "\n" + "\n".join(current_rows) + comment_suffix
            chunks.append(chunk_content)

        if chunks:
            return chunks

        return [f"{chunk_prefix}{comment_suffix}"]

    def _log_chunk_preview(
        self,
        file_id: str,
        source_filename: str,
        storage_key: str,
        chunks: list[DocumentChunk],
    ) -> None:
        """将切块摘要打印到控制台日志。

        参数:
            file_id: 当前文档文件标识。
            source_filename: 上传时的原始文件名。
            storage_key: 文件在本地存储中的相对路径。
            chunks: 已生成的切块列表。

        返回:
            None。日志会输出文件级摘要，以及每个 chunk 的多行预览信息。
        """
        char_counts = ", ".join(str(chunk.char_count) for chunk in chunks)
        LOGGER.info(
            (
                "document_chunk_preview\n"
                "  file_id          : %s\n"
                "  filename         : %s\n"
                "  storage_key      : %s\n"
                "  chunk_count      : %s\n"
                "  chunk_char_counts: [%s]"
            ),
            file_id,
            source_filename,
            storage_key,
            len(chunks),
            char_counts,
        )

        for chunk in chunks:
            LOGGER.info("%s", self._format_chunk_log(chunk))

    def _build_es_preview_document(self, chunk: DocumentChunk) -> dict[str, object]:
        """构造用于 ES 检索的预览文档。

        参数:
            chunk: 已生成的标准化 chunk 对象。

        返回:
            基于当前 chunk 文本和元数据构造的 ES 预览文档。
            当前已包含内容检索与领域词增强字段。
        """
        return chunk.to_search_document()

    def _format_chunk_log(self, chunk: DocumentChunk) -> str:
        """把单个 chunk 渲染为便于控制台阅读的多行文本。"""
        content_preview = self._format_log_multiline_value(
            self._build_content_preview(chunk.content),
            indent="    ",
        )
        merged_terms = ", ".join(chunk.merged_terms) if chunk.merged_terms else "-"

        return (
            "document_chunk_item\n"
            f"  chunk_index      : {chunk.chunk_index}\n"
            f"  chunk_id         : {chunk.chunk_id}\n"
            f"  char_count       : {chunk.char_count}\n"
            f"  merged_terms     : {merged_terms}\n"
            "  content_preview  :\n"
            f"{content_preview}"
        )

    def _build_content_preview(self, content: str, max_chars: int = 220) -> str:
        """构造日志中的正文预览，避免把整段内容直接铺满控制台。"""
        normalized = content.strip()
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[:max_chars]}...(已截断，共{len(normalized)}字符)"

    def _format_log_multiline_value(self, value: str, indent: str) -> str:
        """把多行文本包装并缩进，便于在控制台中按字段阅读。"""
        wrapped_lines: list[str] = []
        for raw_line in value.splitlines() or [""]:
            line = raw_line.strip()
            if not line:
                wrapped_lines.append(indent)
                continue

            wrapped_lines.extend(
                f"{indent}{wrapped_line}"
                for wrapped_line in textwrap.wrap(
                    line,
                    width=88,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )

        return "\n".join(wrapped_lines)
