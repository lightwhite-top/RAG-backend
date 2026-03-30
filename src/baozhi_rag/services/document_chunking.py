"""Word 文档切块服务。"""

from __future__ import annotations

import logging
import re
import subprocess
import textwrap
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.text.paragraph import CT_P
from docx.text.paragraph import Paragraph
from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.term_matching import MaximumMatchingTermMatcher, build_default_term_matcher

LOGGER = logging.getLogger(__name__)


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
    # 切块后的正文内容
    content: str
    # 切块的字符数
    char_count: int
    # 上传时的原始文件名
    source_filename: str
    # 文件在本地存储中的相对路径
    storage_key: str
    # 基于领域词词典抽取出的去重词项，用于检索时显式提权。
    merged_terms: list[str] = field(default_factory=list)
    # 向量化
    content_embedding: list[float] | None = None

    def to_search_document(self) -> dict[str, object]:
        """构造 ES 文本检索入库文档。"""
        return {
            "chunk_id": self.chunk_id,
            "file_id": self.file_id,
            "source_filename": self.source_filename,
            "storage_key": self.storage_key,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "content": self.content,
            "merged_terms": self.merged_terms,
        }


class DocumentChunkService:
    """负责 Word 文档解析、切块与预览日志输出。"""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        convert_temp_dir: Path,
        term_matcher: MaximumMatchingTermMatcher | None = None,
    ) -> None:
        """初始化切块服务。

        参数:
            chunk_size: 单个切块的目标最大字符数。
            chunk_overlap: 相邻切块之间保留的重叠字符数。
            convert_temp_dir: `.doc` 转 `.docx` 时使用的临时目录。
            term_matcher: 领域词匹配器；未传时使用默认金融保险词典。

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

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._convert_temp_dir = convert_temp_dir
        self._term_matcher = term_matcher or build_default_term_matcher()

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

        return self._build_chunks(
            text="\n\n".join(segments),
            source_filename=source_filename,
            storage_key=storage_key,
            file_id=file_id,
        )

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
            )
        except FileNotFoundError as exc:
            msg = "未找到 soffice，无法解析 .doc 文件"
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

    def _extract_docx_segments(self, document: DocxDocument) -> list[str]:
        """抽取 docx 段落并保留标题上下文。

        参数:
            document: `python-docx` 解析后的文档对象。

        返回:
            去除空段落后、带标题上下文的文本片段列表。
        """
        # 维护一个滑动窗口式的标题层级上下文，当遇到标题段时更新上下文并将其与后续正文段合并，形成带上下文的文本片段。
        headings: list[str] = []
        # 抽取段落文本并合并标题上下文，形成切块前的文本片段列表
        segments: list[str] = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue

            # 优先使用大纲级别识别标题，样式名作为兜底
            heading_level = self._get_heading_level(paragraph)
            if heading_level is not None:
                # 滑动窗口式维护当前标题层级上下文
                headings = headings[: heading_level - 1]
                headings.append(text)
                segments.append(" / ".join(headings))
                continue

            context = " / ".join(headings)
            segments.append(f"{context}\n{text}" if context else text)

        return segments

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
        xml_element = paragraph._element
        if isinstance(xml_element, CT_P):
            p_pr = xml_element.pPr
            if p_pr is not None:
                p_style = p_pr.pStyle
                if p_style is not None:
                    style_val = p_style.val
                    if style_val is not None:
                        level = self._parse_heading_level_by_name(style_val)
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

    def _build_chunks(
        self,
        text: str,
        source_filename: str,
        storage_key: str,
        file_id: str,
    ) -> list[DocumentChunk]:
        """按固定窗口与 overlap 生成切块。

        参数:
            text: 已完成段落拼接和标题补全的正文文本。
            source_filename: 上传时的原始文件名。
            storage_key: 文件在本地存储中的相对路径。
            file_id: 文件唯一标识。

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
                chunk_index = len(chunks)
                term_match_result = self._term_matcher.extract_terms(chunk_content)
                chunks.append(
                    DocumentChunk(
                        file_id=file_id,
                        chunk_id=f"{file_id}-chunk-{chunk_index}",
                        chunk_index=chunk_index,
                        content=chunk_content,
                        char_count=len(chunk_content),
                        source_filename=source_filename,
                        storage_key=storage_key,
                        merged_terms=term_match_result.merged_terms,
                    )
                )

            if end >= len(normalized_text):
                break

            next_start = end - self._chunk_overlap
            start = next_start if next_start > start else end

        return chunks

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
