"""Word 文档切块服务。"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.document import Document as DocxDocument

from baozhi_rag.services.term_matching import MaximumMatchingTermMatcher, build_default_term_matcher

LOGGER = logging.getLogger(__name__)


class DocumentChunkingError(Exception):
    """文档切块失败。"""


class UnsupportedDocumentTypeError(DocumentChunkingError):
    """不支持的文档格式。"""


class DocumentConversionError(DocumentChunkingError):
    """文档格式转换失败。"""


class DocumentParseError(DocumentChunkingError):
    """文档内容解析失败。"""


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """标准化切块结果。"""

    file_id: str
    chunk_id: str
    chunk_index: int
    content: str
    char_count: int
    source_filename: str
    storage_key: str
    fmm_terms: list[str] = field(default_factory=list)
    bmm_terms: list[str] = field(default_factory=list)
    merged_terms: list[str] = field(default_factory=list)

    def to_search_document(self) -> dict[str, object]:
        """构造 ES 入库文档。"""
        return {
            "chunk_id": self.chunk_id,
            "file_id": self.file_id,
            "source_filename": self.source_filename,
            "storage_key": self.storage_key,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "content": self.content,
            "fmm_terms": self.fmm_terms,
            "bmm_terms": self.bmm_terms,
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
        headings: list[str] = []
        segments: list[str] = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue

            style_name = paragraph.style.name if paragraph.style else ""
            heading_level = self._parse_heading_level(style_name)
            if heading_level is not None:
                headings = headings[: heading_level - 1]
                headings.append(text)
                segments.append(" / ".join(headings))
                continue

            context = " / ".join(headings)
            segments.append(f"{context}\n{text}" if context else text)

        return segments

    def _parse_heading_level(self, style_name: str | None) -> int | None:
        """解析标题样式级别。

        参数:
            style_name: 段落样式名称，例如 `Heading 1` 或 `标题 1`；为空时返回 None。

        返回:
            若样式表示标题则返回其层级，否则返回 None。
        """
        if style_name is None:
            return None

        matched = re.search(r"(Heading|标题)\s*(\d+)", style_name, flags=re.IGNORECASE)
        if matched is None:
            return None
        return int(matched.group(2))

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
                        fmm_terms=term_match_result.fmm_terms,
                        bmm_terms=term_match_result.bmm_terms,
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
        source_filename: str,
        storage_key: str,
        chunks: list[DocumentChunk],
    ) -> None:
        """将切块摘要打印到控制台日志。

        参数:
            source_filename: 上传时的原始文件名。
            storage_key: 文件在本地存储中的相对路径。
            chunks: 已生成的切块列表。

        返回:
            None。日志会输出 chunk 总数、每个 chunk 的完整正文，以及对应的 ES 预览文档。
        """
        LOGGER.info(
            "document_chunk_preview filename=%s storage_key=%s chunk_count=%s",
            source_filename,
            storage_key,
            len(chunks),
        )

        for chunk in chunks:
            LOGGER.info(
                (
                    "document_chunk_item_full filename=%s chunk_index=%s "
                    "chunk_id=%s char_count=%s content=\n%s"
                ),
                source_filename,
                chunk.chunk_index,
                chunk.chunk_id,
                chunk.char_count,
                chunk.content,
            )
            LOGGER.info(
                "document_chunk_es_preview %s",
                json.dumps(
                    self._build_es_preview_document(chunk),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )

    def _build_es_preview_document(self, chunk: DocumentChunk) -> dict[str, object]:
        """构造用于 ES 检索的预览文档。

        参数:
            chunk: 已生成的标准化 chunk 对象。

        返回:
            基于当前 chunk 文本和元数据构造的 ES 预览文档。
            当前已包含内容检索与 FMM/BMM 词项增强字段。
        """
        return chunk.to_search_document()
