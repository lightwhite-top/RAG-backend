"""PDF 解析器。"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from io import BytesIO
from statistics import median
from typing import cast

import pymupdf
from PIL import Image

from baozhi_rag.services.document_ocr.aliyun_ocr_client import AliyunOcrClientProtocol

BoundingBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PdfParsedImageAsset:
    """PDF 解析阶段的图片资产。"""

    segment_id: str
    asset_id: str
    asset_index: int
    source_anchor: str
    content_type: str
    extension: str
    image_bytes: bytes
    width: int | None = None
    height: int | None = None


@dataclass
class PdfParsedSegment:
    """PDF 解析阶段的标准化片段。"""

    segment_id: str
    content: str
    segment_type: str
    heading_context: str
    page_number: int | None = None
    source_anchor: str | None = None
    bbox: BoundingBox | None = None
    image_assets: list[PdfParsedImageAsset] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _PdfTextBlock:
    """内部文本块表示。"""

    text: str
    bbox: BoundingBox
    font_size: float


class PdfParserError(Exception):
    """PDF 解析器异常。"""


@dataclass(slots=True)
class PdfDocumentParser:
    """基于 PyMuPDF 的 PDF 解析器。"""

    render_dpi: int = 180
    low_text_page_threshold: int = 80
    max_page_count: int = 300
    ocr_enabled: bool = True
    ocr_client: AliyunOcrClientProtocol | None = None

    def parse(self, *, file_path: str, source_filename: str) -> list[PdfParsedSegment]:
        """解析 PDF，返回标准化片段。"""
        try:
            document = pymupdf.open(file_path)
        except Exception as exc:  # pragma: no cover - 三方异常类型不稳定
            raise PdfParserError(f"解析 PDF 文件失败: {source_filename}") from exc

        with document:
            if document.needs_pass:
                raise PdfParserError(f"PDF 文件已加密，暂不支持解析: {source_filename}")
            if document.page_count <= 0:
                raise PdfParserError(f"PDF 文件内容为空，无法解析: {source_filename}")
            if document.page_count > self.max_page_count:
                raise PdfParserError(
                    f"PDF 页数超出限制: {source_filename} page_count={document.page_count}"
                )

            segments: list[PdfParsedSegment] = []
            headings: list[str] = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_number = page_index + 1
                page_segments, headings = self._parse_page(
                    document=document,
                    page=page,
                    page_number=page_number,
                    headings=headings,
                )
                segments.extend(page_segments)

            if not segments:
                raise PdfParserError(f"PDF 文件内容为空，无法解析: {source_filename}")
            return segments

    def _parse_page(
        self,
        *,
        document: pymupdf.Document,
        page: pymupdf.Page,
        page_number: int,
        headings: list[str],
    ) -> tuple[list[PdfParsedSegment], list[str]]:
        """解析单页内容。"""
        image_assets = self._extract_page_image_assets(
            document=document, page=page, page_number=page_number
        )
        text_blocks = self._extract_text_blocks(page)
        total_text_length = sum(len(block.text) for block in text_blocks)

        if total_text_length >= self.low_text_page_threshold:
            segments, next_headings = self._parse_digital_page(
                text_blocks=text_blocks,
                page_number=page_number,
                headings=headings,
            )
        else:
            segments, next_headings = self._parse_scanned_page(
                page=page,
                page_number=page_number,
                headings=headings,
            )

        self._attach_image_assets_to_segments(
            segments=segments,
            image_assets=image_assets,
            page_number=page_number,
        )
        return segments, next_headings

    def _extract_text_blocks(self, page: pymupdf.Page) -> list[_PdfTextBlock]:
        """提取页内文本块。"""
        page_dict = page.get_text("dict")
        blocks: list[_PdfTextBlock] = []
        if not isinstance(page_dict, dict):
            return blocks

        for raw_block in page_dict.get("blocks", []):
            if not isinstance(raw_block, dict) or int(raw_block.get("type", 0)) != 0:
                continue
            bbox = self._read_bbox(raw_block.get("bbox"))
            if bbox is None:
                continue
            texts: list[str] = []
            font_sizes: list[float] = []
            for line in raw_block.get("lines", []):
                if not isinstance(line, dict):
                    continue
                line_text_parts: list[str] = []
                for span in line.get("spans", []):
                    if not isinstance(span, dict):
                        continue
                    span_text = str(span.get("text", "")).strip()
                    if not span_text:
                        continue
                    line_text_parts.append(span_text)
                    try:
                        font_sizes.append(float(span.get("size", 0.0)))
                    except (TypeError, ValueError):
                        continue
                line_text = " ".join(part for part in line_text_parts if part).strip()
                if line_text:
                    texts.append(line_text)
            normalized_text = "\n".join(texts).strip()
            if not normalized_text:
                continue
            blocks.append(
                _PdfTextBlock(
                    text=normalized_text,
                    bbox=bbox,
                    font_size=median(font_sizes) if font_sizes else 0.0,
                )
            )

        return self._restore_reading_order(blocks=blocks, page_width=float(page.rect.width))

    def _restore_reading_order(
        self,
        *,
        blocks: list[_PdfTextBlock],
        page_width: float,
    ) -> list[_PdfTextBlock]:
        """按保守规则恢复阅读顺序。"""
        if len(blocks) < 2:
            return sorted(blocks, key=lambda item: (item.bbox[1], item.bbox[0]))

        left_blocks = [block for block in blocks if block.bbox[0] < page_width * 0.45]
        right_blocks = [block for block in blocks if block.bbox[0] > page_width * 0.55]
        full_width_blocks = [
            block for block in blocks if block not in left_blocks and block not in right_blocks
        ]

        if len(left_blocks) >= 2 and len(right_blocks) >= 2 and len(full_width_blocks) <= 2:
            return sorted(left_blocks, key=lambda item: (item.bbox[1], item.bbox[0])) + sorted(
                right_blocks, key=lambda item: (item.bbox[1], item.bbox[0])
            )

        return sorted(blocks, key=lambda item: (item.bbox[1], item.bbox[0]))

    def _parse_digital_page(
        self,
        *,
        text_blocks: list[_PdfTextBlock],
        page_number: int,
        headings: list[str],
    ) -> tuple[list[PdfParsedSegment], list[str]]:
        """解析数字版页面。"""
        segments: list[PdfParsedSegment] = []
        active_headings = list(headings)
        median_font_size = (
            median([block.font_size for block in text_blocks if block.font_size > 0.0])
            if any(block.font_size > 0.0 for block in text_blocks)
            else 0.0
        )

        for block_index, block in enumerate(text_blocks, start=1):
            source_anchor = f"p:{page_number}:block:{block_index}"
            heading_level = self._resolve_heading_level(
                text=block.text,
                font_size=block.font_size,
                baseline_font_size=median_font_size,
            )
            if heading_level is not None:
                active_headings = active_headings[: heading_level - 1] + [block.text]
                heading_context = " / ".join(active_headings)
                segments.append(
                    PdfParsedSegment(
                        segment_id=source_anchor,
                        content=heading_context,
                        segment_type="paragraph",
                        heading_context=heading_context,
                        page_number=page_number,
                        source_anchor=source_anchor,
                        bbox=block.bbox,
                    )
                )
                continue

            heading_context = " / ".join(active_headings)
            content = f"{heading_context}\n{block.text}" if heading_context else block.text
            segments.append(
                PdfParsedSegment(
                    segment_id=source_anchor,
                    content=content,
                    segment_type="paragraph",
                    heading_context=heading_context,
                    page_number=page_number,
                    source_anchor=source_anchor,
                    bbox=block.bbox,
                )
            )

        return segments, active_headings

    def _parse_scanned_page(
        self,
        *,
        page: pymupdf.Page,
        page_number: int,
        headings: list[str],
    ) -> tuple[list[PdfParsedSegment], list[str]]:
        """解析扫描版页面。"""
        if not self.ocr_enabled or self.ocr_client is None:
            raise PdfParserError(
                f"检测到扫描版 PDF 页面，但当前未启用 OCR: page_number={page_number}"
            )

        page_image_bytes = self._render_page_image(page)
        structure_result = self.ocr_client.recognize_page_structure(image_bytes=page_image_bytes)
        segments: list[PdfParsedSegment] = []
        active_headings = list(headings)

        for block_index, block in enumerate(structure_result.blocks, start=1):
            source_anchor = f"p:{page_number}:block:{block_index}"
            if block.block_type == "figure":
                continue
            if block.block_type == "table":
                table_text = ""
                if block.bbox is not None:
                    cropped_bytes = self._crop_region(
                        image_bytes=page_image_bytes,
                        bbox=block.bbox,
                    )
                    table_text = self.ocr_client.recognize_table_text(image_bytes=cropped_bytes)
                markdown = self._build_fallback_table_markdown(table_text)
                segments.append(
                    PdfParsedSegment(
                        segment_id=source_anchor,
                        content=markdown,
                        segment_type="table",
                        heading_context=" / ".join(active_headings),
                        page_number=page_number,
                        source_anchor=source_anchor,
                        bbox=block.bbox,
                    )
                )
                continue

            if block.block_type == "title":
                active_headings = active_headings[:0] + [block.text]
                heading_context = " / ".join(active_headings)
                segments.append(
                    PdfParsedSegment(
                        segment_id=source_anchor,
                        content=heading_context,
                        segment_type="paragraph",
                        heading_context=heading_context,
                        page_number=page_number,
                        source_anchor=source_anchor,
                        bbox=block.bbox,
                    )
                )
                continue

            heading_context = " / ".join(active_headings)
            content = f"{heading_context}\n{block.text}" if heading_context else block.text
            segments.append(
                PdfParsedSegment(
                    segment_id=source_anchor,
                    content=content,
                    segment_type="paragraph",
                    heading_context=heading_context,
                    page_number=page_number,
                    source_anchor=source_anchor,
                    bbox=block.bbox,
                )
            )

        if segments:
            return segments, active_headings

        ocr_text = self.ocr_client.recognize_text(image_bytes=page_image_bytes, advanced=True)
        normalized_heading_context = " / ".join(active_headings)
        content = (
            f"{normalized_heading_context}\n{ocr_text}" if normalized_heading_context else ocr_text
        )
        if not content.strip():
            raise PdfParserError(f"扫描版 PDF OCR 结果为空: page_number={page_number}")
        return (
            [
                PdfParsedSegment(
                    segment_id=f"p:{page_number}:block:1",
                    content=content,
                    segment_type="paragraph",
                    heading_context=normalized_heading_context,
                    page_number=page_number,
                    source_anchor=f"p:{page_number}:block:1",
                )
            ],
            active_headings,
        )

    def _extract_page_image_assets(
        self,
        *,
        document: pymupdf.Document,
        page: pymupdf.Page,
        page_number: int,
    ) -> list[PdfParsedImageAsset]:
        """抽取页内原始图片对象。"""
        assets: list[PdfParsedImageAsset] = []
        asset_index = 1
        for image_info in page.get_images(full=True):
            if not image_info:
                continue
            xref = int(image_info[0])
            try:
                extracted = document.extract_image(xref)
            except Exception:  # pragma: no cover - 第三方异常类型不稳定
                continue
            image_bytes = extracted.get("image")
            if not isinstance(image_bytes, bytes) or not image_bytes:
                continue
            extension = f".{str(extracted.get('ext', 'bin')).strip() or 'bin'}"
            content_type = mimetypes.guess_type(f"image{extension}", strict=False)[0]
            rects = page.get_image_rects(xref)
            if not rects:
                rects = [page.rect]
            for _rect in rects:
                source_anchor = f"p:{page_number}:image:{asset_index}"
                assets.append(
                    PdfParsedImageAsset(
                        segment_id=source_anchor,
                        asset_id=source_anchor,
                        asset_index=asset_index,
                        source_anchor=source_anchor,
                        content_type=content_type or "application/octet-stream",
                        extension=extension,
                        image_bytes=image_bytes,
                        width=self._safe_int(extracted.get("width")),
                        height=self._safe_int(extracted.get("height")),
                    )
                )
                asset_index += 1
        return assets

    def _attach_image_assets_to_segments(
        self,
        *,
        segments: list[PdfParsedSegment],
        image_assets: list[PdfParsedImageAsset],
        page_number: int,
    ) -> None:
        """把图片资产挂到距离最近的片段上。"""
        if not image_assets:
            return

        if not segments:
            segments.append(
                PdfParsedSegment(
                    segment_id=f"p:{page_number}:block:image",
                    content=f"第{page_number}页图片区域",
                    segment_type="paragraph",
                    heading_context="",
                    page_number=page_number,
                    source_anchor=f"p:{page_number}:block:image",
                    image_assets=list(image_assets),
                )
            )
            return

        for asset in image_assets:
            target_segment = min(
                segments,
                key=lambda segment: self._segment_distance(segment_bbox=segment.bbox),
            )
            target_segment.image_assets.append(asset)

    def _segment_distance(self, *, segment_bbox: BoundingBox | None) -> float:
        """为图片资产选最近片段时的保守距离。"""
        if segment_bbox is None:
            return float("inf")
        return segment_bbox[1]

    def _resolve_heading_level(
        self,
        *,
        text: str,
        font_size: float,
        baseline_font_size: float,
    ) -> int | None:
        """基于字体大小与文本长度做标题层级估计。"""
        normalized_text = text.strip()
        if not normalized_text:
            return None
        if baseline_font_size <= 0:
            return None
        if len(normalized_text) > 60:
            return None
        if font_size >= baseline_font_size * 1.45:
            return 1
        if font_size >= baseline_font_size * 1.2:
            return 2
        return None

    def _render_page_image(self, page: pymupdf.Page) -> bytes:
        """将页渲染为 PNG 字节流。"""
        pixmap = page.get_pixmap(dpi=self.render_dpi, alpha=False)
        return cast(bytes, pixmap.tobytes("png"))

    def _crop_region(self, *, image_bytes: bytes, bbox: BoundingBox) -> bytes:
        """按 OCR 返回的 bbox 裁切区域。"""
        with Image.open(BytesIO(image_bytes)) as image:
            x0, y0, x1, y1 = bbox
            cropped = image.crop((max(x0, 0), max(y0, 0), max(x1, 0), max(y1, 0)))
            buffer = BytesIO()
            cropped.save(buffer, format="PNG")
            return buffer.getvalue()

    def _build_fallback_table_markdown(self, table_text: str) -> str:
        """把 OCR 表格文本退化为一列表格 Markdown。"""
        normalized_lines = [
            line.replace("|", "／").strip() for line in table_text.splitlines() if line.strip()
        ]
        if not normalized_lines:
            normalized_lines = ["未识别到可用表格文本"]
        rows = "\n".join(f"| {line} |" for line in normalized_lines)
        return "| OCR表格文本 |\n|---|\n" + rows

    @staticmethod
    def _read_bbox(raw_bbox: object) -> BoundingBox | None:
        """安全读取 bbox。"""
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
            return None
        try:
            return (
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: object) -> int | None:
        """安全读取整数值。"""
        if not isinstance(value, (str, bytes, bytearray, int, float)):
            return None
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
