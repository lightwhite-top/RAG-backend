"""PDF 解析器。"""

from __future__ import annotations

import hashlib
import mimetypes
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from io import BytesIO
from statistics import median
from typing import Any, cast

import pymupdf
from PIL import Image

from baozhi_rag.services.document_ocr.ocr_capacity_limiter import OcrTaskLimiterProtocol
from baozhi_rag.services.document_ocr.ocr_client_protocol import OcrClientProtocol

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
    bbox: BoundingBox | None = None


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
    ocr_client: OcrClientProtocol | None = None
    ocr_task_limiter: OcrTaskLimiterProtocol | None = None

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
        with self._reserve_ocr_capacity(page_number=page_number):
            structure_result = self.ocr_client.recognize_page_structure(
                image_bytes=page_image_bytes
            )
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

            enhanced_text = self.ocr_client.recognize_text(
                image_bytes=page_image_bytes,
                advanced=True,
            )
        if segments and self._should_prefer_enhanced_ocr_text(
            segments=segments,
            enhanced_text=enhanced_text,
        ):
            normalized_heading_context = " / ".join(active_headings)
            content = (
                f"{normalized_heading_context}\n{enhanced_text}"
                if normalized_heading_context
                else enhanced_text
            )
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

        if segments:
            return segments, active_headings

        ocr_text = enhanced_text
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
        seen_asset_keys: set[tuple[str, tuple[float, float, float, float] | None]] = set()
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
            try:
                rects = page.get_image_rects(xref)
            except Exception:  # pragma: no cover - 第三方异常类型不稳定
                rects = []
            bbox_list = [self._read_bbox(rect) for rect in rects]
            normalized_bboxes: list[BoundingBox | None] = [
                bbox for bbox in bbox_list if bbox is not None
            ]
            if not normalized_bboxes:
                page_bbox = self._read_bbox(page.rect)
                normalized_bboxes = [page_bbox] if page_bbox is not None else [None]
            for bbox in normalized_bboxes:
                asset_index = self._append_page_image_asset(
                    assets=assets,
                    seen_asset_keys=seen_asset_keys,
                    asset_index=asset_index,
                    page_number=page_number,
                    image_bytes=image_bytes,
                    content_type=content_type or "application/octet-stream",
                    extension=extension,
                    width=self._safe_int(extracted.get("width")),
                    height=self._safe_int(extracted.get("height")),
                    bbox=bbox,
                )

        return self._extract_page_image_assets_from_blocks(
            page=page,
            page_number=page_number,
            assets=assets,
            seen_asset_keys=seen_asset_keys,
            next_asset_index=asset_index,
        )

    def _extract_page_image_assets_from_blocks(
        self,
        *,
        page: pymupdf.Page,
        page_number: int,
        assets: list[PdfParsedImageAsset],
        seen_asset_keys: set[tuple[str, tuple[float, float, float, float] | None]],
        next_asset_index: int,
    ) -> list[PdfParsedImageAsset]:
        """用 text-dict 图片块作为兜底，补齐 `get_images()` 可能漏掉的图片。"""
        try:
            page_dict = page.get_text("dict")
        except Exception:  # pragma: no cover - 第三方异常类型不稳定
            return assets
        if not isinstance(page_dict, dict):
            return assets

        asset_index = next_asset_index
        page_bbox = self._read_bbox(page.rect)
        for raw_block in page_dict.get("blocks", []):
            if not isinstance(raw_block, dict) or int(raw_block.get("type", 0)) != 1:
                continue

            image_bytes = raw_block.get("image")
            if not isinstance(image_bytes, bytes) or not image_bytes:
                continue

            extension = f".{str(raw_block.get('ext', 'bin')).strip() or 'bin'}"
            content_type = mimetypes.guess_type(f"image{extension}", strict=False)[0]
            block_bbox = self._read_bbox(raw_block.get("bbox"))
            if (
                page_bbox is not None
                and block_bbox is not None
                and self._replace_page_bbox_placeholder_asset(
                    assets=assets,
                    seen_asset_keys=seen_asset_keys,
                    image_bytes=image_bytes,
                    page_bbox=page_bbox,
                    actual_bbox=block_bbox,
                    content_type=content_type or "application/octet-stream",
                    extension=extension,
                    width=self._safe_int(raw_block.get("width")),
                    height=self._safe_int(raw_block.get("height")),
                )
            ):
                continue
            asset_index = self._append_page_image_asset(
                assets=assets,
                seen_asset_keys=seen_asset_keys,
                asset_index=asset_index,
                page_number=page_number,
                image_bytes=image_bytes,
                content_type=content_type or "application/octet-stream",
                extension=extension,
                width=self._safe_int(raw_block.get("width")),
                height=self._safe_int(raw_block.get("height")),
                bbox=block_bbox,
            )

        return assets

    def _replace_page_bbox_placeholder_asset(
        self,
        *,
        assets: list[PdfParsedImageAsset],
        seen_asset_keys: set[tuple[str, tuple[float, float, float, float] | None]],
        image_bytes: bytes,
        page_bbox: BoundingBox,
        actual_bbox: BoundingBox,
        content_type: str,
        extension: str,
        width: int | None,
        height: int | None,
    ) -> bool:
        """把仅能定位到整页的占位图片资产升级为更精确的图片块坐标。"""
        placeholder_key = self._build_image_asset_dedupe_key(
            image_bytes=image_bytes,
            bbox=page_bbox,
        )
        actual_key = self._build_image_asset_dedupe_key(
            image_bytes=image_bytes,
            bbox=actual_bbox,
        )
        if placeholder_key not in seen_asset_keys or actual_key in seen_asset_keys:
            return False

        for index, asset in enumerate(assets):
            if (
                self._build_image_asset_dedupe_key(
                    image_bytes=asset.image_bytes,
                    bbox=asset.bbox,
                )
                != placeholder_key
            ):
                continue

            assets[index] = replace(
                asset,
                content_type=content_type,
                extension=extension,
                width=width if width is not None else asset.width,
                height=height if height is not None else asset.height,
                bbox=actual_bbox,
            )
            seen_asset_keys.discard(placeholder_key)
            seen_asset_keys.add(actual_key)
            return True

        return False

    def _append_page_image_asset(
        self,
        *,
        assets: list[PdfParsedImageAsset],
        seen_asset_keys: set[tuple[str, tuple[float, float, float, float] | None]],
        asset_index: int,
        page_number: int,
        image_bytes: bytes,
        content_type: str,
        extension: str,
        width: int | None,
        height: int | None,
        bbox: BoundingBox | None,
    ) -> int:
        """向图片资产列表中追加去重后的单个图片。"""
        dedupe_key = self._build_image_asset_dedupe_key(image_bytes=image_bytes, bbox=bbox)
        if dedupe_key in seen_asset_keys:
            return asset_index

        seen_asset_keys.add(dedupe_key)
        source_anchor = f"p:{page_number}:image:{asset_index}"
        assets.append(
            PdfParsedImageAsset(
                segment_id=source_anchor,
                asset_id=source_anchor,
                asset_index=asset_index,
                source_anchor=source_anchor,
                content_type=content_type,
                extension=extension,
                image_bytes=image_bytes,
                width=width,
                height=height,
                bbox=bbox,
            )
        )
        return asset_index + 1

    @staticmethod
    def _build_image_asset_dedupe_key(
        *,
        image_bytes: bytes,
        bbox: BoundingBox | None,
    ) -> tuple[str, tuple[float, float, float, float] | None]:
        """为图片资产构造稳定去重键，避免主路径与兜底路径重复挂载。"""
        bbox_key = (
            (
                round(bbox[0], 3),
                round(bbox[1], 3),
                round(bbox[2], 3),
                round(bbox[3], 3),
            )
            if bbox is not None
            else None
        )
        return hashlib.sha1(image_bytes).hexdigest(), bbox_key

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
            page_fallback_segment = self._ensure_image_fallback_segment(
                segments=segments,
                page_number=page_number,
            )
            page_fallback_segment.image_assets.extend(
                replace(asset, segment_id=page_fallback_segment.segment_id)
                for asset in image_assets
            )
            return

        fallback_segment: PdfParsedSegment | None = None
        for asset in image_assets:
            target_segment = self._select_image_asset_target_segment(
                segments=segments,
                asset=asset,
            )
            if target_segment is None:
                if fallback_segment is None:
                    fallback_segment = self._ensure_image_fallback_segment(
                        segments=segments,
                        page_number=page_number,
                    )
                target_segment = fallback_segment
            target_segment.image_assets.append(replace(asset, segment_id=target_segment.segment_id))

    def _ensure_image_fallback_segment(
        self,
        *,
        segments: list[PdfParsedSegment],
        page_number: int,
    ) -> PdfParsedSegment:
        """确保当前页存在可挂载图片的兜底片段。"""
        synthetic_segment_id = f"p:{page_number}:block:image"
        for segment in segments:
            if segment.segment_id == synthetic_segment_id:
                return segment

        fallback_segment = PdfParsedSegment(
            segment_id=synthetic_segment_id,
            content=f"第{page_number}页图片区域",
            segment_type="paragraph",
            heading_context="",
            page_number=page_number,
            source_anchor=synthetic_segment_id,
        )
        segments.append(fallback_segment)
        return fallback_segment

    def _select_image_asset_target_segment(
        self,
        *,
        segments: list[PdfParsedSegment],
        asset: PdfParsedImageAsset,
    ) -> PdfParsedSegment | None:
        """优先按空间位置选择最接近的片段，失败时再回退。"""
        segments_with_bbox = [segment for segment in segments if segment.bbox is not None]
        asset_bbox = asset.bbox
        if asset_bbox is not None and segments_with_bbox:
            return min(
                segments_with_bbox,
                key=lambda segment: self._bbox_distance(
                    source_bbox=asset_bbox,
                    target_bbox=segment.bbox,
                ),
            )

        if len(segments) == 1:
            return segments[0]
        if len(segments_with_bbox) == 1:
            return segments_with_bbox[0]
        return None

    @staticmethod
    def _bbox_distance(
        *,
        source_bbox: BoundingBox,
        target_bbox: BoundingBox | None,
    ) -> float:
        """计算图片框与文本框之间的保守距离，优先命中重叠或相邻区域。"""
        if target_bbox is None:
            return float("inf")

        horizontal_gap = max(
            target_bbox[0] - source_bbox[2],
            source_bbox[0] - target_bbox[2],
            0.0,
        )
        vertical_gap = max(
            target_bbox[1] - source_bbox[3],
            source_bbox[1] - target_bbox[3],
            0.0,
        )
        source_center_x = (source_bbox[0] + source_bbox[2]) / 2
        source_center_y = (source_bbox[1] + source_bbox[3]) / 2
        target_center_x = (target_bbox[0] + target_bbox[2]) / 2
        target_center_y = (target_bbox[1] + target_bbox[3]) / 2
        center_offset = abs(source_center_x - target_center_x) + abs(
            source_center_y - target_center_y
        )
        return horizontal_gap + vertical_gap + center_offset * 0.01

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
            left = max(min(x0, x1), 0.0)
            top = max(min(y0, y1), 0.0)
            right = min(max(x0, x1), float(image.width))
            bottom = min(max(y0, y1), float(image.height))
            if right <= left or bottom <= top:
                # OCR 返回异常框时回退整图，避免表格二次 OCR 直接丢失内容。
                return image_bytes
            cropped = image.crop((left, top, right, bottom))
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

    def _should_prefer_enhanced_ocr_text(
        self,
        *,
        segments: list[PdfParsedSegment],
        enhanced_text: str,
    ) -> bool:
        """在扫描页结构化结果明显劣化时，回退到全文增强 OCR。

        参数:
            segments: 当前基于页结构识别得到的片段列表。
            enhanced_text: 全页增强 OCR 结果。

        返回:
            当增强 OCR 更可能保留完整正文时返回 `True`，否则返回 `False`。
        """
        normalized_enhanced_text = enhanced_text.strip()
        if not normalized_enhanced_text:
            return False
        if any(segment.segment_type != "paragraph" for segment in segments):
            return False
        if any(segment.heading_context.strip() for segment in segments):
            return False

        structured_text = "\n".join(
            segment.content.strip() for segment in segments if segment.content
        ).strip()
        if not structured_text:
            return True

        structured_digit_count = sum(char.isdigit() for char in structured_text)
        enhanced_digit_count = sum(char.isdigit() for char in normalized_enhanced_text)
        if enhanced_digit_count > structured_digit_count:
            return True
        return len(normalized_enhanced_text) >= len(structured_text) + 12

    def _reserve_ocr_capacity(self, *, page_number: int) -> AbstractContextManager[None]:
        """在真正进入扫描页 OCR 前申请本地 OCR 容量。"""
        if self.ocr_task_limiter is None:
            return nullcontext()
        return self.ocr_task_limiter.reserve_slot(page_number=page_number)

    @staticmethod
    def _read_bbox(raw_bbox: object) -> BoundingBox | None:
        """安全读取 bbox。"""
        if raw_bbox is None:
            return None
        try:
            if (
                hasattr(raw_bbox, "x0")
                and hasattr(raw_bbox, "y0")
                and hasattr(raw_bbox, "x1")
                and hasattr(raw_bbox, "y1")
            ):
                rect_like = cast(Any, raw_bbox)
                return PdfDocumentParser._normalize_bbox(
                    float(rect_like.x0),
                    float(rect_like.y0),
                    float(rect_like.x1),
                    float(rect_like.y1),
                )
            if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
                return None
            return PdfDocumentParser._normalize_bbox(
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            )
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_bbox(x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
        """归一化 bbox，避免出现反向坐标。"""
        return (
            min(x0, x1),
            min(y0, y1),
            max(x0, x1),
            max(y0, y1),
        )

    @staticmethod
    def _safe_int(value: object) -> int | None:
        """安全读取整数值。"""
        if not isinstance(value, (str, bytes, bytearray, int, float)):
            return None
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
