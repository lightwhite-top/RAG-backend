"""文档 OCR 客户端协议与共享结果类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

BoundingBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class OcrRecognizedBlock:
    """OCR 识别出的结构化块。

    参数:
        block_type: 归一化后的块类型，例如 `title`、`paragraph`、`table`、`figure`。
        text: 当前块提取到的稳定文本；表格块允许为空字符串。
        bbox: 当前块在页图上的矩形区域，格式为 `(x0, y0, x1, y1)`。
    """

    block_type: str
    text: str
    bbox: BoundingBox | None = None


@dataclass(frozen=True, slots=True)
class OcrPageStructureResult:
    """整页结构化 OCR 结果。

    参数:
        blocks: 解析后可供 PDF 主链消费的结构化块列表。
        raw_payload: OCR 服务原始响应，便于后续排障时保留上下文。
    """

    blocks: list[OcrRecognizedBlock] = field(default_factory=list)
    raw_payload: dict[str, object] = field(default_factory=dict)


class OcrClientProtocol(Protocol):
    """供 PDF 解析器依赖的 OCR 客户端协议。"""

    def recognize_page_structure(self, *, image_bytes: bytes) -> OcrPageStructureResult:
        """识别整页结构化结果。"""

    def recognize_text(self, *, image_bytes: bytes, advanced: bool = False) -> str:
        """识别普通文本。"""

    def recognize_table_text(self, *, image_bytes: bytes) -> str:
        """识别表格区域文本。"""
