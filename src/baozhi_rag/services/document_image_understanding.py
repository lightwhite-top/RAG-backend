"""文档图片规范化与多模态识别服务。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps

from baozhi_rag.services.llm import ImageRecognitionModelClient


@dataclass(frozen=True, slots=True)
class DocumentImageUnderstandingResult:
    """单张图片的规范化与识别结果。"""

    image_sha256: str
    normalized_image_sha256: str
    normalized_image_bytes: bytes
    width: int | None
    height: int | None
    ocr_text: str
    normalized_ocr_text: str
    summary: str
    image_type: str
    content_type: str
    recognition_model: str


class DocumentImageUnderstandingService:
    """负责文档图片的规范化和多模态识别。"""

    _NORMALIZED_IMAGE_FORMAT = "PNG"
    _NORMALIZED_CONTENT_TYPE = "image/png"

    def __init__(
        self,
        *,
        client: ImageRecognitionModelClient,
        model_name: str,
    ) -> None:
        self._client = client
        self._model_name = model_name.strip()

    def analyze_image(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
    ) -> DocumentImageUnderstandingResult:
        """对单张图片完成规范化、哈希计算与语义识别。"""
        normalized_image_bytes, width, height = self._normalize_image(image_bytes)
        recognition_result = self._client.recognize_image(
            image_bytes=normalized_image_bytes,
            content_type=self._NORMALIZED_CONTENT_TYPE,
            model_name=self._model_name,
        )
        ocr_text = self._normalize_ocr_text(recognition_result.get("ocr_text", ""))
        return DocumentImageUnderstandingResult(
            image_sha256=hashlib.sha256(image_bytes).hexdigest(),
            normalized_image_sha256=hashlib.sha256(normalized_image_bytes).hexdigest(),
            normalized_image_bytes=normalized_image_bytes,
            width=width,
            height=height,
            ocr_text=ocr_text,
            normalized_ocr_text=ocr_text,
            summary=recognition_result.get("summary", "").strip(),
            image_type=recognition_result.get("image_type", "unknown").strip() or "unknown",
            content_type=content_type.strip() or self._NORMALIZED_CONTENT_TYPE,
            recognition_model=self._model_name,
        )

    def ensure_ready(self) -> None:
        """校验图片识别客户端是否可用。"""
        if not self._model_name:
            msg = "未配置图片识别模型名称"
            raise ValueError(msg)
        self._client.ensure_ready()

    def _normalize_image(self, image_bytes: bytes) -> tuple[bytes, int | None, int | None]:
        """将图片规范化为稳定的 PNG 字节流。"""
        with Image.open(BytesIO(image_bytes)) as image:
            normalized_image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = normalized_image.size
            output_buffer = BytesIO()
            normalized_image.save(output_buffer, format=self._NORMALIZED_IMAGE_FORMAT)
            return output_buffer.getvalue(), width, height

    def _normalize_ocr_text(self, raw_text: str) -> str:
        """把 OCR 文本规整为稳定的单行文本，降低空白差异对哈希的影响。"""
        return " / ".join(
            normalized_line
            for normalized_line in (" ".join(part.split()) for part in raw_text.splitlines())
            if normalized_line
        )
