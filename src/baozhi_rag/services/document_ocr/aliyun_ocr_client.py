"""阿里云 OCR 客户端封装。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Protocol

from baozhi_rag.core.exceptions import AppError

try:  # pragma: no cover - 依赖是否可用由运行环境决定
    from alibabacloud_ocr_api20210707 import models as ocr_models  # type: ignore[import-untyped]
    from alibabacloud_ocr_api20210707.client import (  # type: ignore[import-untyped]
        Client as AliyunOcrSdkClient,
    )
    from alibabacloud_tea_openapi import models as openapi_models  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - 在未安装依赖的环境下允许模块被导入
    ocr_models = None
    AliyunOcrSdkClient = None
    openapi_models = None


BoundingBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class OcrRecognizedBlock:
    """OCR 识别出的结构化块。"""

    block_type: str
    text: str
    bbox: BoundingBox | None = None


@dataclass(frozen=True, slots=True)
class OcrPageStructureResult:
    """整页结构化 OCR 结果。"""

    blocks: list[OcrRecognizedBlock] = field(default_factory=list)
    raw_payload: dict[str, object] = field(default_factory=dict)


class AliyunOcrConfigurationError(AppError):
    """阿里云 OCR 配置错误。"""

    default_message = "阿里云 OCR 配置不完整"
    default_error_code = "aliyun_ocr_configuration_error"
    default_status_code = 500


class AliyunOcrInvocationError(AppError):
    """阿里云 OCR 调用失败。"""

    default_message = "调用阿里云 OCR 失败"
    default_error_code = "aliyun_ocr_invocation_error"
    default_status_code = 502


class AliyunOcrClientProtocol(Protocol):
    """供 PDF 解析器依赖的 OCR 客户端协议。"""

    def recognize_page_structure(self, *, image_bytes: bytes) -> OcrPageStructureResult:
        """识别整页结构化结果。"""

    def recognize_text(self, *, image_bytes: bytes, advanced: bool = False) -> str:
        """识别普通文本。"""

    def recognize_table_text(self, *, image_bytes: bytes) -> str:
        """识别表格区域文本。"""

    def recognize_handwriting_text(self, *, image_bytes: bytes) -> str:
        """识别手写区域文本。"""


@dataclass(slots=True)
class AliyunOcrClient:
    """阿里云 OCR 客户端。"""

    endpoint: str
    access_key_id: str
    access_key_secret: str
    region: str
    page_structure_api: str = "RecognizeDocumentStructure"
    text_api: str = "RecognizeGeneral"
    text_api_fallback: str = "RecognizeAdvanced"
    table_api: str = "RecognizeTableOcr"
    handwriting_api: str = "RecognizeHandwriting"
    connect_timeout_ms: int = 10_000
    read_timeout_ms: int = 60_000
    _client: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: Any) -> AliyunOcrClient:
        """基于全局配置创建阿里云 OCR 客户端。"""
        return cls(
            endpoint=str(getattr(settings, "aliyun_ocr_endpoint", "")).strip(),
            access_key_id=str(getattr(settings, "oss_access_key_id", "")).strip(),
            access_key_secret=str(getattr(settings, "oss_access_key_secret", "")).strip(),
            region=str(getattr(settings, "oss_region", "")).strip(),
            page_structure_api=str(
                getattr(settings, "aliyun_ocr_page_structure_api", "RecognizeDocumentStructure")
            ).strip()
            or "RecognizeDocumentStructure",
            text_api=str(getattr(settings, "aliyun_ocr_text_api", "RecognizeGeneral")).strip()
            or "RecognizeGeneral",
            text_api_fallback=str(
                getattr(settings, "aliyun_ocr_text_api_fallback", "RecognizeAdvanced")
            ).strip()
            or "RecognizeAdvanced",
            table_api=str(getattr(settings, "aliyun_ocr_table_api", "RecognizeTableOcr")).strip()
            or "RecognizeTableOcr",
            handwriting_api=str(
                getattr(settings, "aliyun_ocr_handwriting_api", "RecognizeHandwriting")
            ).strip()
            or "RecognizeHandwriting",
        )

    def ensure_ready(self) -> None:
        """校验客户端运行前置条件。"""
        if AliyunOcrSdkClient is None or ocr_models is None or openapi_models is None:
            msg = "未安装阿里云 OCR SDK，无法启用 PDF OCR 能力"
            raise AliyunOcrConfigurationError(msg)
        if (
            not self.endpoint
            or not self.access_key_id
            or not self.access_key_secret
            or not self.region
        ):
            raise AliyunOcrConfigurationError()

    def recognize_page_structure(self, *, image_bytes: bytes) -> OcrPageStructureResult:
        """识别整页结构化结果。"""
        request = ocr_models.RecognizeDocumentStructureRequest(
            need_rotate=True,
            need_sort_page=True,
            output_table=True,
            page=True,
            paragraph=True,
            row=True,
            use_new_style_output=True,
            body=BytesIO(image_bytes),
        )
        payload = self._invoke_operation("recognize_document_structure", request)
        return self._parse_page_structure_payload(payload)

    def recognize_text(self, *, image_bytes: bytes, advanced: bool = False) -> str:
        """识别普通文本。"""
        if advanced:
            request = ocr_models.RecognizeAdvancedRequest(
                need_rotate=True,
                need_sort_page=True,
                output_table=False,
                paragraph=True,
                row=True,
                body=BytesIO(image_bytes),
            )
            payload = self._invoke_operation("recognize_advanced", request)
        else:
            request = ocr_models.RecognizeGeneralRequest(body=BytesIO(image_bytes))
            payload = self._invoke_operation("recognize_general", request)
        return self._extract_plain_text(payload)

    def recognize_table_text(self, *, image_bytes: bytes) -> str:
        """识别表格区域文本。"""
        request = ocr_models.RecognizeTableOcrRequest(
            need_rotate=True,
            skip_detection=False,
            body=BytesIO(image_bytes),
        )
        payload = self._invoke_operation("recognize_table_ocr", request)
        return self._extract_plain_text(payload)

    def recognize_handwriting_text(self, *, image_bytes: bytes) -> str:
        """识别手写区域文本。"""
        request = ocr_models.RecognizeHandwritingRequest(
            need_rotate=True,
            need_sort_page=True,
            paragraph=True,
            body=BytesIO(image_bytes),
        )
        payload = self._invoke_operation("recognize_handwriting", request)
        return self._extract_plain_text(payload)

    def _get_client(self) -> Any:
        """懒加载底层 SDK 客户端。"""
        if self._client is None:
            self.ensure_ready()
            config = openapi_models.Config(
                access_key_id=self.access_key_id,
                access_key_secret=self.access_key_secret,
                endpoint=self.endpoint,
                region_id=self.region,
                connect_timeout=self.connect_timeout_ms,
                read_timeout=self.read_timeout_ms,
            )
            self._client = AliyunOcrSdkClient(config)
        return self._client

    def _invoke_operation(self, operation_name: str, request: Any) -> dict[str, object]:
        """统一调用 OCR OpenAPI，并返回结构化 payload。"""
        client = self._get_client()
        operation = getattr(client, operation_name, None)
        if operation is None:
            msg = f"阿里云 OCR SDK 不支持操作: {operation_name}"
            raise AliyunOcrInvocationError(msg)
        try:
            response = operation(request)
        except Exception as exc:  # pragma: no cover - 第三方 SDK 异常类型不稳定
            raise AliyunOcrInvocationError() from exc

        body = getattr(response, "body", None)
        if body is None:
            raise AliyunOcrInvocationError("阿里云 OCR 返回空响应")

        code = getattr(body, "code", None)
        message = getattr(body, "message", None)
        if isinstance(code, str) and code and code not in {"200", "Success", "OK"}:
            raise AliyunOcrInvocationError(str(message).strip() or "阿里云 OCR 返回异常状态")

        raw_data = getattr(body, "data", None)
        if raw_data is None:
            return {}
        if isinstance(raw_data, str):
            try:
                parsed = json.loads(raw_data)
            except json.JSONDecodeError:
                return {"text": raw_data}
            if isinstance(parsed, dict):
                return parsed
            return {"items": parsed}
        if isinstance(raw_data, dict):
            return raw_data
        return {"raw": raw_data}

    def _parse_page_structure_payload(self, payload: dict[str, object]) -> OcrPageStructureResult:
        """从结构化接口结果中提取可用块。"""
        blocks: list[OcrRecognizedBlock] = []
        for item in self._read_structure_items(payload):
            block_type = self._normalize_block_type(item)
            raw_text = (
                self._read_optional_str(item, "text")
                or self._read_optional_str(item, "paragraphContent")
                or self._read_optional_str(item, "word")
                or self._read_optional_str(item, "content")
                or ""
            )
            text = self._normalize_text(raw_text)
            bbox = self._read_bbox(item)
            if not text and block_type != "table":
                continue
            blocks.append(OcrRecognizedBlock(block_type=block_type, text=text, bbox=bbox))
        return OcrPageStructureResult(blocks=blocks, raw_payload=payload)

    def _read_structure_items(self, payload: dict[str, object]) -> list[dict[str, object]]:
        """读取页结构结果中的块列表。"""
        candidate_keys = [
            "paragraphs",
            "paragraphInfos",
            "paragraph_info",
            "prism_paragraphsInfo",
            "results",
            "items",
        ]
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        nested_payload = payload.get("Data")
        if isinstance(nested_payload, dict):
            return self._read_structure_items(nested_payload)
        return []

    def _normalize_block_type(self, item: dict[str, object]) -> str:
        """归一化阿里云结构块类型。"""
        raw_type = (
            self._read_optional_str(item, "type")
            or self._read_optional_str(item, "paragraphType")
            or self._read_optional_str(item, "category")
            or "paragraph"
        ).lower()
        if "table" in raw_type:
            return "table"
        if "title" in raw_type or "heading" in raw_type:
            return "title"
        if "figure" in raw_type or "image" in raw_type:
            return "figure"
        return "paragraph"

    def _extract_plain_text(self, payload: dict[str, object]) -> str:
        """从 OCR payload 中提取稳定纯文本。"""
        direct_text = self._read_optional_str(payload, "text") or self._read_optional_str(
            payload, "content"
        )
        if direct_text:
            return self._normalize_text(direct_text)

        candidate_keys = [
            "prism_wordsInfo",
            "wordsInfo",
            "paragraphs",
            "paragraphInfos",
            "prism_paragraphsInfo",
            "items",
        ]
        lines: list[str] = []
        for key in candidate_keys:
            value = payload.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                text = (
                    self._read_optional_str(item, "word")
                    or self._read_optional_str(item, "text")
                    or self._read_optional_str(item, "content")
                    or self._read_optional_str(item, "paragraphContent")
                )
                normalized_text = self._normalize_text(text or "")
                if normalized_text:
                    lines.append(normalized_text)
            if lines:
                break

        return "\n".join(dict.fromkeys(lines))

    def _read_bbox(self, item: dict[str, object]) -> BoundingBox | None:
        """从 OCR 结果中读取矩形边界。"""
        for key in ("bbox", "position", "pos", "tableLocation"):
            value = item.get(key)
            if isinstance(value, list) and len(value) >= 4:
                try:
                    coordinates = tuple(float(v) for v in value[:4])
                except (TypeError, ValueError):
                    continue
                return coordinates  # type: ignore[return-value]
            if isinstance(value, dict):
                x0 = value.get("left") or value.get("x")
                y0 = value.get("top") or value.get("y")
                x1 = value.get("right")
                y1 = value.get("bottom")
                if None in {x0, y0, x1, y1}:
                    continue
                normalized_bbox = (
                    self._to_float(x0),
                    self._to_float(y0),
                    self._to_float(x1),
                    self._to_float(y1),
                )
                if any(value is None for value in normalized_bbox):
                    continue
                return normalized_bbox  # type: ignore[return-value]
        return None

    @staticmethod
    def _to_float(value: object) -> float | None:
        """安全转换浮点数。"""
        if not isinstance(value, (str, bytes, bytearray, int, float)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _read_optional_str(item: dict[str, object], key: str) -> str | None:
        """从字典中安全读取可选字符串。"""
        value = item.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalize_text(text: str) -> str:
        """规整 OCR 文本。"""
        return "\n".join(
            normalized_line
            for normalized_line in (" ".join(line.split()) for line in text.splitlines())
            if normalized_line
        )
