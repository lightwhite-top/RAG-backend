"""远程 HTTP OCR 客户端封装。"""

from __future__ import annotations

import json
import mimetypes
import socket
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.document_ocr.ocr_capacity_limiter import OcrCapacityPendingError
from baozhi_rag.services.document_ocr.ocr_client_protocol import (
    BoundingBox,
    OcrClientProtocol,
    OcrPageStructureResult,
    OcrRecognizedBlock,
)


class RemoteHttpOcrConfigurationError(AppError):
    """远程 HTTP OCR 配置错误。"""

    default_message = "本地 HTTP OCR 服务配置不完整"
    default_error_code = "remote_http_ocr_configuration_error"
    default_status_code = 500


class RemoteHttpOcrInvocationError(AppError):
    """远程 HTTP OCR 调用失败。"""

    default_message = "调用本地 HTTP OCR 服务失败"
    default_error_code = "remote_http_ocr_invocation_error"
    default_status_code = 502


@dataclass(slots=True)
class RemoteHttpOcrClient(OcrClientProtocol):
    """通过 HTTP 调用远程 OCR 服务。

    参数:
        base_url: OCR 服务基础地址，例如 `http://127.0.0.1:18080`。
        timeout_seconds: 单次 OCR 请求超时时间，单位为秒。
    """

    base_url: str
    timeout_seconds: float = 30.0

    @classmethod
    def from_settings(cls, settings: Any) -> RemoteHttpOcrClient:
        """基于全局配置创建远程 HTTP OCR 客户端。"""
        return cls(
            base_url=str(getattr(settings, "ocr_service_base_url", "")).strip(),
            timeout_seconds=float(getattr(settings, "ocr_service_timeout_seconds", 30.0)),
        )

    def ensure_ready(self) -> None:
        """通过健康检查确认 OCR 服务可用。"""
        self._validate_configuration()
        self._request_text("healthz", image_bytes=None)

    def recognize_page_structure(self, *, image_bytes: bytes) -> OcrPageStructureResult:
        """识别整页结构化结果。"""
        payload = self._request_json("ocr/page-structure", image_bytes=image_bytes)
        raw_blocks = payload.get("blocks")
        if not isinstance(raw_blocks, list):
            raise RemoteHttpOcrInvocationError("OCR 页结构结果缺少 blocks 列表")

        blocks: list[OcrRecognizedBlock] = []
        for index, item in enumerate(raw_blocks, start=1):
            if not isinstance(item, dict):
                raise RemoteHttpOcrInvocationError(
                    f"OCR 页结构结果第 {index} 个 blocks 元素格式异常"
                )
            block_type = self._normalize_block_type(item)
            text = self._normalize_text(self._extract_block_text(item))
            bbox = self._read_bbox(item)
            if not text and block_type != "table":
                continue
            blocks.append(OcrRecognizedBlock(block_type=block_type, text=text, bbox=bbox))
        return OcrPageStructureResult(blocks=blocks, raw_payload=payload)

    def recognize_text(self, *, image_bytes: bytes, advanced: bool = False) -> str:
        """识别普通文本。

        参数:
            image_bytes: 需要识别的图片字节流。
            advanced: 为兼容现有调用方保留的参数；远程服务当前统一走同一个文本接口。
        """

        del advanced
        payload = self._request_json("ocr/text", image_bytes=image_bytes)
        return self._extract_text_from_payload(payload=payload, line_key="lines")

    def recognize_table_text(self, *, image_bytes: bytes) -> str:
        """识别表格区域文本。"""
        payload = self._request_json("ocr/table-text", image_bytes=image_bytes)
        return self._extract_text_from_payload(payload=payload, line_key="rows")

    def _validate_configuration(self) -> None:
        """校验客户端配置是否合法。"""
        if not self.base_url:
            raise RemoteHttpOcrConfigurationError("未配置 OCR_SERVICE_BASE_URL")
        if self.timeout_seconds <= 0:
            raise RemoteHttpOcrConfigurationError("OCR_SERVICE_TIMEOUT_SECONDS 必须大于 0")

    def _request_json(self, path: str, *, image_bytes: bytes | None) -> dict[str, object]:
        """统一发起 OCR 请求并读取 JSON 响应。"""
        response_text = self._request_text(path, image_bytes=image_bytes)
        try:
            payload = json.loads(response_text)
        except JSONDecodeError as exc:
            raise RemoteHttpOcrInvocationError("OCR 服务返回的 JSON 格式无效") from exc

        if not isinstance(payload, dict):
            raise RemoteHttpOcrInvocationError("OCR 服务返回的 JSON 顶层结构必须为对象")
        return payload

    def _request_text(self, path: str, *, image_bytes: bytes | None) -> str:
        """统一发起 OCR 请求并读取文本响应。"""
        self._validate_configuration()
        request = self._build_request(path=path, image_bytes=image_bytes)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status_code = getattr(response, "status", HTTPStatus.OK)
                response_text = response.read().decode("utf-8")
        except HTTPError as exc:
            self._raise_http_error(exc)
        except URLError as exc:
            reason = exc.reason
            if isinstance(reason, socket.timeout):
                raise RemoteHttpOcrInvocationError("OCR 服务请求超时") from exc
            raise RemoteHttpOcrInvocationError("OCR 服务不可达") from exc
        except TimeoutError as exc:
            raise RemoteHttpOcrInvocationError("OCR 服务请求超时") from exc
        except Exception as exc:
            raise RemoteHttpOcrInvocationError() from exc

        if status_code != HTTPStatus.OK:
            raise RemoteHttpOcrInvocationError(f"OCR 服务返回异常状态码: {status_code}")
        return str(response_text)

    def _build_request(self, *, path: str, image_bytes: bytes | None) -> Request:
        """构造 GET/POST 请求对象。"""
        target_url = urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))
        if image_bytes is None:
            return Request(target_url, method="GET")

        body, headers = self._build_multipart_form_data(image_bytes=image_bytes)
        request = Request(target_url, data=body, method="POST")
        for key, value in headers.items():
            request.add_header(key, value)
        return request

    def _build_multipart_form_data(self, *, image_bytes: bytes) -> tuple[bytes, dict[str, str]]:
        """构造远程 OCR 服务所需的 `multipart/form-data` 请求体。"""
        boundary = f"----BaozhiRagOcrBoundary{uuid.uuid4().hex}"
        content_type = mimetypes.guess_type("page.png", strict=False)[0] or "image/png"
        body_parts = [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="file"; filename="page.png"\r\n',
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            image_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        body = b"".join(body_parts)
        return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}

    def _extract_text_from_payload(self, *, payload: dict[str, object], line_key: str) -> str:
        """从 OCR 文本响应中提取稳定纯文本。"""
        direct_text = payload.get("text")
        if isinstance(direct_text, str) and direct_text.strip():
            return self._normalize_text(direct_text)

        line_items = payload.get(line_key)
        if not isinstance(line_items, list):
            raise RemoteHttpOcrInvocationError(f"OCR 结果缺少 {line_key} 字段")

        normalized_lines: list[str] = []
        for item in line_items:
            normalized_line = self._normalize_text(self._stringify_line_item(item))
            if normalized_line:
                normalized_lines.append(normalized_line)

        if not normalized_lines:
            raise RemoteHttpOcrInvocationError("OCR 文本结果为空")
        return "\n".join(normalized_lines)

    def _extract_block_text(self, item: dict[str, object]) -> str:
        """从页结构块中提取可用文本。"""
        for key in ("text", "content", "label", "title"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value

        lines = item.get("lines")
        if isinstance(lines, list):
            return "\n".join(
                self._stringify_line_item(line)
                for line in lines
                if self._stringify_line_item(line).strip()
            )
        return ""

    def _normalize_block_type(self, item: dict[str, object]) -> str:
        """把远程服务返回的块类型映射为主链可消费的类型。"""
        raw_type = ""
        for key in ("block_type", "type", "category", "label"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                raw_type = value.strip().lower()
                break

        if "table" in raw_type:
            return "table"
        if any(keyword in raw_type for keyword in ("title", "heading", "header")):
            return "title"
        if any(keyword in raw_type for keyword in ("figure", "image", "picture")):
            return "figure"
        return "paragraph"

    def _read_bbox(self, item: dict[str, object]) -> BoundingBox | None:
        """从远程 OCR 响应中读取矩形边界。"""
        bbox_value = item.get("bbox") or item.get("box") or item.get("bounding_box")
        if isinstance(bbox_value, list) and len(bbox_value) >= 4:
            try:
                return (
                    float(bbox_value[0]),
                    float(bbox_value[1]),
                    float(bbox_value[2]),
                    float(bbox_value[3]),
                )
            except (TypeError, ValueError):
                raise RemoteHttpOcrInvocationError("OCR 返回的 bbox 坐标格式非法") from None
        if isinstance(bbox_value, dict):
            try:
                if {"x0", "y0", "x1", "y1"} <= set(bbox_value):
                    return (
                        float(bbox_value["x0"]),
                        float(bbox_value["y0"]),
                        float(bbox_value["x1"]),
                        float(bbox_value["y1"]),
                    )
                if {"left", "top", "right", "bottom"} <= set(bbox_value):
                    return (
                        float(bbox_value["left"]),
                        float(bbox_value["top"]),
                        float(bbox_value["right"]),
                        float(bbox_value["bottom"]),
                    )
            except KeyError:
                return None
            except (TypeError, ValueError):
                raise RemoteHttpOcrInvocationError("OCR 返回的 bbox 坐标格式非法") from None
        return None

    def _raise_http_error(self, exc: HTTPError) -> None:
        """把 HTTP 层错误统一转换为项目异常。"""
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        if exc.code == HTTPStatus.TOO_MANY_REQUESTS:
            raise OcrCapacityPendingError(
                "OCR 服务当前繁忙，请稍后重试",
                retry_after_seconds=1.0,
            ) from exc
        message = f"OCR 服务返回 HTTP {exc.code}"
        if body:
            message = f"{message}: {body[:200]}"
        raise RemoteHttpOcrInvocationError(message) from exc

    @staticmethod
    def _stringify_line_item(item: object) -> str:
        """把 `lines` 或 `rows` 中的任意元素规整为文本。"""
        if isinstance(item, str):
            return item
        if isinstance(item, list):
            return " ".join(str(cell).strip() for cell in item if str(cell).strip())
        if isinstance(item, dict):
            for key in ("text", "content", "line"):
                value = item.get(key)
                if isinstance(value, str):
                    return value
            cells = item.get("cells")
            if isinstance(cells, list):
                return " ".join(str(cell).strip() for cell in cells if str(cell).strip())
        return str(item).strip() if item is not None else ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        """规整 OCR 文本。"""
        return "\n".join(
            normalized_line
            for normalized_line in (" ".join(line.split()) for line in text.splitlines())
            if normalized_line
        )
