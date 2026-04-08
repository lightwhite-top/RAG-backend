"""FastAPI 全局异常处理注册。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from http import HTTPStatus

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.core.request_context import REQUEST_ID_HEADER_NAME, ensure_request_id
from baozhi_rag.schemas.common import ErrorResponse, ValidationErrorItem

LOGGER = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """为 FastAPI 应用注册统一异常处理器。

    参数:
        app: 待注册全局异常处理器的 FastAPI 应用实例。

    返回:
        None。
    """
    app.exception_handler(AppError)(handle_app_error)
    app.exception_handler(RequestValidationError)(handle_request_validation_error)
    app.exception_handler(StarletteHTTPException)(handle_http_exception)
    app.exception_handler(Exception)(handle_unexpected_exception)


async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """将领域异常转换为统一错误响应。"""
    request_id = ensure_request_id(request)
    _log_exception(
        request=request,
        request_id=request_id,
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        exc=exc if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR else None,
    )
    return _build_error_response(
        request_id=request_id,
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
    )


async def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """统一处理请求参数校验异常。"""
    request_id = ensure_request_id(request)
    details = [
        ValidationErrorItem(
            field=".".join(str(item) for item in error.get("loc", ())),
            message=_translate_validation_error(error),
        )
        for error in exc.errors()
    ]
    _log_exception(
        request=request,
        request_id=request_id,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        error_code="request_validation_error",
        message="请求参数校验失败",
    )
    return _build_error_response(
        request_id=request_id,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        error_code="request_validation_error",
        message="请求参数校验失败",
        details=details,
    )


async def handle_http_exception(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """统一处理 HTTP 协议异常。"""
    request_id = ensure_request_id(request)
    message = _resolve_http_exception_message(exc)
    error_code = _resolve_http_exception_code(exc.status_code)
    _log_exception(
        request=request,
        request_id=request_id,
        status_code=exc.status_code,
        error_code=error_code,
        message=message,
    )
    return _build_error_response(
        request_id=request_id,
        status_code=exc.status_code,
        error_code=error_code,
        message=message,
    )


async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    """兜底处理未显式声明的系统异常。"""
    request_id = ensure_request_id(request)
    _log_exception(
        request=request,
        request_id=request_id,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="internal_server_error",
        message="服务内部错误",
        exc=exc,
    )
    return _build_error_response(
        request_id=request_id,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="internal_server_error",
        message="服务内部错误",
    )


def _build_error_response(
    *,
    request_id: str,
    status_code: int,
    error_code: str,
    message: str,
    details: list[ValidationErrorItem] | None = None,
) -> JSONResponse:
    """构造统一错误响应。"""
    payload = ErrorResponse(
        code=error_code,
        message=message,
        request_id=request_id,
        details=details,
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(exclude_none=True),
        headers={REQUEST_ID_HEADER_NAME: request_id},
    )


def _log_exception(
    *,
    request: Request,
    request_id: str,
    status_code: int,
    error_code: str,
    message: str,
    exc: Exception | None = None,
) -> None:
    """按统一格式记录异常日志。"""
    detailed_message = _build_log_message(message=message, exc=exc)

    if exc is None:
        LOGGER.warning(
            "request_failed request_id=%s method=%s path=%s status_code=%s error_code=%s message=%s",
            request_id,
            request.method,
            request.url.path,
            status_code,
            error_code,
            detailed_message,
        )
        return

    LOGGER.error(
        "request_failed request_id=%s method=%s path=%s status_code=%s error_code=%s message=%s",
        request_id,
        request.method,
        request.url.path,
        status_code,
        error_code,
        detailed_message,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def _build_log_message(*, message: str, exc: Exception | None) -> str:
    """为日志消息附加根因摘要，避免只剩泛化错误提示。"""
    if exc is None:
        return message

    cause_summary = _summarize_root_cause(exc)
    if not cause_summary:
        return message
    return f"{message}; {cause_summary}"


def _summarize_root_cause(exc: Exception) -> str:
    """提取异常链最深层根因中的关键字段。"""
    root_cause = _resolve_root_cause(exc)
    if root_cause is None:
        return ""

    parts = [f"cause_type={type(root_cause).__name__}"]

    status_code = getattr(root_cause, "status_code", None)
    if status_code is not None:
        parts.append(f"cause_status_code={status_code}")

    error_code = getattr(root_cause, "code", None)
    if error_code:
        parts.append(f"cause_code={error_code}")

    request_id = getattr(root_cause, "request_id", None)
    if request_id:
        parts.append(f"cause_request_id={request_id}")

    raw_message = str(root_cause).strip()
    if raw_message:
        parts.append(f"cause_message={_truncate_log_text(raw_message)}")

    body = getattr(root_cause, "body", None)
    if body is not None:
        parts.append(f"cause_body={_truncate_log_text(_serialize_log_value(body))}")

    return " ".join(parts)


def _resolve_root_cause(exc: Exception) -> Exception | None:
    """沿异常链找到最深层根因。"""
    current = exc
    last_cause: Exception | None = None

    for _ in range(8):
        next_cause = current.__cause__ or current.__context__
        if not isinstance(next_cause, Exception):
            break
        last_cause = next_cause
        current = next_cause

    return last_cause


def _serialize_log_value(value: object) -> str:
    """把异常 body 等复杂字段转换为可检索文本。"""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _truncate_log_text(value: str, *, max_length: int = 500) -> str:
    """限制日志字段长度，避免错误体过长污染日志。"""
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[:max_length]}..."


def _translate_validation_error(error: dict[str, object]) -> str:
    """把常见校验错误翻译为更易读的中文提示。"""
    error_type = str(error.get("type", ""))
    context_dict = _normalize_context_mapping(error.get("ctx"))

    if error_type == "missing":
        return "字段不能为空"
    if error_type == "string_too_short":
        min_length = context_dict.get("min_length")
        return f"长度不能少于 {min_length}" if min_length is not None else "字符串长度过短"
    if error_type == "string_too_long":
        max_length = context_dict.get("max_length")
        return f"长度不能超过 {max_length}" if max_length is not None else "字符串长度过长"
    if error_type == "greater_than":
        gt_value = context_dict.get("gt")
        return f"值必须大于 {gt_value}" if gt_value is not None else "值不满足下限要求"
    if error_type == "greater_than_equal":
        ge_value = context_dict.get("ge")
        return f"值必须大于等于 {ge_value}" if ge_value is not None else "值不满足下限要求"
    if error_type == "less_than":
        lt_value = context_dict.get("lt")
        return f"值必须小于 {lt_value}" if lt_value is not None else "值超过允许上限"
    if error_type == "less_than_equal":
        le_value = context_dict.get("le")
        return f"值必须小于等于 {le_value}" if le_value is not None else "值超过允许上限"
    if error_type in {"int_parsing", "int_type"}:
        return "必须是整数"
    if error_type in {"float_parsing", "float_type"}:
        return "必须是数字"
    if error_type in {"bool_parsing", "bool_type"}:
        return "必须是布尔值"
    if error_type == "string_type":
        return "必须是字符串"
    if error_type == "list_type":
        return "必须是数组"
    if error_type == "dict_type":
        return "必须是对象"
    if error_type == "literal_error":
        return "值不在允许范围内"
    if error_type == "extra_forbidden":
        return "不允许传入该字段"

    translated_message = _translate_raw_validation_message(error.get("msg"))
    return translated_message or "请求参数不合法"


def _translate_raw_validation_message(raw_message: object) -> str | None:
    """兜底翻译 Pydantic 未显式映射的原始校验消息。"""
    if not isinstance(raw_message, str):
        return None

    message = raw_message.strip()
    if not message:
        return None

    if message.startswith("Value error, "):
        normalized_message = message.removeprefix("Value error, ").strip()
        return normalized_message or "请求参数不合法"

    direct_message_map = {
        "Field required": "字段不能为空",
        "Input should be a valid string": "必须是字符串",
        "Input should be a valid integer": "必须是整数",
        "Input should be a valid number": "必须是数字",
        "Input should be a valid boolean": "必须是布尔值",
        "Input should be a valid list": "必须是数组",
        "Input should be a valid dictionary": "必须是对象",
        "Extra inputs are not permitted": "不允许传入额外字段",
    }
    if message in direct_message_map:
        return direct_message_map[message]

    if "valid email address" in message or "@-sign" in message:
        return "邮箱格式不正确"

    if matched := re.search(r"at least (\d+) characters", message):
        return f"长度不能少于 {matched.group(1)}"
    if matched := re.search(r"at most (\d+) characters", message):
        return f"长度不能超过 {matched.group(1)}"
    if matched := re.search(r"greater than or equal to ([^ ]+)", message):
        return f"值必须大于等于 {matched.group(1)}"
    if matched := re.search(r"greater than ([^ ]+)", message):
        return f"值必须大于 {matched.group(1)}"
    if matched := re.search(r"less than or equal to ([^ ]+)", message):
        return f"值必须小于等于 {matched.group(1)}"
    if matched := re.search(r"less than ([^ ]+)", message):
        return f"值必须小于 {matched.group(1)}"

    return message if re.search(r"[\u4e00-\u9fff]", message) else None


def _normalize_context_mapping(value: object) -> Mapping[str, object]:
    """把校验上下文归一化为键为字符串的只读映射。"""
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, object] = {}
    for key, item in value.items():
        normalized[str(key)] = item
    return normalized


def _resolve_http_exception_message(exc: StarletteHTTPException) -> str:
    """解析 HTTP 异常的返回提示。"""
    if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        return "服务内部错误"

    detail = exc.detail.strip()
    if detail:
        return _translate_http_detail_message(detail) or detail

    return _resolve_http_status_message(exc.status_code)


def _translate_http_detail_message(detail: str) -> str | None:
    """翻译常见的框架默认 HTTP 提示。"""
    detail_message_map = {
        "Bad Request": "请求不合法",
        "Unauthorized": "未认证或登录已失效",
        "Forbidden": "无权访问",
        "Not Found": "请求的资源不存在",
        "Method Not Allowed": "请求方法不被允许",
        "Unprocessable Entity": "请求参数校验失败",
        "Too Many Requests": "请求过于频繁，请稍后再试",
    }
    return detail_message_map.get(detail)


def _resolve_http_status_message(status_code: int) -> str:
    """基于状态码生成默认中文提示。"""
    default_message_map = {
        status.HTTP_400_BAD_REQUEST: "请求不合法",
        status.HTTP_401_UNAUTHORIZED: "未认证或登录已失效",
        status.HTTP_403_FORBIDDEN: "无权访问",
        status.HTTP_404_NOT_FOUND: "请求的资源不存在",
        status.HTTP_405_METHOD_NOT_ALLOWED: "请求方法不被允许",
        status.HTTP_422_UNPROCESSABLE_CONTENT: "请求参数校验失败",
        status.HTTP_429_TOO_MANY_REQUESTS: "请求过于频繁，请稍后再试",
    }
    if status_code in default_message_map:
        return default_message_map[status_code]

    try:
        status_phrase = HTTPStatus(status_code).phrase
    except ValueError:
        return "请求处理失败"
    return _translate_http_detail_message(status_phrase) or "请求处理失败"


def _resolve_http_exception_code(status_code: int) -> str:
    """基于 HTTP 状态码生成稳定错误码。"""
    if status_code == status.HTTP_404_NOT_FOUND:
        return "not_found"
    if status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
        return "method_not_allowed"
    if status_code == status.HTTP_401_UNAUTHORIZED:
        return "unauthorized"
    if status_code == status.HTTP_403_FORBIDDEN:
        return "forbidden"
    if status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        return "internal_server_error"
    return "http_error"
