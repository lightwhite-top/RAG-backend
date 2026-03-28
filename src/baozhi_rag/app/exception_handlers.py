"""FastAPI 全局异常处理注册。"""

from __future__ import annotations

import logging
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

    @app.exception_handler(AppError)
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

    @app.exception_handler(RequestValidationError)
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

    @app.exception_handler(StarletteHTTPException)
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

    @app.exception_handler(Exception)
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
    if exc is None:
        LOGGER.warning(
            "request_failed request_id=%s method=%s path=%s status_code=%s error_code=%s message=%s",
            request_id,
            request.method,
            request.url.path,
            status_code,
            error_code,
            message,
        )
        return

    LOGGER.error(
        "request_failed request_id=%s method=%s path=%s status_code=%s error_code=%s message=%s",
        request_id,
        request.method,
        request.url.path,
        status_code,
        error_code,
        message,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def _translate_validation_error(error: dict[str, object]) -> str:
    """把常见校验错误翻译为更易读的中文提示。"""
    error_type = str(error.get("type", ""))
    context = error.get("ctx")
    context_dict = context if isinstance(context, dict) else {}

    if error_type == "missing":
        return "字段不能为空"
    if error_type == "string_too_short":
        min_length = context_dict.get("min_length")
        return f"长度不能少于 {min_length}" if min_length is not None else "字符串长度过短"
    if error_type == "greater_than":
        gt_value = context_dict.get("gt")
        return f"值必须大于 {gt_value}" if gt_value is not None else "值不满足下限要求"
    if error_type == "less_than_equal":
        le_value = context_dict.get("le")
        return f"值必须小于等于 {le_value}" if le_value is not None else "值超过允许上限"
    if error_type == "int_parsing":
        return "必须是整数"
    if error_type == "string_type":
        return "必须是字符串"

    raw_message = error.get("msg")
    return str(raw_message) if raw_message else "请求参数不合法"


def _resolve_http_exception_message(exc: StarletteHTTPException) -> str:
    """解析 HTTP 异常的返回提示。"""
    if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        return "服务内部错误"

    detail = exc.detail
    if isinstance(detail, str) and detail.strip():
        return detail

    try:
        return HTTPStatus(exc.status_code).phrase
    except ValueError:
        return "请求处理失败"


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
