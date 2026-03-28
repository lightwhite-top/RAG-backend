"""应用统一异常定义。"""

from __future__ import annotations

from typing import ClassVar

from fastapi import status


class AppError(Exception):
    """应用统一异常基类。

    参数:
        message: 可直接暴露给接口调用方的错误提示；未传时使用类默认提示。
        error_code: 可选的稳定错误码；未传时使用类默认错误码。
        status_code: 可选的 HTTP 状态码；未传时使用类默认状态码。

    返回:
        None。
    """

    default_message: ClassVar[str] = "服务内部错误"
    default_error_code: ClassVar[str] = "internal_server_error"
    default_status_code: ClassVar[int] = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        """初始化应用异常对象。"""
        self.message = message or self.default_message
        self.error_code = error_code or self.default_error_code
        self.status_code = status_code or self.default_status_code
        super().__init__(self.message)
