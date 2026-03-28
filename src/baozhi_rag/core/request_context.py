"""请求上下文工具。"""

from __future__ import annotations

from uuid import uuid4

from fastapi import Request

REQUEST_ID_HEADER_NAME = "X-Request-ID"


def ensure_request_id(request: Request) -> str:
    """确保当前请求持有稳定的 request_id。

    参数:
        request: 当前 FastAPI 请求对象。

    返回:
        请求关联的 request_id；若请求头未提供则自动生成。
    """
    request_id = getattr(request.state, "request_id", None)
    if isinstance(request_id, str) and request_id:
        return request_id

    header_request_id = request.headers.get(REQUEST_ID_HEADER_NAME, "").strip()
    resolved_request_id = header_request_id or uuid4().hex
    request.state.request_id = resolved_request_id
    return resolved_request_id
