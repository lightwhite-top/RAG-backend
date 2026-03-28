"""健康检查路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.core.request_context import ensure_request_id
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.schemas.system import HealthResponse

router = APIRouter()


@router.get(
    "/health/live",
    response_model=SuccessResponse[HealthResponse],
    summary="存活检查",
)
def live(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SuccessResponse[HealthResponse]:
    """返回服务当前的基础运行状态。

    参数:
        request: 当前 HTTP 请求对象，用于附加 request_id。
        settings: 当前应用配置，用于填充服务元信息。

    返回:
        表示服务可用状态的健康检查响应。
    """
    return SuccessResponse[HealthResponse].success(
        message="服务存活",
        request_id=ensure_request_id(request),
        data=HealthResponse(
            status="ok",
            service=settings.app_name,
            environment=settings.app_env,
            version=settings.version,
        ),
    )
