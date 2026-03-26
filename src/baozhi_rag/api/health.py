"""健康检查路由。"""

from typing import Annotated

from fastapi import APIRouter, Depends

from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.schemas.system import HealthResponse

router = APIRouter()


@router.get("/health/live", response_model=HealthResponse, summary="存活检查")
def live(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """返回服务当前的基础运行状态。

    参数:
        settings: 当前应用配置，用于填充服务元信息。

    返回:
        表示服务可用状态的健康检查响应。
    """
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_env,
        version=settings.version,
    )
