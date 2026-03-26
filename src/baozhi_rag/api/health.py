"""健康检查路由。"""

from typing import Annotated

from fastapi import APIRouter, Depends

from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.schemas.system import HealthResponse

router = APIRouter()


@router.get("/health/live", response_model=HealthResponse, summary="存活检查")
def live(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """返回服务当前的基础运行状态。"""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_env,
        version=settings.version,
    )
