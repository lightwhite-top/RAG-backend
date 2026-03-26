"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from baozhi_rag.api.routes import router
from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.core.logging import configure_logging
from baozhi_rag.schemas.system import ServiceInfoResponse

LOGGER = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""
    current_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """管理应用生命周期事件。"""
        configure_logging(current_settings)
        LOGGER.info(
            "service_startup env=%s version=%s",
            current_settings.app_env,
            current_settings.version,
        )
        yield
        LOGGER.info("service_shutdown")

    app = FastAPI(
        title=current_settings.app_name,
        debug=current_settings.debug,
        version=current_settings.version,
        lifespan=lifespan,
    )
    app.dependency_overrides[get_settings] = lambda: current_settings
    app.include_router(router)

    @app.get("/", response_model=ServiceInfoResponse, summary="服务信息")
    def root() -> ServiceInfoResponse:
        """返回服务基础信息，便于环境探活与调试。"""
        return ServiceInfoResponse(
            service=current_settings.app_name,
            environment=current_settings.app_env,
            version=current_settings.version,
            docs_url=app.docs_url or "",
        )

    return app


app = create_app()
