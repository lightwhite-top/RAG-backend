"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from baozhi_rag.api.routes import router
from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.core.logging import configure_logging
from baozhi_rag.infra.llm.aliyun_model_studio import AlibabaModelStudioClient
from baozhi_rag.infra.retrieval.elasticsearch_chunk_store import ElasticsearchChunkStore
from baozhi_rag.schemas.system import ServiceInfoResponse

LOGGER = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建并配置 FastAPI 应用实例。

    参数:
        settings: 可选的应用配置对象；未传入时从环境变量加载默认配置。

    返回:
        配置完成并挂载路由的 FastAPI 实例。
    """
    current_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """管理应用生命周期事件。

        参数:
            _: 当前 FastAPI 应用实例，本实现中不直接使用。

        返回:
            一个异步上下文管理器，在启动时初始化日志并在关闭时记录停机日志。
        """
        configure_logging(current_settings)
        if current_settings.chunk_embedding_enabled:
            AlibabaModelStudioClient.from_settings(current_settings).ensure_ready()
        ElasticsearchChunkStore.from_settings(current_settings).ensure_ready()
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
        """返回服务基础信息，便于环境探活与调试。

        返回:
            包含服务名、运行环境、版本号和文档地址的服务信息响应。
        """
        return ServiceInfoResponse(
            service=current_settings.app_name,
            environment=current_settings.app_env,
            version=current_settings.version,
            docs_url=app.docs_url or "",
        )

    return app


app = create_app()
