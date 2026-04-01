"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from baozhi_rag.api.routes import router
from baozhi_rag.app.exception_handlers import register_exception_handlers
from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.core.logging import configure_logging
from baozhi_rag.core.request_context import REQUEST_ID_HEADER_NAME, ensure_request_id
from baozhi_rag.infra.llm.aliyun_model_studio import AlibabaModelStudioClient
from baozhi_rag.infra.retrieval.hybrid_chunk_store import HybridChunkStore
from baozhi_rag.schemas.common import SuccessResponse
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
        AlibabaModelStudioClient.from_settings(current_settings).ensure_ready()
        HybridChunkStore.from_settings(current_settings).ensure_ready()
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
    register_exception_handlers(app)

    if current_settings.cors_allow_origins or current_settings.cors_allow_origin_regex:
        # 先注册 CORS，再注册 request_id 中间件，确保预检请求也能携带追踪头。
        app.add_middleware(
            CORSMiddleware,
            allow_origins=current_settings.cors_allow_origins,
            allow_origin_regex=current_settings.cors_allow_origin_regex,
            allow_credentials=current_settings.cors_allow_credentials,
            allow_methods=current_settings.cors_allow_methods,
            allow_headers=current_settings.cors_allow_headers,
            expose_headers=current_settings.cors_expose_headers,
        )

    @app.middleware("http")
    async def attach_request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """为每个请求补充 request_id，并回写到响应头。"""
        request_id = ensure_request_id(request)
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER_NAME] = request_id
        return response

    app.include_router(router)

    @app.get("/", response_model=SuccessResponse[ServiceInfoResponse], summary="服务信息")
    def root(request: Request) -> SuccessResponse[ServiceInfoResponse]:
        """返回服务基础信息，便于环境探活与调试。

        参数:
            request: 当前 HTTP 请求对象，用于附加 request_id。

        返回:
            包含服务名、运行环境、版本号和文档地址的服务信息响应。
        """
        return SuccessResponse[ServiceInfoResponse].success(
            message="获取服务信息成功",
            request_id=ensure_request_id(request),
            data=ServiceInfoResponse(
                service=current_settings.app_name,
                environment=current_settings.app_env,
                version=current_settings.version,
                docs_url=app.docs_url or "",
            ),
        )

    return app


app = create_app()
