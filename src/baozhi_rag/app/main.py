"""FastAPI 应用入口。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from baozhi_rag.api.dependencies import get_current_user
from baozhi_rag.api.routes import router
from baozhi_rag.app.exception_handlers import register_exception_handlers
from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.core.logging import configure_logging
from baozhi_rag.core.request_context import REQUEST_ID_HEADER_NAME, ensure_request_id
from baozhi_rag.domain.user import CurrentUser
from baozhi_rag.infra.database.knowledge_file_repository import SqlAlchemyKnowledgeFileRepository
from baozhi_rag.infra.database.knowledge_upload_task_repository import (
    SqlAlchemyKnowledgeUploadTaskRepository,
)
from baozhi_rag.infra.database.mysql import DatabaseManager
from baozhi_rag.infra.llm.aliyun_model_studio import AlibabaModelStudioClient
from baozhi_rag.infra.retrieval.hybrid_chunk_store import HybridChunkStore
from baozhi_rag.infra.storage.aliyun_oss_file_store import AliyunOssFileStore
from baozhi_rag.infra.storage.local_file_store import LocalFileStore
from baozhi_rag.schemas.common import SuccessResponse
from baozhi_rag.schemas.system import ServiceInfoResponse
from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.document_chunking import DocumentChunkService
from baozhi_rag.services.term_matching import build_default_term_matcher
from baozhi_rag.services.upload_tasks import KnowledgeUploadProcessor, KnowledgeUploadWorker

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
        database_manager = DatabaseManager.from_settings(current_settings)
        database_manager.ensure_ready()
        database_manager.ensure_schema()
        object_store = AliyunOssFileStore.from_settings(current_settings)
        object_store.ensure_ready()
        bailian_client = AlibabaModelStudioClient.from_settings(current_settings)
        bailian_client.ensure_ready()
        chunk_store = HybridChunkStore.from_settings(current_settings)
        chunk_store.ensure_ready()

        worker_tasks: list[asyncio.Task[None]] = []
        worker_instance_id = uuid4().hex[:8]
        for worker_index in range(current_settings.upload_worker_concurrency):
            processor = KnowledgeUploadProcessor(
                temp_file_store=LocalFileStore(current_settings.upload_root_dir),
                object_store=object_store,
                final_object_prefix=current_settings.normalized_oss_object_prefix,
                task_repository=SqlAlchemyKnowledgeUploadTaskRepository(
                    database_manager.session_factory
                ),
                knowledge_file_repository=SqlAlchemyKnowledgeFileRepository(
                    database_manager.session_factory
                ),
                chunk_service=DocumentChunkService(
                    chunk_size=current_settings.doc_chunk_size,
                    chunk_overlap=current_settings.doc_chunk_overlap,
                    convert_temp_dir=current_settings.doc_convert_temp_dir,
                    doc_convert_timeout_seconds=current_settings.doc_convert_timeout_seconds,
                    term_matcher=build_default_term_matcher(
                        current_settings.domain_dictionary_path
                    ),
                ),
                chunk_store=chunk_store,
                chunk_embedding_service=ChunkEmbeddingService(bailian_client),
                lease_seconds=current_settings.upload_task_lease_seconds,
                heartbeat_interval_seconds=current_settings.upload_task_heartbeat_interval_seconds,
            )
            worker = KnowledgeUploadWorker(
                processor=processor,
                worker_id=f"{worker_instance_id}-{worker_index}",
                poll_interval_seconds=current_settings.upload_worker_poll_interval_seconds,
            )
            worker_task = asyncio.create_task(worker.run())
            worker_task.set_name(f"knowledge-upload-worker-{worker_index}")
            worker_tasks.append(worker_task)

        LOGGER.info(
            "service_startup env=%s version=%s",
            current_settings.app_env,
            current_settings.version,
        )
        yield
        for worker_task in worker_tasks:
            worker_task.cancel()
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
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
    def root(
        request: Request,
        _: Annotated[CurrentUser, Depends(get_current_user)],
    ) -> SuccessResponse[ServiceInfoResponse]:
        """返回服务基础信息，便于环境探活与调试。

        参数:
            request: 当前 HTTP 请求对象，用于附加 request_id。
            _: 当前登录用户，仅用于在根路径上统一启用鉴权。

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
