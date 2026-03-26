"""聚合服务路由。"""

from fastapi import APIRouter

from baozhi_rag.api.files import router as files_router
from baozhi_rag.api.health import router as health_router

router = APIRouter()
router.include_router(health_router)
router.include_router(files_router)
