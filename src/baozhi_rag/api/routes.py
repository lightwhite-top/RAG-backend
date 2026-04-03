"""聚合服务路由。"""

from fastapi import APIRouter, Depends

from baozhi_rag.api.admin_users import router as admin_users_router
from baozhi_rag.api.auth import router as auth_router
from baozhi_rag.api.chat import router as chat_router
from baozhi_rag.api.dependencies import get_current_user, require_admin
from baozhi_rag.api.files import router as files_router
from baozhi_rag.api.health import router as health_router
from baozhi_rag.api.search import router as search_router

router = APIRouter()
router.include_router(health_router)
router.include_router(auth_router)
router.include_router(
    files_router,
    dependencies=[Depends(get_current_user)],
)
router.include_router(
    search_router,
    dependencies=[Depends(get_current_user)],
)
router.include_router(
    chat_router,
    dependencies=[Depends(get_current_user)],
)
router.include_router(
    admin_users_router,
    dependencies=[Depends(require_admin)],
)
