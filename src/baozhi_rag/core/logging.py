"""日志初始化。"""

from __future__ import annotations

import logging

from baozhi_rag.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """在应用启动前初始化根日志配置。"""
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
