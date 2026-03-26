"""日志初始化。"""

from __future__ import annotations

import logging

from baozhi_rag.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """在应用启动前初始化根日志配置。

    参数:
        settings: 当前应用配置，主要使用其中的日志级别设置根日志行为。

    返回:
        None。函数通过 `logging.basicConfig` 修改全局日志配置。
    """
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
