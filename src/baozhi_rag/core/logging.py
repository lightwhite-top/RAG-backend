"""日志初始化。"""

from __future__ import annotations

import logging
import sys

from baozhi_rag.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """在应用启动前初始化根日志配置。

    参数:
        settings: 当前应用配置，主要使用其中的日志级别设置根日志行为。

    返回:
        None。函数通过 `logging.basicConfig` 修改全局日志配置。
    """
    # Uvicorn 在应用生命周期前通常已经初始化过日志；这里需要强制覆盖根日志，
    # 否则业务代码里的 LOGGER 很容易因为 basicConfig 不生效而无法稳定打印到控制台。
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
