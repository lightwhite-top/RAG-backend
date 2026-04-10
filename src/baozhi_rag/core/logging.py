"""日志初始化。"""

from __future__ import annotations

import logging
import sys

from baozhi_rag.core.config import Settings


class _UvicornAccessPathFilter(logging.Filter):
    """按请求路径过滤 Uvicorn 访问日志。"""

    _SUPPRESSED_PATH_PREFIXES = ("/health/live",)

    def filter(self, record: logging.LogRecord) -> bool:
        """当访问日志命中被抑制路径时返回 False。"""
        request_path = self._extract_request_path(record)
        if request_path is None:
            return True
        return not any(
            request_path.startswith(path_prefix) for path_prefix in self._SUPPRESSED_PATH_PREFIXES
        )

    def _extract_request_path(self, record: logging.LogRecord) -> str | None:
        """从 Uvicorn access log record 中提取请求路径。"""
        if record.name != "uvicorn.access":
            return None

        # Uvicorn 默认 access log 的 args 通常为：
        # (client_addr, method, path, http_version, status_code)
        record_args = record.args
        if not isinstance(record_args, tuple) or len(record_args) < 3:
            return None

        request_path = record_args[2]
        if not isinstance(request_path, str):
            return None
        return request_path


def _configure_uvicorn_access_logging() -> None:
    """为 Uvicorn 访问日志安装健康检查过滤器。"""
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    # 避免应用重复初始化时把同一个过滤器叠加多次。
    uvicorn_access_logger.filters = [
        existing_filter
        for existing_filter in uvicorn_access_logger.filters
        if not isinstance(existing_filter, _UvicornAccessPathFilter)
    ]
    uvicorn_access_logger.addFilter(_UvicornAccessPathFilter())


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
    _configure_uvicorn_access_logging()
