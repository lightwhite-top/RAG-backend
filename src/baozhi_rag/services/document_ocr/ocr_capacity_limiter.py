"""本地 OCR 容量门控。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from threading import BoundedSemaphore, Lock
from typing import Protocol

from baozhi_rag.core.exceptions import AppError


class OcrCapacityPendingError(AppError):
    """OCR 容量暂不可用时抛出的可恢复异常。"""

    default_message = "OCR 通道繁忙，请稍后重试"
    default_error_code = "ocr_capacity_pending"
    default_status_code = 503

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after_seconds: float,
    ) -> None:
        """初始化 OCR 容量等待异常。"""
        super().__init__(message or self.default_message)
        self.retry_after_seconds = retry_after_seconds


class OcrTaskLimiterProtocol(Protocol):
    """PDF 解析链依赖的 OCR 容量门控协议。"""

    def reserve_slot(self, *, page_number: int | None = None) -> AbstractContextManager[None]:
        """申请一次 OCR 执行槽位。"""


@dataclass(slots=True)
class LocalOcrTaskLimiter(OcrTaskLimiterProtocol):
    """基于进程内信号量的 OCR 限流器。

    参数:
        max_concurrent: 同时允许执行 OCR 的任务数量。
        max_waiters: 允许短暂等待 OCR 槽位的线程数。
        wait_timeout_seconds: 等待 OCR 槽位的最长时间，超时后要求任务回队。
    """

    max_concurrent: int = 1
    max_waiters: int = 2
    wait_timeout_seconds: float = 1.0
    _slots: BoundedSemaphore = field(init=False, repr=False)
    _waiters: int = field(default=0, init=False, repr=False)
    _waiter_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """初始化并校验本地 OCR 限流器。"""
        if self.max_concurrent <= 0:
            msg = "OCR 最大并发数必须大于 0"
            raise ValueError(msg)
        if self.max_waiters < 0:
            msg = "OCR 最大等待线程数不能小于 0"
            raise ValueError(msg)
        if self.wait_timeout_seconds <= 0:
            msg = "OCR 等待超时时间必须大于 0"
            raise ValueError(msg)
        self._slots = BoundedSemaphore(self.max_concurrent)

    @contextmanager
    def reserve_slot(self, *, page_number: int | None = None) -> Iterator[None]:
        """申请一次 OCR 执行槽位。"""
        self._acquire_waiter_slot(page_number=page_number)
        acquired = False
        try:
            acquired = self._slots.acquire(timeout=self.wait_timeout_seconds)
        finally:
            self._release_waiter_slot()

        if not acquired:
            page_hint = f": page_number={page_number}" if page_number is not None else ""
            raise OcrCapacityPendingError(
                f"OCR 槽位等待超时{page_hint}",
                retry_after_seconds=self.wait_timeout_seconds,
            )

        try:
            yield
        finally:
            self._slots.release()

    def _acquire_waiter_slot(self, *, page_number: int | None) -> None:
        """限制同时等待 OCR 的线程数量。"""
        with self._waiter_lock:
            if self._waiters >= self.max_waiters:
                page_hint = f": page_number={page_number}" if page_number is not None else ""
                raise OcrCapacityPendingError(
                    f"OCR 等待队列已满{page_hint}",
                    retry_after_seconds=self.wait_timeout_seconds,
                )
            self._waiters += 1

    def _release_waiter_slot(self) -> None:
        """释放等待名额。"""
        with self._waiter_lock:
            if self._waiters > 0:
                self._waiters -= 1
