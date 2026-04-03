"""注册邮箱验证码仓储协议。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from baozhi_rag.domain.registration_verification import RegistrationVerificationCode


class RegistrationVerificationRepository(Protocol):
    """注册邮箱验证码仓储协议。"""

    def create_code(
        self,
        *,
        email: str,
        code_digest: str,
        sent_at: datetime,
        expires_at: datetime,
    ) -> RegistrationVerificationCode:
        """创建新的注册验证码记录。"""

    def get_latest_code(self, email: str) -> RegistrationVerificationCode | None:
        """按邮箱获取最新的一条验证码记录。"""

    def invalidate_active_codes(self, *, email: str, invalidated_at: datetime) -> int:
        """把指定邮箱下尚未使用的旧验证码统一置为失效。"""

    def invalidate_code(
        self,
        code_id: str,
        *,
        invalidated_at: datetime,
    ) -> RegistrationVerificationCode | None:
        """把指定验证码记录置为失效。"""

    def increment_failed_attempts(
        self,
        code_id: str,
    ) -> RegistrationVerificationCode | None:
        """把指定验证码记录的失败次数加一。"""

    def mark_used(
        self,
        code_id: str,
        *,
        used_at: datetime,
    ) -> RegistrationVerificationCode | None:
        """把指定验证码记录标记为已使用。"""
