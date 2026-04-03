"""注册邮箱验证码领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RegistrationVerificationCode:
    """注册邮箱验证码记录。"""

    id: str
    email: str
    code_digest: str
    failed_attempts: int
    sent_at: datetime
    expires_at: datetime
    used_at: datetime | None
    invalidated_at: datetime | None

    def is_expired(self, *, now: datetime) -> bool:
        """判断验证码是否已过期。

        参数:
            now: 当前 UTC 时间，用于与过期时间比较。

        返回:
            若验证码已过期则返回 `True`，否则返回 `False`。
        """
        return self.expires_at <= now

    def is_active(self, *, now: datetime) -> bool:
        """判断验证码当前是否仍可用于注册。

        参数:
            now: 当前 UTC 时间，用于综合判断过期、失效与已使用状态。

        返回:
            仅当验证码未过期、未使用且未被新验证码淘汰时返回 `True`。
        """
        return not self.is_expired(now=now) and self.used_at is None and self.invalidated_at is None
