"""注册邮箱验证码生成与摘要工具。"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import string
from dataclasses import dataclass

from baozhi_rag.core.config import Settings


@dataclass(frozen=True, slots=True)
class RegistrationCodePolicy:
    """注册验证码策略配置。"""

    length: int
    expire_minutes: int
    resend_interval_seconds: int
    max_attempts: int


class RegistrationCodeManager:
    """负责生成注册验证码并计算安全摘要。"""

    def __init__(self, *, secret: str, policy: RegistrationCodePolicy) -> None:
        """初始化验证码管理器。

        参数:
            secret: 用于计算验证码摘要的签名密钥。
            policy: 注册验证码长度、时效和重试限制等策略。

        返回:
            None。
        """
        self._secret = secret.encode()
        self._policy = policy

    @property
    def policy(self) -> RegistrationCodePolicy:
        """返回当前验证码策略。"""
        return self._policy

    def generate_code(self) -> str:
        """生成纯数字注册验证码。

        返回:
            指定位数的数字验证码字符串。
        """
        digits = string.digits
        return "".join(secrets.choice(digits) for _ in range(self._policy.length))

    def build_code_digest(self, *, email: str, code: str) -> str:
        """计算注册验证码的不可逆摘要。

        参数:
            email: 与验证码绑定的邮箱地址。
            code: 用户收到的纯文本验证码。

        返回:
            基于 HMAC-SHA256 计算出的摘要字符串。
        """
        normalized_email = email.strip().lower()
        normalized_code = code.strip()
        payload = f"{normalized_email}:{normalized_code}".encode()
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    @classmethod
    def from_settings(cls, settings: Settings) -> RegistrationCodeManager:
        """从应用配置构造验证码管理器。

        参数:
            settings: 当前应用配置对象。

        返回:
            已填充策略参数的验证码管理器实例。
        """
        return cls(
            secret=settings.registration_code_secret,
            policy=RegistrationCodePolicy(
                length=settings.registration_code_length,
                expire_minutes=settings.registration_code_expire_minutes,
                resend_interval_seconds=settings.registration_code_resend_interval_seconds,
                max_attempts=settings.registration_code_max_attempts,
            ),
        )
