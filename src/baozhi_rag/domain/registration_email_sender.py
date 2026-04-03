"""注册验证码邮件发送协议。"""

from __future__ import annotations

from typing import Protocol


class RegistrationEmailSender(Protocol):
    """注册验证码邮件发送器协议。"""

    def send_registration_code(
        self,
        *,
        to_email: str,
        code: str,
        expires_in_minutes: int,
    ) -> None:
        """发送注册验证码邮件。

        参数:
            to_email: 收件人邮箱地址。
            code: 本次发送的纯文本验证码。
            expires_in_minutes: 验证码有效期，单位为分钟。

        返回:
            None。

        异常:
            发送失败时抛出业务异常，供上层转为统一错误响应。
        """
