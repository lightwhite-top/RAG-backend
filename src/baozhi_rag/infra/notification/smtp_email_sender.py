"""基于 SMTP 的注册验证码邮件发送器。"""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr

from baozhi_rag.core.config import Settings
from baozhi_rag.domain.user_errors import EmailDeliveryFailedError


class SmtpRegistrationEmailSender:
    """使用 SMTP 发送注册验证码邮件。"""

    def __init__(
        self,
        *,
        host: str | None,
        port: int,
        username: str | None,
        password: str | None,
        use_tls: bool,
        use_ssl: bool,
        from_email: str | None,
        from_name: str,
        timeout_seconds: float,
        app_name: str,
    ) -> None:
        """初始化邮件发送器。

        参数:
            host: SMTP 主机地址。
            port: SMTP 端口。
            username: SMTP 登录用户名，可留空。
            password: SMTP 登录密码，可留空。
            use_tls: 是否在明文 SMTP 上升级到 TLS。
            use_ssl: 是否直接使用 SMTPS。
            from_email: 发件邮箱地址。
            from_name: 发件人展示名称。
            timeout_seconds: SMTP 网络超时时间。
            app_name: 应用名称，用于邮件主题和正文。

        返回:
            None。
        """
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._use_tls = use_tls
        self._use_ssl = use_ssl
        self._from_email = from_email
        self._from_name = from_name
        self._timeout_seconds = timeout_seconds
        self._app_name = app_name

    def send_registration_code(
        self,
        *,
        to_email: str,
        code: str,
        expires_in_minutes: int,
    ) -> None:
        """发送注册验证码邮件。

        参数:
            to_email: 收件邮箱地址。
            code: 本次发送的验证码原文。
            expires_in_minutes: 验证码有效期，单位为分钟。

        返回:
            None。

        异常:
            EmailDeliveryFailedError: SMTP 未配置完整或邮件发送失败时抛出。
        """
        self._ensure_configured()
        host = self._host
        if host is None:
            raise EmailDeliveryFailedError("注册验证码邮件通道未配置完成")

        message = EmailMessage()
        message["Subject"] = f"[{self._app_name}] 注册验证码"
        message["From"] = formataddr((self._from_name, str(self._from_email)))
        message["To"] = to_email
        # 纯文本邮件兼容性最好，适合验证码这类高到达率场景。
        message.set_content(
            "\n".join(
                [
                    f"你正在注册 {self._app_name}。",
                    f"本次验证码：{code}",
                    f"有效期：{expires_in_minutes} 分钟",
                    "如非本人操作，请忽略本邮件。",
                ]
            )
        )

        try:
            if self._use_ssl:
                with smtplib.SMTP_SSL(
                    host,
                    self._port,
                    timeout=self._timeout_seconds,
                    context=ssl.create_default_context(),
                ) as client:
                    self._login_if_needed(client)
                    client.send_message(message)
                return

            with smtplib.SMTP(
                host,
                self._port,
                timeout=self._timeout_seconds,
            ) as client:
                if self._use_tls:
                    client.starttls(context=ssl.create_default_context())
                self._login_if_needed(client)
                client.send_message(message)
        except (smtplib.SMTPException, OSError) as exc:
            raise EmailDeliveryFailedError() from exc

    @classmethod
    def from_settings(cls, settings: Settings) -> SmtpRegistrationEmailSender:
        """基于应用配置构造邮件发送器。

        参数:
            settings: 当前应用配置对象。

        返回:
            一个延迟校验 SMTP 配置的邮件发送器实例。
        """
        return cls(
            host=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password,
            use_tls=settings.smtp_use_tls,
            use_ssl=settings.smtp_use_ssl,
            from_email=settings.smtp_from_email,
            from_name=settings.smtp_from_name or settings.app_name,
            timeout_seconds=settings.smtp_timeout_seconds,
            app_name=settings.app_name,
        )

    def _ensure_configured(self) -> None:
        """校验注册验证码邮件发送所需的最小配置。"""
        if not self._host or not self._from_email:
            raise EmailDeliveryFailedError("注册验证码邮件通道未配置完成")

    def _login_if_needed(self, client: smtplib.SMTP) -> None:
        """按需执行 SMTP 认证。"""
        if not self._username:
            return
        client.login(self._username, self._password or "")
