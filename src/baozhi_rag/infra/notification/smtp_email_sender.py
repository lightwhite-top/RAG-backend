"""基于 SMTP 的注册验证码邮件发送器。"""

from __future__ import annotations

import html
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
        message.set_content(
            self._build_plain_text_body(
                code=code,
                expires_in_minutes=expires_in_minutes,
            )
        )
        message.add_alternative(
            self._build_html_body(
                code=code,
                expires_in_minutes=expires_in_minutes,
            ),
            subtype="html",
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

    def _build_plain_text_body(self, *, code: str, expires_in_minutes: int) -> str:
        """构造纯文本邮件正文，兼容不支持 HTML 的客户端。"""
        return "\n".join(
            [
                f"你正在注册 {self._app_name}。",
                "",
                f"本次验证码：{code}",
                f"有效期：{expires_in_minutes} 分钟",
                "",
                "如非本人操作，请忽略本邮件。",
            ]
        )

    def _build_html_body(self, *, code: str, expires_in_minutes: int) -> str:
        """构造 HTML 邮件正文，提升验证码邮件的可读性。"""
        escaped_app_name = html.escape(self._app_name)
        escaped_app_name_upper = html.escape(self._app_name.upper())
        escaped_code = html.escape(code)
        escaped_expires = html.escape(str(expires_in_minutes))
        preview_text = html.escape(
            f"你正在注册 {self._app_name}，验证码为 {code}，{expires_in_minutes} 分钟内有效。"
        )

        return f"""\
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>[{escaped_app_name}] 注册验证码</title>
  </head>
  <body style="margin:0;padding:0;background-color:#f4f7fb;color:#0f172a;">
    <div style="display:none;max-height:0;overflow:hidden;opacity:0;mso-hide:all;">
      {preview_text}
    </div>
    <table
      role="presentation"
      width="100%"
      cellpadding="0"
      cellspacing="0"
      border="0"
      style="width:100%;background-color:#f4f7fb;"
    >
      <tr>
        <td align="center" style="padding:24px 12px;">
          <table
            role="presentation"
            width="100%"
            cellpadding="0"
            cellspacing="0"
            border="0"
            style="max-width:560px;background-color:#ffffff;border-radius:24px;overflow:hidden;"
          >
            <tr>
              <td
                style="padding:32px;background-color:#0f172a;background-image:linear-gradient(135deg,#0f172a 0%,#2563eb 100%);color:#ffffff;"
              >
                <p
                  style="margin:0 0 10px;font-size:12px;letter-spacing:1.4px;opacity:0.84;"
                >
                  {escaped_app_name_upper}
                </p>
                <h1 style="margin:0;font-size:28px;line-height:1.35;">邮箱验证码</h1>
                <p style="margin:12px 0 0;font-size:15px;line-height:1.8;opacity:0.92;">
                  正在为你的账号完成邮箱验证，请在注册页面输入下面的验证码。
                </p>
              </td>
            </tr>
            <tr>
              <td style="padding:32px;">
                <p style="margin:0;font-size:14px;line-height:1.8;color:#475569;">
                  本次验证码仅用于注册 {escaped_app_name}，请勿转发给他人。
                </p>
                <div
                  style="margin:24px 0;padding:24px 20px;border:1px solid #bfdbfe;border-radius:20px;background-color:#eff6ff;text-align:center;"
                >
                  <p
                    style="margin:0 0 12px;font-size:13px;letter-spacing:2px;text-transform:uppercase;color:#2563eb;"
                  >
                    Verification Code
                  </p>
                  <p
                    style="margin:0;font-size:34px;line-height:1.2;font-weight:700;letter-spacing:6px;color:#0f172a;font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;white-space:nowrap;"
                  >
                    {escaped_code}
                  </p>
                </div>
                <p style="margin:0;font-size:15px;line-height:1.8;color:#334155;">
                  验证码将在 <strong>{escaped_expires} 分钟</strong> 后失效，输入完成后即可继续注册。
                </p>
                <div
                  style="margin-top:24px;padding:16px 18px;border-radius:16px;background-color:#f8fafc;color:#64748b;font-size:13px;line-height:1.8;"
                >
                  如果这不是你本人的操作，请直接忽略本邮件，无需进行任何处理。
                </div>
              </td>
            </tr>
            <tr>
              <td style="padding:0 32px 32px;">
                <div
                  style="border-top:1px solid #e2e8f0;padding-top:20px;font-size:12px;line-height:1.7;color:#94a3b8;"
                >
                  这是一封系统自动发送的邮件，请勿直接回复。
                </div>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""
