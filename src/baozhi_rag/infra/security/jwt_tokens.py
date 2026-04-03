"""JWT 令牌管理。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from baozhi_rag.core.config import Settings
from baozhi_rag.domain.user import UserRole
from baozhi_rag.domain.user_errors import AccessTokenExpiredError, AccessTokenInvalidError


@dataclass(frozen=True, slots=True)
class AccessTokenIssueResult:
    """签发访问令牌后的结果。"""

    access_token: str
    expires_at: datetime
    expires_in_seconds: int


@dataclass(frozen=True, slots=True)
class AccessTokenPayload:
    """解码后的访问令牌载荷。"""

    subject: str
    role: UserRole
    issued_at: datetime
    expires_at: datetime


class JwtTokenManager:
    """负责签发与校验 JWT。"""

    def __init__(
        self,
        *,
        secret_key: str,
        algorithm: str,
        expire_days: int,
    ) -> None:
        """初始化 JWT 管理器。"""
        self._secret_key = secret_key
        self._algorithm = algorithm
        self._expire_days = expire_days

    def issue_access_token(self, *, user_id: str, role: UserRole) -> AccessTokenIssueResult:
        """为指定用户签发访问令牌。"""
        issued_at = datetime.now(UTC)
        expires_at = issued_at + timedelta(days=self._expire_days)
        payload = {
            "sub": user_id,
            "role": role.value,
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        token = jwt.encode(payload, self._secret_key, algorithm=self._algorithm)
        return AccessTokenIssueResult(
            access_token=token,
            expires_at=expires_at,
            expires_in_seconds=int((expires_at - issued_at).total_seconds()),
        )

    def decode_access_token(self, token: str) -> AccessTokenPayload:
        """校验并解析访问令牌。"""
        try:
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
            )
        except ExpiredSignatureError as exc:
            raise AccessTokenExpiredError() from exc
        except InvalidTokenError as exc:
            raise AccessTokenInvalidError() from exc

        subject = str(payload.get("sub", "")).strip()
        role_text = str(payload.get("role", "")).strip()
        issued_at = _read_timestamp(payload.get("iat"))
        expires_at = _read_timestamp(payload.get("exp"))
        if not subject or not role_text or issued_at is None or expires_at is None:
            raise AccessTokenInvalidError()

        try:
            role = UserRole(role_text)
        except ValueError as exc:
            raise AccessTokenInvalidError() from exc

        return AccessTokenPayload(
            subject=subject,
            role=role,
            issued_at=issued_at,
            expires_at=expires_at,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> JwtTokenManager:
        """基于应用配置构造 JWT 管理器。"""
        return JwtTokenManager(
            secret_key=settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
            expire_days=settings.jwt_access_token_expire_days,
        )


def _read_timestamp(value: object) -> datetime | None:
    """安全地把时间戳字段解析为 UTC 时间。"""
    if not isinstance(value, int | float):
        return None
    return datetime.fromtimestamp(float(value), tz=UTC)
