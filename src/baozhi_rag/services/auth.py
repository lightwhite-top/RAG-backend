"""认证与当前用户相关服务。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from baozhi_rag.domain.registration_email_sender import RegistrationEmailSender
from baozhi_rag.domain.registration_verification import RegistrationVerificationCode
from baozhi_rag.domain.registration_verification_repository import (
    RegistrationVerificationRepository,
)
from baozhi_rag.domain.user import CurrentUser, User, UserRole, build_current_user
from baozhi_rag.domain.user_errors import (
    AuthenticationRequiredError,
    EmailAlreadyExistsError,
    InvalidCredentialsError,
    InvalidUserOperationError,
    RegistrationCodeAttemptsExceededError,
    RegistrationCodeExpiredError,
    RegistrationCodeInvalidError,
    RegistrationCodeNotFoundError,
    RegistrationCodeSendTooFrequentError,
    UserNotFoundError,
)
from baozhi_rag.domain.user_repository import UserRepository
from baozhi_rag.infra.security.jwt_tokens import JwtTokenManager
from baozhi_rag.infra.security.passwords import PasswordHasherAdapter
from baozhi_rag.infra.security.registration_codes import RegistrationCodeManager


@dataclass(frozen=True, slots=True)
class LoginResult:
    """登录成功后的返回结果。"""

    access_token: str
    token_type: str
    expires_in_seconds: int
    expires_at: datetime
    user: CurrentUser


@dataclass(frozen=True, slots=True)
class SendRegistrationCodeResult:
    """注册验证码发送成功后的返回结果。"""

    expires_in_seconds: int
    resend_interval_seconds: int


class AuthService:
    """编排注册、登录与当前用户信息。"""

    def __init__(
        self,
        *,
        user_repository: UserRepository,
        password_hasher: PasswordHasherAdapter,
        token_manager: JwtTokenManager,
        registration_code_repository: RegistrationVerificationRepository,
        registration_email_sender: RegistrationEmailSender,
        registration_code_manager: RegistrationCodeManager,
    ) -> None:
        """初始化认证服务。"""
        self._user_repository = user_repository
        self._password_hasher = password_hasher
        self._token_manager = token_manager
        self._registration_code_repository = registration_code_repository
        self._registration_email_sender = registration_email_sender
        self._registration_code_manager = registration_code_manager

    def send_registration_code(self, *, email: str) -> SendRegistrationCodeResult:
        """发送邮箱注册验证码。

        参数:
            email: 待注册邮箱地址。

        返回:
            包含验证码有效期和重发等待时间的结果对象。

        异常:
            EmailAlreadyExistsError: 邮箱已被注册。
            RegistrationCodeSendTooFrequentError: 距离上次发码时间过短。
        """
        normalized_email = self._normalize_email(email)
        if self._user_repository.get_user_by_email(normalized_email) is not None:
            raise EmailAlreadyExistsError()

        now = datetime.now(UTC)
        latest_code = self._registration_code_repository.get_latest_code(normalized_email)
        if latest_code is not None and latest_code.is_active(now=now):
            elapsed_seconds = int((now - latest_code.sent_at).total_seconds())
            resend_interval_seconds = self._registration_code_manager.policy.resend_interval_seconds
            if elapsed_seconds < resend_interval_seconds:
                remaining_seconds = resend_interval_seconds - elapsed_seconds
                raise RegistrationCodeSendTooFrequentError(
                    f"验证码发送过于频繁，请在 {remaining_seconds} 秒后重试"
                )

        verification_code = self._registration_code_manager.generate_code()
        code_record = self._store_registration_code(
            email=normalized_email,
            verification_code=verification_code,
            now=now,
        )

        try:
            self._registration_email_sender.send_registration_code(
                to_email=normalized_email,
                code=verification_code,
                expires_in_minutes=self._registration_code_manager.policy.expire_minutes,
            )
        except Exception:
            # 邮件发送失败时立即让本次验证码失效，避免数据库中遗留不可达的有效码。
            self._registration_code_repository.invalidate_code(
                code_record.id,
                invalidated_at=now,
            )
            raise

        return SendRegistrationCodeResult(
            expires_in_seconds=self._registration_code_manager.policy.expire_minutes * 60,
            resend_interval_seconds=self._registration_code_manager.policy.resend_interval_seconds,
        )

    def register(
        self,
        *,
        email: str,
        password: str,
        username: str,
        verification_code: str,
    ) -> CurrentUser:
        """注册普通用户并校验邮箱验证码。"""
        normalized_email = self._normalize_email(email)
        if self._user_repository.get_user_by_email(normalized_email) is not None:
            raise EmailAlreadyExistsError()

        normalized_username = self._normalize_username(username)
        code_record = self._require_valid_registration_code(
            email=normalized_email,
            verification_code=verification_code,
        )
        password_hash = self._password_hasher.hash_password(password)
        user = self._user_repository.create_user(
            email=normalized_email,
            username=normalized_username,
            password_hash=password_hash,
            role=UserRole.USER,
        )
        self._registration_code_repository.mark_used(
            code_record.id,
            used_at=datetime.now(UTC),
        )
        return build_current_user(user)

    def login(self, *, email: str, password: str) -> LoginResult:
        """校验凭证并签发访问令牌。"""
        normalized_email = self._normalize_email(email)
        user = self._user_repository.get_user_by_email(normalized_email)
        if user is None or not self._password_hasher.verify_password(password, user.password_hash):
            raise InvalidCredentialsError()

        issue_result = self._token_manager.issue_access_token(user_id=user.id, role=user.role)
        return LoginResult(
            access_token=issue_result.access_token,
            token_type="Bearer",
            expires_in_seconds=issue_result.expires_in_seconds,
            expires_at=issue_result.expires_at,
            user=build_current_user(user),
        )

    def get_current_user_from_token(self, token: str) -> CurrentUser:
        """基于访问令牌解析并查询当前用户。"""
        payload = self._token_manager.decode_access_token(token)
        user = self._user_repository.get_user_by_id(payload.subject)
        if user is None:
            raise AuthenticationRequiredError("用户不存在或登录已失效")
        return build_current_user(user)

    def update_profile(self, *, user_id: str, username: str) -> CurrentUser:
        """更新当前用户的基础资料。"""
        updated_user = self._user_repository.update_user(
            user_id,
            username=self._normalize_username(username),
        )
        if updated_user is None:
            raise UserNotFoundError()
        return build_current_user(updated_user)

    def change_password(
        self,
        *,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> None:
        """校验旧密码并更新为新密码。"""
        user = self._require_user(user_id)
        if not self._password_hasher.verify_password(current_password, user.password_hash):
            raise InvalidCredentialsError("当前密码错误")

        new_password_hash = self._password_hasher.hash_password(new_password)
        updated_user = self._user_repository.update_user(
            user_id,
            password_hash=new_password_hash,
        )
        if updated_user is None:
            raise UserNotFoundError()

    def _require_user(self, user_id: str) -> User:
        """按 ID 获取用户，不存在则抛错。"""
        user = self._user_repository.get_user_by_id(user_id)
        if user is None:
            raise UserNotFoundError()
        return user

    def _store_registration_code(
        self,
        *,
        email: str,
        verification_code: str,
        now: datetime,
    ) -> RegistrationVerificationCode:
        """保存新的注册验证码，并让旧验证码统一失效。"""
        expires_at = now + timedelta(minutes=self._registration_code_manager.policy.expire_minutes)
        code_digest = self._registration_code_manager.build_code_digest(
            email=email,
            code=verification_code,
        )
        # 先淘汰旧码再写入新码，确保一次只允许最新一封邮件里的验证码生效。
        self._registration_code_repository.invalidate_active_codes(
            email=email,
            invalidated_at=now,
        )
        return self._registration_code_repository.create_code(
            email=email,
            code_digest=code_digest,
            sent_at=now,
            expires_at=expires_at,
        )

    def _require_valid_registration_code(
        self,
        *,
        email: str,
        verification_code: str,
    ) -> RegistrationVerificationCode:
        """校验注册验证码是否存在、未过期且未超过最大尝试次数。"""
        now = datetime.now(UTC)
        code_record = self._registration_code_repository.get_latest_code(email)
        if code_record is None:
            raise RegistrationCodeNotFoundError()
        if code_record.used_at is not None or code_record.invalidated_at is not None:
            raise RegistrationCodeNotFoundError("验证码已失效，请重新获取")
        if code_record.is_expired(now=now):
            raise RegistrationCodeExpiredError()
        if code_record.failed_attempts >= self._registration_code_manager.policy.max_attempts:
            self._registration_code_repository.invalidate_code(
                code_record.id,
                invalidated_at=now,
            )
            raise RegistrationCodeAttemptsExceededError()

        expected_digest = self._registration_code_manager.build_code_digest(
            email=email,
            code=self._normalize_verification_code(verification_code),
        )
        if code_record.code_digest == expected_digest:
            return code_record

        updated_code_record = self._registration_code_repository.increment_failed_attempts(
            code_record.id
        )
        next_failed_attempts = (
            updated_code_record.failed_attempts
            if updated_code_record is not None
            else code_record.failed_attempts + 1
        )
        if next_failed_attempts >= self._registration_code_manager.policy.max_attempts:
            self._registration_code_repository.invalidate_code(
                code_record.id,
                invalidated_at=now,
            )
            raise RegistrationCodeAttemptsExceededError()
        raise RegistrationCodeInvalidError()

    def _normalize_email(self, email: str) -> str:
        """统一规整邮箱格式。"""
        normalized_email = email.strip().lower()
        if not normalized_email:
            raise InvalidUserOperationError("邮箱不能为空")
        return normalized_email

    def _normalize_username(self, username: str) -> str:
        """统一规整用户名。"""
        normalized_username = username.strip()
        if not normalized_username:
            raise InvalidUserOperationError("用户名不能为空")
        return normalized_username

    def _normalize_verification_code(self, verification_code: str) -> str:
        """统一规整注册验证码。"""
        normalized_code = verification_code.strip()
        if not normalized_code:
            raise InvalidUserOperationError("验证码不能为空")
        if not normalized_code.isdigit():
            raise InvalidUserOperationError("验证码必须为数字")
        return normalized_code
