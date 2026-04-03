"""用户与认证相关异常。"""

from __future__ import annotations

from fastapi import status

from baozhi_rag.core.exceptions import AppError


class UserError(AppError):
    """用户与认证模块异常基类。"""

    default_message = "用户操作失败"
    default_error_code = "user_error"
    default_status_code = status.HTTP_400_BAD_REQUEST


class EmailAlreadyExistsError(UserError):
    """邮箱已存在。"""

    default_message = "邮箱已存在"
    default_error_code = "email_already_exists"
    default_status_code = status.HTTP_409_CONFLICT


class UsernameAlreadyExistsError(UserError):
    """用户名已存在。"""

    default_message = "用户名已存在"
    default_error_code = "username_already_exists"
    default_status_code = status.HTTP_409_CONFLICT


class UserNotFoundError(UserError):
    """用户不存在。"""

    default_message = "用户不存在"
    default_error_code = "user_not_found"
    default_status_code = status.HTTP_404_NOT_FOUND


class InvalidCredentialsError(UserError):
    """登录凭证非法。"""

    default_message = "邮箱或密码错误"
    default_error_code = "invalid_credentials"
    default_status_code = status.HTTP_401_UNAUTHORIZED


class AuthenticationRequiredError(UserError):
    """未提供有效登录态。"""

    default_message = "未提供有效的身份凭证"
    default_error_code = "unauthorized"
    default_status_code = status.HTTP_401_UNAUTHORIZED


class AccessTokenInvalidError(UserError):
    """访问令牌非法。"""

    default_message = "访问令牌无效"
    default_error_code = "token_invalid"
    default_status_code = status.HTTP_401_UNAUTHORIZED


class AccessTokenExpiredError(UserError):
    """访问令牌已过期。"""

    default_message = "登录已过期，请重新登录"
    default_error_code = "token_expired"
    default_status_code = status.HTTP_401_UNAUTHORIZED


class PermissionDeniedError(UserError):
    """无权执行当前操作。"""

    default_message = "无权执行当前操作"
    default_error_code = "forbidden"
    default_status_code = status.HTTP_403_FORBIDDEN


class InvalidUserOperationError(UserError):
    """用户操作参数或状态不合法。"""

    default_message = "用户操作不合法"
    default_error_code = "invalid_user_operation"
    default_status_code = status.HTTP_400_BAD_REQUEST


class RegistrationCodeNotFoundError(UserError):
    """尚未获取可用注册验证码。"""

    default_message = "请先获取注册验证码"
    default_error_code = "registration_code_not_found"
    default_status_code = status.HTTP_400_BAD_REQUEST


class RegistrationCodeExpiredError(UserError):
    """注册验证码已过期。"""

    default_message = "验证码已过期，请重新获取"
    default_error_code = "registration_code_expired"
    default_status_code = status.HTTP_400_BAD_REQUEST


class RegistrationCodeInvalidError(UserError):
    """注册验证码不正确。"""

    default_message = "验证码错误"
    default_error_code = "registration_code_invalid"
    default_status_code = status.HTTP_400_BAD_REQUEST


class RegistrationCodeAttemptsExceededError(UserError):
    """注册验证码尝试次数过多。"""

    default_message = "验证码错误次数过多，请重新获取"
    default_error_code = "registration_code_attempts_exceeded"
    default_status_code = status.HTTP_400_BAD_REQUEST


class RegistrationCodeSendTooFrequentError(UserError):
    """注册验证码发送过于频繁。"""

    default_message = "验证码发送过于频繁，请稍后再试"
    default_error_code = "registration_code_send_too_frequent"
    default_status_code = status.HTTP_429_TOO_MANY_REQUESTS


class EmailDeliveryFailedError(UserError):
    """邮件发送失败。"""

    default_message = "验证码邮件发送失败，请稍后再试"
    default_error_code = "email_delivery_failed"
    default_status_code = status.HTTP_503_SERVICE_UNAVAILABLE
