"""聊天会话相关异常。"""

from __future__ import annotations

from fastapi import status

from baozhi_rag.core.exceptions import AppError


class ChatSessionError(AppError):
    """聊天会话模块异常基类。"""

    default_message = "聊天会话处理失败"
    default_error_code = "chat_session_error"
    default_status_code = status.HTTP_400_BAD_REQUEST


class ChatSessionNotFoundError(ChatSessionError):
    """聊天会话不存在。"""

    default_message = "会话不存在"
    default_error_code = "chat_session_not_found"
    default_status_code = status.HTTP_404_NOT_FOUND


class ChatSessionForbiddenError(ChatSessionError):
    """当前用户无权访问聊天会话。"""

    default_message = "无权访问该会话"
    default_error_code = "chat_session_forbidden"
    default_status_code = status.HTTP_403_FORBIDDEN


class ChatSessionDeletedError(ChatSessionError):
    """聊天会话已删除。"""

    default_message = "会话已删除"
    default_error_code = "chat_session_deleted"
    default_status_code = status.HTTP_409_CONFLICT


class ChatSessionValidationError(ChatSessionError):
    """聊天会话参数非法。"""

    default_message = "会话参数非法"
    default_error_code = "chat_session_validation_error"
    default_status_code = status.HTTP_400_BAD_REQUEST


class ChatSessionModeInvalidMessagesError(ChatSessionValidationError):
    """会话模式下的消息列表不符合约束。"""

    default_message = "session_id 模式下 messages 必须且只能包含一条 user 消息"
    default_error_code = "chat_session_mode_invalid_messages"
    default_status_code = status.HTTP_400_BAD_REQUEST


class ChatMessageConflictError(ChatSessionError):
    """聊天消息冲突。"""

    default_message = "聊天消息写入冲突"
    default_error_code = "chat_message_conflict"
    default_status_code = status.HTTP_409_CONFLICT
