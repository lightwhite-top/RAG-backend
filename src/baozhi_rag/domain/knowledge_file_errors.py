"""知识文件相关异常。"""

from __future__ import annotations

from fastapi import status

from baozhi_rag.core.exceptions import AppError


class KnowledgeFileError(AppError):
    """知识文件模块异常基类。"""

    default_message = "文件处理失败"
    default_error_code = "knowledge_file_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class KnowledgeFileNotFoundError(KnowledgeFileError):
    """文件记录不存在。"""

    default_message = "文件不存在"
    default_error_code = "knowledge_file_not_found"
    default_status_code = status.HTTP_404_NOT_FOUND


class KnowledgeFileConflictError(KnowledgeFileError):
    """文件唯一约束冲突。"""

    default_message = "文件记录冲突"
    default_error_code = "knowledge_file_conflict"
    default_status_code = status.HTTP_409_CONFLICT


class KnowledgeUploadTaskNotFoundError(KnowledgeFileError):
    """上传任务不存在。"""

    default_message = "上传任务不存在"
    default_error_code = "knowledge_upload_task_not_found"
    default_status_code = status.HTTP_404_NOT_FOUND


class KnowledgeUploadTaskRetryNotAllowedError(KnowledgeFileError):
    """上传任务当前状态不允许重试。"""

    default_message = "当前任务状态不允许重试"
    default_error_code = "knowledge_upload_task_retry_not_allowed"
    default_status_code = status.HTTP_409_CONFLICT


class ObjectStorageError(KnowledgeFileError):
    """对象存储调用失败。"""

    default_message = "对象存储调用失败"
    default_error_code = "object_storage_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class ObjectStorageDependencyError(ObjectStorageError):
    """对象存储依赖缺失或配置不可用。"""

    default_message = "对象存储依赖不可用"
    default_error_code = "object_storage_dependency_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
