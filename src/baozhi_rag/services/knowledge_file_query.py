"""知识文件列表查询服务。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from baozhi_rag.domain.knowledge_file import KnowledgeFile, KnowledgeFileListPage
from baozhi_rag.domain.knowledge_file_repository import KnowledgeFileRepository
from baozhi_rag.domain.user import CurrentUser


class PresignedFileUrlBuilder(Protocol):
    """文件访问地址生成协议。"""

    def build_presigned_get_url(
        self,
        *,
        storage_key: str,
        expires_seconds: int = 900,
    ) -> str:
        """为对象键生成短时可访问地址。"""


@dataclass(frozen=True, slots=True)
class KnowledgeFileListItemResult:
    """文件列表单项结果。"""

    file_id: str
    uploader_user_id: str
    original_filename: str
    content_type: str
    size: int
    storage_key: str
    file_url: str
    storage_provider: str
    visibility_scope: str
    chunk_count: int
    uploaded_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class KnowledgeFileListResult:
    """文件列表分页结果。"""

    items: list[KnowledgeFileListItemResult]
    total: int
    page: int
    page_size: int


class KnowledgeFileQueryService:
    """负责分页查询文件列表并补充访问地址。"""

    _DEFAULT_FILE_URL_EXPIRES_SECONDS = 900

    def __init__(
        self,
        *,
        knowledge_file_repository: KnowledgeFileRepository,
        file_url_builder: PresignedFileUrlBuilder,
    ) -> None:
        """初始化文件列表查询服务。"""
        self._knowledge_file_repository = knowledge_file_repository
        self._file_url_builder = file_url_builder

    def list_global_files(
        self,
        *,
        page: int,
        page_size: int,
    ) -> KnowledgeFileListResult:
        """分页查询全局文件。"""
        page_result = self._knowledge_file_repository.list_global_files(
            page=page,
            page_size=page_size,
        )
        return self._build_list_result(page_result)

    def list_my_files(
        self,
        *,
        current_user: CurrentUser,
        page: int,
        page_size: int,
    ) -> KnowledgeFileListResult:
        """分页查询当前用户自己上传的文件。"""
        page_result = self._knowledge_file_repository.list_user_files(
            uploader_user_id=current_user.id,
            page=page,
            page_size=page_size,
        )
        return self._build_list_result(page_result)

    def _build_list_result(self, page_result: KnowledgeFileListPage) -> KnowledgeFileListResult:
        """把领域分页结果转换为接口友好的查询结果。"""
        return KnowledgeFileListResult(
            items=[self._build_item(file) for file in page_result.items],
            total=page_result.total,
            page=page_result.page,
            page_size=page_result.page_size,
        )

    def _build_item(self, file: KnowledgeFile) -> KnowledgeFileListItemResult:
        """构造带访问地址的文件列表单项。"""
        # 文件地址在列表查询阶段实时签发，避免把易过期的 URL 长期固化在数据库里。
        file_url = self._file_url_builder.build_presigned_get_url(
            storage_key=file.storage_key,
            expires_seconds=self._DEFAULT_FILE_URL_EXPIRES_SECONDS,
        )
        return KnowledgeFileListItemResult(
            file_id=file.id,
            uploader_user_id=file.uploader_user_id,
            original_filename=file.original_filename,
            content_type=file.content_type,
            size=file.size,
            storage_key=file.storage_key,
            file_url=file_url,
            storage_provider=file.storage_provider.value,
            visibility_scope=file.visibility_scope.value,
            chunk_count=file.chunk_count,
            uploaded_at=file.uploaded_at,
            updated_at=file.updated_at,
        )
