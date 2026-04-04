"""API 依赖构造函数。"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.domain.knowledge_file_repository import KnowledgeFileRepository
from baozhi_rag.domain.knowledge_upload_task_repository import KnowledgeUploadTaskRepository
from baozhi_rag.domain.registration_verification_repository import (
    RegistrationVerificationRepository,
)
from baozhi_rag.domain.user import CurrentUser, UserRole
from baozhi_rag.domain.user_errors import AuthenticationRequiredError, PermissionDeniedError
from baozhi_rag.domain.user_repository import UserRepository
from baozhi_rag.infra.database.knowledge_file_repository import SqlAlchemyKnowledgeFileRepository
from baozhi_rag.infra.database.knowledge_upload_task_repository import (
    SqlAlchemyKnowledgeUploadTaskRepository,
)
from baozhi_rag.infra.database.mysql import DatabaseManager
from baozhi_rag.infra.database.registration_verification_repository import (
    SqlAlchemyRegistrationVerificationRepository,
)
from baozhi_rag.infra.database.user_repository import SqlAlchemyUserRepository
from baozhi_rag.infra.llm.aliyun_model_studio import AlibabaModelStudioClient
from baozhi_rag.infra.notification.smtp_email_sender import SmtpRegistrationEmailSender
from baozhi_rag.infra.retrieval.hybrid_chunk_store import HybridChunkStore
from baozhi_rag.infra.security.jwt_tokens import JwtTokenManager
from baozhi_rag.infra.security.passwords import PasswordHasherAdapter
from baozhi_rag.infra.security.registration_codes import RegistrationCodeManager
from baozhi_rag.infra.storage.aliyun_oss_file_store import AliyunOssFileStore
from baozhi_rag.infra.storage.local_file_store import LocalFileStore
from baozhi_rag.services.auth import AuthService
from baozhi_rag.services.chat import ChatService
from baozhi_rag.services.chunk_embedding import ChunkEmbeddingService
from baozhi_rag.services.chunk_search import ChunkSearchService
from baozhi_rag.services.document_chunking import DocumentChunkService
from baozhi_rag.services.document_preview import DocumentPreviewService
from baozhi_rag.services.file_upload import FileUploadService
from baozhi_rag.services.knowledge_file_delete import KnowledgeFileDeleteService
from baozhi_rag.services.knowledge_file_query import KnowledgeFileQueryService
from baozhi_rag.services.term_matching import build_default_term_matcher
from baozhi_rag.services.upload_tasks import KnowledgeUploadService
from baozhi_rag.services.user_admin import UserAdminService

bearer_scheme = HTTPBearer(auto_error=False)


def _build_chunk_embedding_service(settings: Settings) -> ChunkEmbeddingService:
    """构造必选的 chunk 向量化服务。"""
    return ChunkEmbeddingService(AlibabaModelStudioClient.from_settings(settings))


def get_database_manager(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DatabaseManager:
    """构造数据库管理器。"""
    return DatabaseManager.from_settings(settings)


def get_user_repository(
    database_manager: Annotated[DatabaseManager, Depends(get_database_manager)],
) -> UserRepository:
    """构造用户仓储。"""
    return SqlAlchemyUserRepository(database_manager.session_factory)


def get_knowledge_file_repository(
    database_manager: Annotated[DatabaseManager, Depends(get_database_manager)],
) -> KnowledgeFileRepository:
    """构造知识文件仓储。"""
    return SqlAlchemyKnowledgeFileRepository(database_manager.session_factory)


def get_registration_verification_repository(
    database_manager: Annotated[DatabaseManager, Depends(get_database_manager)],
) -> RegistrationVerificationRepository:
    """构造注册验证码仓储。"""
    return SqlAlchemyRegistrationVerificationRepository(database_manager.session_factory)


def get_knowledge_upload_task_repository(
    database_manager: Annotated[DatabaseManager, Depends(get_database_manager)],
) -> KnowledgeUploadTaskRepository:
    """构造上传任务仓储。"""
    return SqlAlchemyKnowledgeUploadTaskRepository(database_manager.session_factory)


def get_local_temp_file_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LocalFileStore:
    """构造本地临时文件存储。"""
    return LocalFileStore(settings.upload_root_dir)


def get_aliyun_oss_file_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AliyunOssFileStore:
    """构造阿里云 OSS 文件存储。"""
    return AliyunOssFileStore.from_settings(settings)


def get_document_preview_service(
    settings: Annotated[Settings, Depends(get_settings)],
    temp_file_store: Annotated[LocalFileStore, Depends(get_local_temp_file_store)],
    object_store: Annotated[AliyunOssFileStore, Depends(get_aliyun_oss_file_store)],
    knowledge_file_repository: Annotated[
        KnowledgeFileRepository,
        Depends(get_knowledge_file_repository),
    ],
) -> DocumentPreviewService:
    """构造文件上传与切块预览服务。"""
    term_matcher = build_default_term_matcher(settings.domain_dictionary_path)
    chunk_store = HybridChunkStore.from_settings(settings)
    chunk_embedding_service = _build_chunk_embedding_service(settings)

    return DocumentPreviewService(
        file_upload_service=FileUploadService(temp_file_store),
        chunk_service=DocumentChunkService(
            chunk_size=settings.doc_chunk_size,
            chunk_overlap=settings.doc_chunk_overlap,
            convert_temp_dir=settings.doc_convert_temp_dir,
            doc_convert_timeout_seconds=settings.doc_convert_timeout_seconds,
            term_matcher=term_matcher,
        ),
        temp_file_store=temp_file_store,
        object_store=object_store,
        knowledge_file_repository=knowledge_file_repository,
        chunk_store=chunk_store,
        chunk_embedding_service=chunk_embedding_service,
        oss_object_prefix=settings.normalized_oss_object_prefix,
    )


def get_knowledge_upload_service(
    settings: Annotated[Settings, Depends(get_settings)],
    temp_file_store: Annotated[LocalFileStore, Depends(get_local_temp_file_store)],
    task_repository: Annotated[
        KnowledgeUploadTaskRepository,
        Depends(get_knowledge_upload_task_repository),
    ],
) -> KnowledgeUploadService:
    """构造知识文件异步上传任务服务。"""
    return KnowledgeUploadService(
        file_upload_service=FileUploadService(temp_file_store),
        temp_file_store=temp_file_store,
        task_repository=task_repository,
        ingest_version=settings.upload_ingest_version,
    )


def get_knowledge_file_query_service(
    knowledge_file_repository: Annotated[
        KnowledgeFileRepository,
        Depends(get_knowledge_file_repository),
    ],
    object_store: Annotated[AliyunOssFileStore, Depends(get_aliyun_oss_file_store)],
) -> KnowledgeFileQueryService:
    """构造知识文件列表查询服务。"""
    return KnowledgeFileQueryService(
        knowledge_file_repository=knowledge_file_repository,
        file_url_builder=object_store,
    )


def get_knowledge_file_delete_service(
    settings: Annotated[Settings, Depends(get_settings)],
    knowledge_file_repository: Annotated[
        KnowledgeFileRepository,
        Depends(get_knowledge_file_repository),
    ],
    object_store: Annotated[AliyunOssFileStore, Depends(get_aliyun_oss_file_store)],
) -> KnowledgeFileDeleteService:
    """构造知识文件删除服务。"""
    return KnowledgeFileDeleteService(
        knowledge_file_repository=knowledge_file_repository,
        chunk_store=HybridChunkStore.from_settings(settings),
        object_store=object_store,
    )


def get_chunk_search_service(
    settings: Annotated[Settings, Depends(get_settings)],
    knowledge_file_repository: Annotated[
        KnowledgeFileRepository,
        Depends(get_knowledge_file_repository),
    ],
) -> ChunkSearchService:
    """构造 chunk 检索服务。"""
    return ChunkSearchService(
        term_matcher=build_default_term_matcher(settings.domain_dictionary_path),
        store=HybridChunkStore.from_settings(settings),
        chunk_embedding_service=_build_chunk_embedding_service(settings),
        knowledge_file_repository=knowledge_file_repository,
    )


def get_chat_service(
    settings: Annotated[Settings, Depends(get_settings)],
    chunk_search_service: Annotated[ChunkSearchService, Depends(get_chunk_search_service)],
) -> ChatService:
    """构造聊天服务。"""
    return ChatService(
        chat_client=AlibabaModelStudioClient.from_settings(settings),
        chunk_search_service=chunk_search_service,
        system_prompt=settings.chat_system_prompt,
    )


def get_password_hasher() -> PasswordHasherAdapter:
    """构造密码哈希器。"""
    return PasswordHasherAdapter.from_default()


def get_jwt_token_manager(
    settings: Annotated[Settings, Depends(get_settings)],
) -> JwtTokenManager:
    """构造 JWT 管理器。"""
    return JwtTokenManager.from_settings(settings)


def get_registration_code_manager(
    settings: Annotated[Settings, Depends(get_settings)],
) -> RegistrationCodeManager:
    """构造注册验证码管理器。"""
    return RegistrationCodeManager.from_settings(settings)


def get_registration_email_sender(
    settings: Annotated[Settings, Depends(get_settings)],
) -> SmtpRegistrationEmailSender:
    """构造注册验证码邮件发送器。"""
    return SmtpRegistrationEmailSender.from_settings(settings)


def get_auth_service(
    user_repository: Annotated[UserRepository, Depends(get_user_repository)],
    password_hasher: Annotated[PasswordHasherAdapter, Depends(get_password_hasher)],
    token_manager: Annotated[JwtTokenManager, Depends(get_jwt_token_manager)],
    registration_code_repository: Annotated[
        RegistrationVerificationRepository,
        Depends(get_registration_verification_repository),
    ],
    registration_email_sender: Annotated[
        SmtpRegistrationEmailSender,
        Depends(get_registration_email_sender),
    ],
    registration_code_manager: Annotated[
        RegistrationCodeManager,
        Depends(get_registration_code_manager),
    ],
) -> AuthService:
    """构造认证服务。"""
    return AuthService(
        user_repository=user_repository,
        password_hasher=password_hasher,
        token_manager=token_manager,
        registration_code_repository=registration_code_repository,
        registration_email_sender=registration_email_sender,
        registration_code_manager=registration_code_manager,
    )


def get_user_admin_service(
    user_repository: Annotated[UserRepository, Depends(get_user_repository)],
    password_hasher: Annotated[PasswordHasherAdapter, Depends(get_password_hasher)],
) -> UserAdminService:
    """构造管理员用户服务。"""
    return UserAdminService(
        user_repository=user_repository,
        password_hasher=password_hasher,
    )


def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> CurrentUser:
    """解析 Bearer Token 并返回当前用户。"""
    if credentials is None or not credentials.credentials.strip():
        raise AuthenticationRequiredError()

    current_user = auth_service.get_current_user_from_token(credentials.credentials)
    request.state.current_user = current_user
    return current_user


def require_admin(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> CurrentUser:
    """要求当前用户必须具备管理员角色。"""
    if current_user.role is not UserRole.ADMIN:
        raise PermissionDeniedError()
    return current_user
