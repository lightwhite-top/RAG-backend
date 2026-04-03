"""文件上传暂存服务。"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.infra.storage.local_file_store import LocalFileStore

LOGGER = logging.getLogger(__name__)


class FileUploadError(AppError):
    """文件上传失败基类。"""

    default_message = "文件上传失败"
    default_error_code = "file_upload_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class InvalidUploadFileError(FileUploadError):
    """上传文件参数非法。"""

    default_message = "上传文件参数非法"
    default_error_code = "invalid_upload_file"
    default_status_code = status.HTTP_400_BAD_REQUEST


class FileStorageError(FileUploadError):
    """文件保存或回滚失败。"""

    default_message = "文件保存失败"
    default_error_code = "file_storage_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


@dataclass(frozen=True, slots=True)
class FileUploadInput:
    """服务层使用的上传文件输入。"""

    filename: str
    content_type: str | None
    stream: BinaryIO


@dataclass(frozen=True, slots=True)
class StagedUploadFileResult:
    """本地临时暂存文件结果。"""

    stage_id: str
    original_filename: str
    safe_filename: str
    content_type: str
    size: int
    sha256: str
    temp_storage_key: str
    staged_at: datetime


@dataclass(frozen=True, slots=True)
class UploadedFileResult:
    """上传接口返回的最终文件结果。"""

    file_id: str
    original_filename: str
    content_type: str
    size: int
    storage_key: str
    uploaded_at: datetime
    chunk_count: int = 0
    storage_provider: str = "aliyun_oss"
    deduplicated: bool = False
    replaced: bool = False
    title_updated: bool = False


class FileUploadService:
    """负责把上传文件暂存到本地临时目录并计算哈希。"""

    def __init__(self, file_store: LocalFileStore) -> None:
        """初始化文件上传服务。"""
        self._file_store = file_store

    def stage_files(self, files: list[FileUploadInput]) -> list[StagedUploadFileResult]:
        """批量暂存文件并在失败时回滚临时文件。"""
        stored_keys: list[str] = []
        results: list[StagedUploadFileResult] = []

        try:
            for file_input in files:
                result = self._stage_single_file(file_input)
                stored_keys.append(result.temp_storage_key)
                results.append(result)
        except FileUploadError:
            self._rollback(stored_keys)
            raise
        except OSError as exc:
            self._rollback(stored_keys)
            raise FileStorageError("文件保存失败") from exc

        return results

    def upload_files(self, files: list[FileUploadInput]) -> list[StagedUploadFileResult]:
        """兼容旧调用方的暂存入口。"""
        return self.stage_files(files)

    def _stage_single_file(self, file_input: FileUploadInput) -> StagedUploadFileResult:
        """暂存单个上传文件并计算内容哈希。"""
        original_filename = self._validate_filename(file_input.filename)
        safe_filename = self._sanitize_filename(original_filename)
        stage_id = uuid4().hex
        staged_at = datetime.now(UTC)
        temp_storage_key = self._build_temp_storage_key(staged_at, stage_id, safe_filename)
        destination = self._file_store.resolve_path(temp_storage_key)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            file_input.stream.seek(0)
            sha256_hasher = hashlib.sha256()
            size = 0
            with destination.open("wb") as target:
                while chunk := file_input.stream.read(1024 * 1024):
                    sha256_hasher.update(chunk)
                    size += len(chunk)
                    target.write(chunk)
        except OSError as exc:
            raise FileStorageError(f"保存文件失败: {original_filename}") from exc

        result = StagedUploadFileResult(
            stage_id=stage_id,
            original_filename=original_filename,
            safe_filename=safe_filename,
            content_type=file_input.content_type or "application/octet-stream",
            size=size,
            sha256=sha256_hasher.hexdigest(),
            temp_storage_key=temp_storage_key,
            staged_at=staged_at,
        )
        LOGGER.info(
            "file_stage_success filename=%s size=%s sha256=%s temp_storage_key=%s",
            result.original_filename,
            result.size,
            result.sha256,
            result.temp_storage_key,
        )
        return result

    def _rollback(self, storage_keys: list[str]) -> None:
        """在批量暂存失败时回滚已保存文件。"""
        rollback_errors: list[str] = []
        for storage_key in reversed(storage_keys):
            try:
                self._file_store.delete(storage_key)
            except OSError:
                rollback_errors.append(storage_key)

        if rollback_errors:
            LOGGER.error("file_stage_rollback_failed storage_keys=%s", rollback_errors)

    def _validate_filename(self, filename: str) -> str:
        """校验原始文件名是否可用。"""
        if not filename or not filename.strip():
            raise InvalidUploadFileError("上传文件名不能为空")

        normalized = Path(filename).name.strip()
        if normalized in {"", ".", ".."}:
            raise InvalidUploadFileError("上传文件名非法")

        return normalized

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名中的危险字符，保留可读性。"""
        normalized = unicodedata.normalize("NFKC", filename)
        sanitized = "".join(
            character if character.isalnum() or character in {".", "-", "_"} else "_"
            for character in normalized
        )
        sanitized = re.sub(r"_+", "_", sanitized).strip("._")

        if not sanitized:
            return "file"

        return sanitized

    def _build_temp_storage_key(self, staged_at: datetime, stage_id: str, filename: str) -> str:
        """生成本地临时文件路径。"""
        date_path = staged_at.strftime("%Y/%m/%d")
        return f"_tmp/{date_path}/{stage_id}_{filename}"


