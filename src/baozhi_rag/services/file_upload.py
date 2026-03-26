"""文件上传服务。"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from baozhi_rag.infra.storage.local_file_store import LocalFileStore

LOGGER = logging.getLogger(__name__)


class FileUploadError(Exception):
    """文件上传失败基类。"""


class InvalidUploadFileError(FileUploadError):
    """上传文件参数非法。"""


class FileStorageError(FileUploadError):
    """文件保存或回滚失败。"""


@dataclass(frozen=True, slots=True)
class FileUploadInput:
    """服务层使用的上传文件输入。"""

    filename: str
    content_type: str | None
    stream: BinaryIO


@dataclass(frozen=True, slots=True)
class UploadedFileResult:
    """服务层返回的文件上传结果。"""

    file_id: str
    original_filename: str
    content_type: str
    size: int
    storage_key: str
    uploaded_at: datetime


class FileUploadService:
    """编排文件元数据生成与本地存储写入。"""

    def __init__(self, file_store: LocalFileStore) -> None:
        """初始化文件上传服务。"""
        self._file_store = file_store

    def upload_files(self, files: list[FileUploadInput]) -> list[UploadedFileResult]:
        """批量上传文件并在失败时回滚已写入内容。"""
        stored_keys: list[str] = []
        results: list[UploadedFileResult] = []

        try:
            for file_input in files:
                result = self._upload_single_file(file_input)
                stored_keys.append(result.storage_key)
                results.append(result)
        except FileUploadError:
            self._rollback(stored_keys)
            raise
        except OSError as exc:
            self._rollback(stored_keys)
            msg = "文件保存失败"
            raise FileStorageError(msg) from exc

        return results

    def _upload_single_file(self, file_input: FileUploadInput) -> UploadedFileResult:
        """处理单个文件上传。"""
        original_filename = self._validate_filename(file_input.filename)
        safe_filename = self._sanitize_filename(original_filename)
        file_id = str(uuid4())
        uploaded_at = datetime.now(UTC)
        storage_key = self._build_storage_key(uploaded_at, file_id, safe_filename)

        try:
            file_input.stream.seek(0)
            size = self._measure_size(file_input.stream)
            file_input.stream.seek(0)
            self._file_store.save(file_input.stream, storage_key)
        except OSError as exc:
            msg = f"保存文件失败: {original_filename}"
            raise FileStorageError(msg) from exc

        content_type = file_input.content_type or "application/octet-stream"
        result = UploadedFileResult(
            file_id=file_id,
            original_filename=original_filename,
            content_type=content_type,
            size=size,
            storage_key=storage_key,
            uploaded_at=uploaded_at,
        )
        LOGGER.info(
            "file_upload_success file_id=%s filename=%s size=%s storage_key=%s",
            result.file_id,
            result.original_filename,
            result.size,
            result.storage_key,
        )
        return result

    def _rollback(self, storage_keys: list[str]) -> None:
        """在批量上传失败时回滚已保存文件。"""
        rollback_errors: list[str] = []
        for storage_key in reversed(storage_keys):
            try:
                self._file_store.delete(storage_key)
            except OSError:
                rollback_errors.append(storage_key)

        if rollback_errors:
            LOGGER.error("file_upload_rollback_failed storage_keys=%s", rollback_errors)

    def _validate_filename(self, filename: str) -> str:
        """校验原始文件名是否可用。"""
        if not filename or not filename.strip():
            msg = "上传文件名不能为空"
            raise InvalidUploadFileError(msg)

        normalized = Path(filename).name.strip()
        if normalized in {"", ".", ".."}:
            msg = "上传文件名非法"
            raise InvalidUploadFileError(msg)

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

    def _build_storage_key(self, uploaded_at: datetime, file_id: str, filename: str) -> str:
        """生成日期分层的相对存储路径。"""
        date_path = uploaded_at.strftime("%Y/%m/%d")
        return f"{date_path}/{file_id}_{filename}"

    def _measure_size(self, file_obj: BinaryIO) -> int:
        """计算文件流大小。"""
        current_position = file_obj.tell()
        file_obj.seek(0, 2)
        size = file_obj.tell()
        file_obj.seek(current_position)
        return size
