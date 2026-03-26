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
        """初始化文件上传服务。

        参数:
            file_store: 本地文件存储适配器，负责底层落盘与删除。

        返回:
            None。
        """
        self._file_store = file_store

    def upload_files(self, files: list[FileUploadInput]) -> list[UploadedFileResult]:
        """批量上传文件并在失败时回滚已写入内容。

        参数:
            files: 待上传文件列表，每项包含文件名、内容类型与二进制流。

        返回:
            与输入顺序一致的上传结果列表。

        异常:
            FileUploadError: 当文件名校验失败或文件保存失败时抛出。
            FileStorageError: 当底层文件系统写入失败时抛出。
        """
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
        """处理单个文件上传。

        参数:
            file_input: 单个文件的上传输入对象。

        返回:
            单个文件的上传结果，包含文件标识、大小和存储路径。

        异常:
            InvalidUploadFileError: 当文件名为空或非法时抛出。
            FileStorageError: 当文件保存失败时抛出。
        """
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
        """在批量上传失败时回滚已保存文件。

        参数:
            storage_keys: 已成功写入磁盘的相对存储路径列表。

        返回:
            None。若部分回滚失败，仅记录错误日志而不再次抛出异常。
        """
        rollback_errors: list[str] = []
        for storage_key in reversed(storage_keys):
            try:
                self._file_store.delete(storage_key)
            except OSError:
                rollback_errors.append(storage_key)

        if rollback_errors:
            LOGGER.error("file_upload_rollback_failed storage_keys=%s", rollback_errors)

    def _validate_filename(self, filename: str) -> str:
        """校验原始文件名是否可用。

        参数:
            filename: 上传请求中携带的原始文件名。

        返回:
            去除路径和首尾空白后的安全原始文件名。

        异常:
            InvalidUploadFileError: 当文件名为空、仅包含空白或为非法路径片段时抛出。
        """
        if not filename or not filename.strip():
            msg = "上传文件名不能为空"
            raise InvalidUploadFileError(msg)

        normalized = Path(filename).name.strip()
        if normalized in {"", ".", ".."}:
            msg = "上传文件名非法"
            raise InvalidUploadFileError(msg)

        return normalized

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名中的危险字符，保留可读性。

        参数:
            filename: 通过基础校验后的原始文件名。

        返回:
            仅保留字母数字、点、下划线和连字符的安全文件名。
        """
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
        """生成日期分层的相对存储路径。

        参数:
            uploaded_at: 文件上传完成时间，用于生成日期目录。
            file_id: 文件唯一标识。
            filename: 已安全化处理后的文件名。

        返回:
            形如 `YYYY/MM/DD/{file_id}_{filename}` 的相对路径。
        """
        date_path = uploaded_at.strftime("%Y/%m/%d")
        return f"{date_path}/{file_id}_{filename}"

    def _measure_size(self, file_obj: BinaryIO) -> int:
        """计算文件流大小。

        参数:
            file_obj: 支持 `tell` 与 `seek` 的二进制文件对象。

        返回:
            文件流总字节数，并在结束前恢复原始游标位置。
        """
        current_position = file_obj.tell()
        file_obj.seek(0, 2)
        size = file_obj.tell()
        file_obj.seek(current_position)
        return size
