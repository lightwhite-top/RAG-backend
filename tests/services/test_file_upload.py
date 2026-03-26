"""文件上传服务测试。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pytest

from baozhi_rag.infra.storage.local_file_store import LocalFileStore
from baozhi_rag.services.file_upload import (
    FileStorageError,
    FileUploadInput,
    FileUploadService,
)


def test_upload_service_sanitizes_storage_key_and_saves_file(tmp_path: Path) -> None:
    """服务应生成安全存储路径并保存文件。"""
    service = FileUploadService(LocalFileStore(tmp_path))

    result = service.upload_files(
        [
            FileUploadInput(
                filename="理赔 资料?.pdf",
                content_type="application/pdf",
                stream=BytesIO(b"claim-file"),
            )
        ]
    )[0]

    assert result.original_filename == "理赔 资料?.pdf"
    assert result.content_type == "application/pdf"
    assert result.size == len(b"claim-file")
    assert " " not in result.storage_key
    assert "?" not in result.storage_key
    assert (tmp_path / result.storage_key).exists()


def test_upload_service_rolls_back_on_partial_failure(tmp_path: Path) -> None:
    """批量上传中途失败时应删除已保存文件。"""

    class FailingLocalFileStore(LocalFileStore):
        """第二次保存时模拟底层写入失败。"""

        def __init__(self, root_dir: Path) -> None:
            super().__init__(root_dir)
            self._save_count = 0

        def save(self, file_obj: BinaryIO, storage_key: str) -> None:
            self._save_count += 1
            if self._save_count == 2:
                msg = "disk full"
                raise OSError(msg)
            super().save(file_obj, storage_key)

    service = FileUploadService(FailingLocalFileStore(tmp_path))

    with pytest.raises(FileStorageError):
        service.upload_files(
            [
                FileUploadInput(
                    filename="first.txt",
                    content_type="text/plain",
                    stream=BytesIO(b"first"),
                ),
                FileUploadInput(
                    filename="second.txt",
                    content_type="text/plain",
                    stream=BytesIO(b"second"),
                ),
            ]
        )

    assert list(tmp_path.rglob("*.*")) == []
