"""本地文件系统存储适配。"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO


class LocalFileStore:
    """负责将文件流保存到本地目录。"""

    def __init__(self, root_dir: Path) -> None:
        """初始化本地文件存储根目录。"""
        self._root_dir = root_dir.resolve()

    def save(self, file_obj: BinaryIO, storage_key: str) -> None:
        """将文件流保存到指定相对路径。"""
        destination = self._resolve_storage_path(storage_key)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as target:
            while chunk := file_obj.read(1024 * 1024):
                target.write(chunk)

    def delete(self, storage_key: str) -> None:
        """删除指定的已保存文件，用于失败回滚。"""
        destination = self._resolve_storage_path(storage_key)
        if destination.exists():
            destination.unlink()

    def exists(self, storage_key: str) -> bool:
        """判断指定相对路径文件是否存在。"""
        return self._resolve_storage_path(storage_key).exists()

    def resolve_path(self, storage_key: str) -> Path:
        """返回相对路径对应的绝对文件路径。"""
        return self._resolve_storage_path(storage_key)

    def _resolve_storage_path(self, storage_key: str) -> Path:
        """解析并校验相对存储路径，防止越界写入。"""
        destination = (self._root_dir / storage_key).resolve()
        if not destination.is_relative_to(self._root_dir):
            msg = f"非法存储路径: {storage_key}"
            raise ValueError(msg)
        return destination
