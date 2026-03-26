"""本地文件系统存储适配。"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO


class LocalFileStore:
    """负责将文件流保存到本地目录。"""

    def __init__(self, root_dir: Path) -> None:
        """初始化本地文件存储根目录。

        参数:
            root_dir: 文件存储根目录；初始化时会被解析为绝对路径。

        返回:
            None。
        """
        self._root_dir = root_dir.resolve()

    def save(self, file_obj: BinaryIO, storage_key: str) -> None:
        """将文件流保存到指定相对路径。

        参数:
            file_obj: 可读取二进制内容的文件对象。
            storage_key: 相对于存储根目录的文件路径。

        返回:
            None。

        异常:
            ValueError: 当 `storage_key` 解析后越过存储根目录时抛出。
            OSError: 当目录创建或文件写入失败时抛出。
        """
        destination = self._resolve_storage_path(storage_key)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as target:
            while chunk := file_obj.read(1024 * 1024):
                target.write(chunk)

    def delete(self, storage_key: str) -> None:
        """删除指定的已保存文件，用于失败回滚。

        参数:
            storage_key: 相对于存储根目录的文件路径。

        返回:
            None。

        异常:
            ValueError: 当 `storage_key` 非法并越过存储根目录时抛出。
            OSError: 当底层删除文件失败时抛出。
        """
        destination = self._resolve_storage_path(storage_key)
        if destination.exists():
            destination.unlink()
            self._cleanup_empty_parent_dirs(destination.parent)

    def exists(self, storage_key: str) -> bool:
        """判断指定相对路径文件是否存在。

        参数:
            storage_key: 相对于存储根目录的文件路径。

        返回:
            若文件存在则返回 True，否则返回 False。

        异常:
            ValueError: 当 `storage_key` 非法并越过存储根目录时抛出。
        """
        return self._resolve_storage_path(storage_key).exists()

    def resolve_path(self, storage_key: str) -> Path:
        """返回相对路径对应的绝对文件路径。

        参数:
            storage_key: 相对于存储根目录的文件路径。

        返回:
            解析后的绝对 Path 对象。

        异常:
            ValueError: 当 `storage_key` 非法并越过存储根目录时抛出。
        """
        return self._resolve_storage_path(storage_key)

    def _resolve_storage_path(self, storage_key: str) -> Path:
        """解析并校验相对存储路径，防止越界写入。

        参数:
            storage_key: 相对于存储根目录的文件路径。

        返回:
            校验通过后的绝对 Path 对象。

        异常:
            ValueError: 当目标路径解析后不在存储根目录下时抛出。
        """
        destination = (self._root_dir / storage_key).resolve()
        if not destination.is_relative_to(self._root_dir):
            msg = f"非法存储路径: {storage_key}"
            raise ValueError(msg)
        return destination

    def _cleanup_empty_parent_dirs(self, current_dir: Path) -> None:
        """删除上传根目录下的空父目录。

        参数:
            current_dir: 从该目录开始向上尝试清理空目录。

        返回:
            None。遇到非空目录或越过根目录时停止。
        """
        while current_dir != self._root_dir and current_dir.is_relative_to(self._root_dir):
            try:
                current_dir.rmdir()
            except OSError:
                break
            current_dir = current_dir.parent
