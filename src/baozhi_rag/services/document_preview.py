"""上传后文档切块预览编排服务。"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

from baozhi_rag.infra.storage.local_file_store import LocalFileStore
from baozhi_rag.services.chunk_search import ChunkSearchStore
from baozhi_rag.services.document_chunking import DocumentChunk, DocumentChunkService
from baozhi_rag.services.file_upload import FileUploadInput, FileUploadService, UploadedFileResult


@dataclass(frozen=True, slots=True)
class ChunkedFileResult:
    """已上传并完成切块预览的文件结果。"""

    upload: UploadedFileResult
    chunks: list[DocumentChunk]


class DocumentPreviewService:
    """编排文件上传与切块预览。"""

    def __init__(
        self,
        file_upload_service: FileUploadService,
        chunk_service: DocumentChunkService,
        file_store: LocalFileStore,
        chunk_store: ChunkSearchStore | None = None,
    ) -> None:
        """初始化上传预览服务。

        参数:
            file_upload_service: 负责文件落盘的上传服务。
            chunk_service: 负责按文件格式解析并切块的服务。
            file_store: 本地文件存储适配器，用于在失败时执行回滚。
            chunk_store: 可选的检索存储适配器；存在时会在上传后执行 ES 入库。

        返回:
            None。
        """
        self._file_upload_service = file_upload_service
        self._chunk_service = chunk_service
        self._file_store = file_store
        self._chunk_store = chunk_store

    def upload_and_preview_files(self, files: list[FileUploadInput]) -> list[ChunkedFileResult]:
        """上传文件并生成切块预览。

        参数:
            files: 待上传文件列表，每个文件会在落盘后继续执行切块预览。

        返回:
            包含上传结果和切块结果的文件结果列表。

        异常:
            Exception: 透传上传或切块阶段抛出的异常，并在抛出前回滚本次已落盘文件。
        """
        uploaded_files = self._file_upload_service.upload_files(files)
        indexed_file_ids: list[str] = []

        try:
            results = [
                # 依次对每个已上传文件执行切块，并将结果封装在 ChunkedFileResult 中返回
                ChunkedFileResult(
                    upload=uploaded_file,
                    # 通过文件存储适配器解析出文件的绝对路径，并传递给切块服务进行切块预览
                    chunks=self._chunk_service.chunk_document(
                        file_path=self._file_store.resolve_path(uploaded_file.storage_key),
                        source_filename=uploaded_file.original_filename,
                        storage_key=uploaded_file.storage_key,
                        file_id=uploaded_file.file_id,
                    ),
                )
                for uploaded_file in uploaded_files
            ]
            indexed_file_ids = self._index_chunks(results)
            return results
        except Exception:
            self._rollback_indexed_chunks(indexed_file_ids)
            self._rollback_uploaded_files(uploaded_files)
            raise

    def _index_chunks(self, results: list[ChunkedFileResult]) -> list[str]:
        """在启用检索存储时写入 chunk 文档。"""
        if self._chunk_store is None:
            return []

        self._chunk_store.ensure_index()
        indexed_file_ids: list[str] = []
        for result in results:
            self._chunk_store.index_chunks(result.chunks)
            indexed_file_ids.append(result.upload.file_id)
        return indexed_file_ids

    def _rollback_uploaded_files(self, uploaded_files: list[UploadedFileResult]) -> None:
        """切块失败时删除本次请求已落盘文件。

        参数:
            uploaded_files: 本次请求中已经成功写入磁盘的文件结果列表。

        返回:
            None。
        """
        for uploaded_file in reversed(uploaded_files):
            self._file_store.delete(uploaded_file.storage_key)

    def _rollback_indexed_chunks(self, indexed_file_ids: list[str]) -> None:
        """索引失败时删除已写入的 chunk 文档。"""
        if self._chunk_store is None:
            return

        for file_id in reversed(indexed_file_ids):
            with suppress(Exception):
                self._chunk_store.delete_chunks_by_file_id(file_id)
