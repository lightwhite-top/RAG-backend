"""上传预览服务 ES 入库编排测试。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from docx import Document

from baozhi_rag.infra.storage.local_file_store import LocalFileStore
from baozhi_rag.services.chunk_search import ChunkSearchHit, ChunkSearchRequest
from baozhi_rag.services.document_chunking import DocumentChunkService
from baozhi_rag.services.document_preview import DocumentPreviewService
from baozhi_rag.services.file_upload import FileUploadInput, FileUploadService


class RecordingChunkStore:
    """记录入库调用的测试替身。"""

    def __init__(self) -> None:
        self.ensure_calls = 0
        self.indexed_chunk_batches: list[list[Any]] = []
        self.deleted_file_ids: list[str] = []

    def ensure_index(self) -> None:
        self.ensure_calls += 1

    def index_chunks(self, chunks: list[Any]) -> int:
        self.indexed_chunk_batches.append(chunks)
        return len(chunks)

    def delete_chunks_by_file_id(self, file_id: str) -> None:
        self.deleted_file_ids.append(file_id)

    def search(self, request: ChunkSearchRequest) -> list[ChunkSearchHit]:
        return []


def test_upload_and_preview_files_always_indexes_chunks(tmp_path: Path) -> None:
    """上传切块后应始终执行 ES 入库。"""
    file_store = LocalFileStore(tmp_path)
    chunk_store = RecordingChunkStore()
    service = DocumentPreviewService(
        file_upload_service=FileUploadService(file_store),
        chunk_service=DocumentChunkService(
            chunk_size=200,
            chunk_overlap=20,
            convert_temp_dir=tmp_path / "tmp",
        ),
        file_store=file_store,
        chunk_store=chunk_store,
    )

    results = service.upload_and_chunk_files(
        [
            FileUploadInput(
                filename="产品说明.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                stream=BytesIO(_build_docx_bytes("保险责任免除和免赔额说明。")),
            )
        ]
    )

    assert len(results) == 1
    assert chunk_store.ensure_calls == 1
    assert len(chunk_store.indexed_chunk_batches) == 1
    indexed_chunks = chunk_store.indexed_chunk_batches[0]
    assert len(indexed_chunks) == 1
    chunk = indexed_chunks[0]
    assert chunk.merged_terms == ["保险责任", "免赔额", "保险", "责任免除"]
    assert chunk_store.deleted_file_ids == []


def _build_docx_bytes(text: str) -> bytes:
    """构造单段落 docx 二进制。"""
    document = Document()
    document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
