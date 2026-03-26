"""文件上传路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from baozhi_rag.core.config import Settings, get_settings
from baozhi_rag.infra.storage.local_file_store import LocalFileStore
from baozhi_rag.schemas.files import UploadedFileItem, UploadFilesResponse
from baozhi_rag.services.file_upload import (
    FileStorageError,
    FileUploadInput,
    FileUploadService,
    InvalidUploadFileError,
)

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload", response_model=UploadFilesResponse, summary="上传文件")
async def upload_files(
    files: Annotated[list[UploadFile], File(description="待上传文件列表")],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UploadFilesResponse:
    """接收多个文件并保存到本地文件系统。"""
    service = FileUploadService(LocalFileStore(settings.upload_root_dir))

    try:
        results = service.upload_files(
            [
                FileUploadInput(
                    filename=file.filename or "",
                    content_type=file.content_type,
                    stream=file.file,
                )
                for file in files
            ]
        )
    except InvalidUploadFileError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    finally:
        for file in files:
            await file.close()

    return UploadFilesResponse(
        files=[
            UploadedFileItem(
                file_id=result.file_id,
                original_filename=result.original_filename,
                content_type=result.content_type,
                size=result.size,
                storage_key=result.storage_key,
                uploaded_at=result.uploaded_at,
            )
            for result in results
        ]
    )
