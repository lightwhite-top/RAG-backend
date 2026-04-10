"""知识文件图片资产领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class KnowledgeFileImageAsset:
    """文档图片资产领域实体。"""

    id: str
    file_id: str
    segment_id: str
    semantic_chunk_id: str
    asset_index: int
    uploader_user_id: str
    source_anchor: str
    image_sha256: str
    normalized_image_sha256: str
    storage_key: str
    thumbnail_storage_key: str | None
    content_type: str
    width: int | None
    height: int | None
    ocr_text: str
    summary: str
    image_type: str
    recognition_model: str
    created_at: datetime
    updated_at: datetime
