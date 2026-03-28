"""文件上传接口模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UploadedFileItem(BaseModel):
    """单个上传文件结果。"""

    file_id: str = Field(description="文件唯一标识")
    original_filename: str = Field(description="原始文件名")
    content_type: str = Field(description="文件内容类型")
    size: int = Field(description="文件字节数")
    storage_key: str = Field(description="相对存储路径")
    chunk_count: int = Field(description="切块数量")
    uploaded_at: datetime = Field(description="上传完成时间")


class FileUploadResponseData(BaseModel):
    """文件上传成功后的业务数据。"""

    file_count: int = Field(description="本次成功处理的文件数")
    files: list[UploadedFileItem] = Field(description="本次上传文件结果列表")
