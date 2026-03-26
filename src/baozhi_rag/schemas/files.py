"""文件上传接口模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UploadedFileItem(BaseModel):
    """单个已上传文件的接口响应。"""

    file_id: str = Field(description="文件唯一标识")
    original_filename: str = Field(description="原始文件名")
    content_type: str = Field(description="文件内容类型")
    size: int = Field(description="文件大小，单位为字节")
    storage_key: str = Field(description="相对存储路径")
    uploaded_at: datetime = Field(description="上传完成时间")


class UploadFilesResponse(BaseModel):
    """批量文件上传响应。"""

    files: list[UploadedFileItem] = Field(description="上传成功的文件列表")
