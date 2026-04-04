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
    storage_key: str = Field(description="OSS 对象键")
    storage_provider: str = Field(description="文件存储提供商")
    deduplicated: bool = Field(description="本次是否命中重复入库")
    replaced: bool = Field(description="本次是否覆盖了旧版本")
    chunk_count: int = Field(description="切块数量")
    uploaded_at: datetime = Field(description="上传完成时间")


class FileUploadResponseData(BaseModel):
    """文件上传成功后的业务数据。"""

    file_count: int = Field(description="本次成功处理的文件数")
    files: list[UploadedFileItem] = Field(description="本次上传文件结果列表")


class UploadTaskItem(BaseModel):
    """上传任务摘要。"""

    task_id: str = Field(description="上传任务唯一标识")
    status: str = Field(description="任务状态")
    stage: str = Field(description="任务阶段")
    original_filename: str = Field(description="当前任务对应的原始文件名")
    content_type: str = Field(description="文件内容类型")
    size: int = Field(description="文件字节数")
    file_id: str | None = Field(default=None, description="处理完成后的文件标识")
    chunk_count: int = Field(description="处理完成后的切块数量")
    deduplicated: bool = Field(description="本次任务是否命中重复文件")
    replaced: bool = Field(description="本次任务是否替换旧版本")
    title_updated: bool = Field(description="本次任务是否仅更新标题")
    error_code: str | None = Field(default=None, description="失败错误码")
    error_message: str | None = Field(default=None, description="失败错误提示")
    created_at: datetime = Field(description="任务创建时间")
    updated_at: datetime = Field(description="任务最近更新时间")
    completed_at: datetime | None = Field(default=None, description="任务完成时间")


class FileUploadSubmitResponseData(BaseModel):
    """提交上传任务后的业务数据。"""

    file_count: int = Field(description="本次提交的文件数")
    tasks: list[UploadTaskItem] = Field(description="本次创建或复用的上传任务列表")


class UploadTaskListResponseData(BaseModel):
    """上传任务列表业务数据。"""

    task_count: int = Field(description="当前返回的任务条数")
    tasks: list[UploadTaskItem] = Field(description="当前用户最近的上传任务列表")


class KnowledgeFileItem(BaseModel):
    """知识文件列表单项。"""

    file_id: str = Field(description="文件唯一标识")
    uploader_user_id: str = Field(description="上传者用户 ID")
    original_filename: str = Field(description="原始文件名")
    content_type: str = Field(description="文件内容类型")
    size: int = Field(description="文件字节数")
    storage_key: str = Field(description="OSS 对象键")
    file_url: str = Field(description="文件可访问地址")
    storage_provider: str = Field(description="文件存储提供商")
    visibility_scope: str = Field(description="文件可见范围")
    chunk_count: int = Field(description="切块数量")
    uploaded_at: datetime = Field(description="上传完成时间")
    updated_at: datetime = Field(description="最近更新时间")


class KnowledgeFileListResponseData(BaseModel):
    """知识文件列表响应数据。"""

    items: list[KnowledgeFileItem] = Field(description="当前页文件列表")
