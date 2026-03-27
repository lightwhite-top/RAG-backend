"""通用接口响应模型。"""

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """仅返回提示消息的通用响应。"""

    message: str = Field(description="接口处理结果提示")
