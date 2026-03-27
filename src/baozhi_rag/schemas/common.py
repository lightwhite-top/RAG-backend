"""通用接口响应模型。"""

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """包含业务状态的通用响应。"""

    state: str = Field(description="业务处理状态，成功为 success，失败为 error")
    message: str = Field(description="接口处理结果提示")

    @classmethod
    def success_response(cls, message: str) -> "MessageResponse":
        """构造成功响应。"""
        return cls(state="success", message=message)

    @classmethod
    def failure_response(cls, message: str) -> "MessageResponse":
        """构造失败响应。"""
        return cls(state="error", message=message)
