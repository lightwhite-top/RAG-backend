"""系统级响应模型。"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(description="服务状态")
    service: str = Field(description="服务名称")
    environment: str = Field(description="运行环境")
    version: str = Field(description="服务版本")


class ServiceInfoResponse(BaseModel):
    """根路径响应。"""

    service: str = Field(description="服务名称")
    environment: str = Field(description="运行环境")
    version: str = Field(description="服务版本")
    docs_url: str = Field(description="文档地址")
