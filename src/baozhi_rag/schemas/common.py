"""通用接口响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SuccessResponse[DataT](BaseModel):
    """通用成功响应模型。"""

    state: Literal["success"] = Field(default="success", description="成功响应固定为 success")
    message: str = Field(description="接口处理结果提示")
    data: DataT | None = Field(default=None, description="成功响应承载的业务数据")
    meta: dict[str, Any] | None = Field(default=None, description="分页、统计等附加元信息")
    request_id: str = Field(description="请求链路编号，便于日志检索与问题排查")

    @classmethod
    def success(
        cls,
        *,
        message: str,
        request_id: str,
        data: DataT | None = None,
        meta: dict[str, Any] | None = None,
    ) -> SuccessResponse[DataT]:
        """构造成功响应。

        参数:
            message: 面向调用方的成功提示。
            request_id: 请求链路编号。
            data: 可选业务数据载荷。
            meta: 可选附加元信息，例如分页信息。

        返回:
            标准化的成功响应对象。
        """
        return cls(message=message, data=data, meta=meta, request_id=request_id)


class ValidationErrorItem(BaseModel):
    """请求字段校验明细。"""

    field: str = Field(description="出错字段路径，例如 query.q 或 body.files.0")
    message: str = Field(description="字段错误提示")


class ErrorResponse(BaseModel):
    """统一错误响应模型。"""

    state: Literal["error"] = Field(default="error", description="错误响应固定为 error")
    code: str = Field(description="稳定错误码，便于调用方做程序化处理")
    message: str = Field(description="面向调用方的错误提示")
    request_id: str = Field(description="请求链路编号，便于日志检索与问题排查")
    details: list[ValidationErrorItem] | None = Field(
        default=None,
        description="可选字段级校验明细",
    )
