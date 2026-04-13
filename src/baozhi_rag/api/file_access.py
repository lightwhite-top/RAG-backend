"""知识文件同源访问地址辅助函数。"""

from __future__ import annotations

from fastapi import Request


def build_backend_file_access_payload(
    request: Request,
    *,
    file_id: str,
) -> dict[str, str | None]:
    """构造后端同源文件访问地址。

    参数:
        request: 当前 HTTP 请求，用于基于当前后端地址生成绝对 URL。
        file_id: 知识文件 ID；为空时返回空载荷，便于调用方兜底到其他方案。
    返回:
        包含同源 `url` 与 `expires_at` 的字典；同源地址不需要过期时间。
    """
    normalized_file_id = file_id.strip()
    if not normalized_file_id:
        return {"url": None, "expires_at": None}

    return {
        "url": str(request.url_for("get_file_content", file_id=normalized_file_id)),
        "expires_at": None,
    }


def build_backend_image_asset_access_payload(
    request: Request,
    *,
    asset_id: str,
) -> dict[str, str | None]:
    """构造后端同源图片原图访问地址。"""
    normalized_asset_id = asset_id.strip()
    if not normalized_asset_id:
        return {"url": None, "expires_at": None}

    return {
        "url": str(request.url_for("get_image_asset_content", asset_id=normalized_asset_id)),
        "expires_at": None,
    }


def build_backend_image_asset_preview_payload(
    request: Request,
    *,
    asset_id: str,
) -> dict[str, str | None]:
    """构造后端同源图片缩略图访问地址。"""
    normalized_asset_id = asset_id.strip()
    if not normalized_asset_id:
        return {"url": None, "expires_at": None}

    return {
        "url": str(request.url_for("get_image_asset_preview", asset_id=normalized_asset_id)),
        "expires_at": None,
    }
