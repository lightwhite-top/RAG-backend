"""chunk 检索接口模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkSearchHitItem(BaseModel):
    """单个 chunk 检索命中结果。"""

    chunk_id: str = Field(description="chunk 唯一标识")
    file_id: str = Field(description="文件唯一标识")
    source_filename: str = Field(description="原始文件名")
    storage_key: str = Field(description="相对存储路径")
    chunk_index: int = Field(description="chunk 序号")
    char_count: int = Field(description="chunk 字符数")
    content: str = Field(description="chunk 正文")
    merged_terms: list[str] = Field(description="合并去重后的领域词项")
    score: float | None = Field(description="检索得分")


class ChunkSearchResponse(BaseModel):
    """chunk 检索响应。"""

    query: str = Field(description="原始查询文本")
    size: int = Field(description="命中数量")
    hits: list[ChunkSearchHitItem] = Field(description="命中结果列表")
