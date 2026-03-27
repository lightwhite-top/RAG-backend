"""chunk 向量化服务。"""

from __future__ import annotations

from dataclasses import replace

from baozhi_rag.services.document_chunking import DocumentChunk
from baozhi_rag.services.llm import EmbeddingModelClient


class ChunkEmbeddingError(Exception):
    """chunk 向量化失败。"""


class ChunkEmbeddingService:
    """编排 chunk 与查询文本的向量化。"""

    def __init__(self, embedding_client: EmbeddingModelClient) -> None:
        """初始化向量化服务。

        参数:
            embedding_client: 负责调用外部向量模型的底层客户端。

        返回:
            None。
        """
        self._embedding_client = embedding_client

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """为 chunk 列表补充向量字段。

        参数:
            chunks: 待向量化的标准化 chunk 列表。

        返回:
            包含 `content_embedding` 字段的新 chunk 列表。

        异常:
            ChunkEmbeddingError: 当外部向量模型返回数量异常时抛出。
        """
        if not chunks:
            return []

        embeddings = self._embedding_client.embed_texts([chunk.content for chunk in chunks])
        if len(embeddings) != len(chunks):
            msg = "向量模型返回数量与 chunk 数量不一致"
            raise ChunkEmbeddingError(msg)

        return [
            replace(chunk, content_embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

    def embed_query(self, query_text: str) -> list[float]:
        """把查询文本转换为语义检索向量。

        参数:
            query_text: 已完成基础清洗的查询文本。

        返回:
            查询文本对应的向量结果。

        异常:
            ChunkEmbeddingError: 当模型未返回单个查询向量时抛出。
        """
        embeddings = self._embedding_client.embed_texts([query_text])
        if len(embeddings) != 1:
            msg = "向量模型未返回单个查询向量"
            raise ChunkEmbeddingError(msg)
        return embeddings[0]
