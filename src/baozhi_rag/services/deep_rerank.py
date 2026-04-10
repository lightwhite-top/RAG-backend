"""深度重排服务。"""

from __future__ import annotations

from baozhi_rag.services.chunk_search import ChunkSearchHit
from baozhi_rag.services.rerank import ChunkRerankService


class DeepRerankService:
    """按阈值条件触发的大模型深度重排。"""

    def __init__(
        self,
        *,
        client: ChunkRerankService,
        score_threshold: float,
        margin_threshold: float,
    ) -> None:
        self._client = client
        self._score_threshold = score_threshold
        self._margin_threshold = margin_threshold

    def rerank(
        self,
        *,
        query_text: str,
        hits: list[ChunkSearchHit],
    ) -> tuple[list[ChunkSearchHit], bool]:
        """满足触发条件时执行深度重排。"""
        if not self.should_trigger(hits):
            return hits, False
        return self._client.rerank(query_text=query_text, hits=hits), True

    def should_trigger(self, hits: list[ChunkSearchHit]) -> bool:
        """判断是否应触发深度重排。"""
        if len(hits) <= 1:
            return False

        top_score = hits[0].score
        second_score = hits[1].score
        if top_score is None or second_score is None:
            return True
        if top_score < self._score_threshold:
            return True
        return abs(top_score - second_score) < self._margin_threshold
