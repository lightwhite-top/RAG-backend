"""快速重排服务。"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baozhi_rag.services.chunk_search import ChunkSearchHit


def _normalize_text(text: str) -> str:
    """把文本规整为便于关键词匹配的单行形式。"""
    return " ".join(text.split()).lower()


def _extract_query_terms(text: str) -> list[str]:
    """从查询文本中提取简单关键词。"""
    normalized_text = _normalize_text(text)
    return [
        term
        for term in re.split(r"[\s,.;:!?/\\|()\[\]{}<>，。；：！？、]+", normalized_text)
        if len(term) >= 2
    ]


class FastRerankService:
    """使用轻量启发式规则对候选做快速重排。"""

    def rerank(
        self,
        *,
        query_text: str,
        query_intent: str,
        hits: list[ChunkSearchHit],
    ) -> list[ChunkSearchHit]:
        """按查询文本和查询意图快速重排候选。"""
        if len(hits) <= 1:
            return hits

        query_terms = _extract_query_terms(query_text)
        rescored_hits: list[ChunkSearchHit] = []
        for hit in hits:
            rescored_hits.append(
                replace(
                    hit,
                    score=round(self._score_hit(hit, query_terms, query_intent), 6),
                )
            )

        return sorted(
            rescored_hits,
            key=lambda item: (item.score or 0.0, -item.chunk_index),
            reverse=True,
        )

    def _score_hit(
        self,
        hit: ChunkSearchHit,
        query_terms: list[str],
        query_intent: str,
    ) -> float:
        """计算启发式快速重排分数。"""
        base_score = hit.score or 0.0
        content_text = _normalize_text(hit.content)
        heading_text = _normalize_text(" ".join(hit.heading_path))
        section_title_text = _normalize_text(hit.section_title or "")

        keyword_hits = sum(1 for term in query_terms if term in content_text)
        heading_hits = sum(
            1 for term in query_terms if term in heading_text or term in section_title_text
        )
        score = base_score + keyword_hits * 0.03 + heading_hits * 0.05

        if query_intent == "structured" and hit.content_type == "table":
            score += 0.08
        if query_intent == "document_location" and heading_hits:
            score += 0.08
        if query_intent == "procedure" and heading_hits:
            score += 0.05

        return score
