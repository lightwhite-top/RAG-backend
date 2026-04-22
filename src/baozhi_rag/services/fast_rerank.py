"""快速重排服务。"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from baozhi_rag.services.retrieval_signals import (
    extract_query_terms as extract_signal_query_terms,
)
from baozhi_rag.services.retrieval_signals import (
    extract_version_rank,
    normalize_filename_title,
    normalize_text,
    query_prefers_old_version,
)

if TYPE_CHECKING:
    from baozhi_rag.services.chunk_search import ChunkSearchHit


def _extract_query_terms(text: str) -> list[str]:
    """从查询文本中提取更稳定的排序信号。"""
    return extract_signal_query_terms(text)


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
        normalized_query_text = normalize_text(query_text)
        prefers_old_version = query_prefers_old_version(query_text)
        rescored_hits: list[ChunkSearchHit] = []
        for hit in hits:
            rescored_hits.append(
                replace(
                    hit,
                    score=round(
                        self._score_hit(
                            hit,
                            query_terms=query_terms,
                            query_intent=query_intent,
                            normalized_query_text=normalized_query_text,
                            prefers_old_version=prefers_old_version,
                        ),
                        6,
                    ),
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
        *,
        query_terms: list[str],
        query_intent: str,
        normalized_query_text: str,
        prefers_old_version: bool,
    ) -> float:
        """计算启发式快速重排分数。"""
        base_score = hit.score or 0.0
        content_text = normalize_text(hit.content)
        heading_text = normalize_text(" ".join(hit.heading_path))
        section_title_text = normalize_text(hit.section_title or "")
        source_filename_text = normalize_text(normalize_filename_title(hit.source_filename))

        keyword_hits = sum(1 for term in query_terms if normalize_text(term) in content_text)
        heading_hits = sum(
            1
            for term in query_terms
            if normalize_text(term) in heading_text or normalize_text(term) in section_title_text
        )
        filename_hits = sum(
            1 for term in query_terms if normalize_text(term) in source_filename_text
        )
        score = base_score + keyword_hits * 0.03 + heading_hits * 0.05 + filename_hits * 0.09

        if normalized_query_text and normalized_query_text in content_text:
            score += 0.18
        if normalized_query_text and normalized_query_text in source_filename_text:
            score += 0.12

        version_rank = extract_version_rank(hit.source_filename)
        if version_rank > 0:
            if prefers_old_version:
                score += max(0.0, 100 - version_rank) * 0.001
            else:
                score += version_rank * 0.012

        if query_intent == "structured" and hit.content_type == "table":
            score += 0.08
        if query_intent == "document_location" and heading_hits:
            score += 0.08
        if query_intent == "procedure" and heading_hits:
            score += 0.05

        return score
