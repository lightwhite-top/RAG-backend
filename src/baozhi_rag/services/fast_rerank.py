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


def _normalize_compact_text(text: str) -> str:
    """把文本规整为去空白的小写形式，便于匹配 OCR 中不稳定的间隔。"""
    return re.sub(r"\s+", "", text).lower()


_NUMERIC_UNIT_PATTERN = re.compile(
    r"\d+(?:\.\d+)?(?:\s*(?:个工作日|工作日|小时|分钟|秒|天|周|月|年|元|万元|次|位|页|条|%))?"
)
_IDENTIFIER_PATTERN = re.compile(r"(?:样例|案例|版本|v)\s*\d+", re.IGNORECASE)


def _extract_query_terms(text: str) -> list[str]:
    """从查询文本中提取简单关键词，并保留数值与编号类精确信号。"""
    normalized_text = _normalize_text(text)
    terms = [
        term
        for term in re.split(r"[\s,.;:!?/\\|()\[\]{}<>，。；：！？、]+", normalized_text)
        if len(term) >= 2 or term.isdigit()
    ]

    for pattern in (_IDENTIFIER_PATTERN, _NUMERIC_UNIT_PATTERN):
        for matched_text in pattern.findall(normalized_text):
            cleaned_text = matched_text.strip()
            if cleaned_text:
                terms.append(cleaned_text)
                compact_text = _normalize_compact_text(cleaned_text)
                if compact_text and compact_text != cleaned_text:
                    terms.append(compact_text)

    deduplicated_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in terms:
        normalized_term = term.strip()
        if not normalized_term or normalized_term in seen_terms:
            continue
        seen_terms.add(normalized_term)
        deduplicated_terms.append(normalized_term)
    return deduplicated_terms


def _is_precise_term(term: str) -> bool:
    """判断一个查询词是否属于数字、编号或量词类精确信号。"""
    compact_term = _normalize_compact_text(term)
    return any(character.isdigit() for character in compact_term)


def _compute_filename_overlap_ratio(query_terms: list[str], filename_title_text: str) -> float:
    """计算查询词与文件名标题的覆盖率。"""
    candidate_terms = [
        term for term in query_terms if len(term.strip()) >= 2 and not _is_precise_term(term)
    ]
    if not candidate_terms or not filename_title_text:
        return 0.0
    matched_count = sum(1 for term in candidate_terms if term in filename_title_text)
    return matched_count / len(candidate_terms)


def _compute_ngram_overlap_ratio(query_text: str, filename_title_text: str) -> float:
    """用紧凑字符 bigram 估算中文问句和文件名的重叠度。"""
    compact_query = _normalize_compact_text(query_text)
    compact_filename = _normalize_compact_text(filename_title_text)
    if len(compact_query) < 2 or len(compact_filename) < 2:
        return 0.0
    query_ngrams = {
        compact_query[index : index + 2]
        for index in range(len(compact_query) - 1)
        if compact_query[index : index + 2].strip()
    }
    if not query_ngrams:
        return 0.0
    matched_count = sum(1 for ngram in query_ngrams if ngram in compact_filename)
    return matched_count / len(query_ngrams)


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
                    score=round(self._score_hit(hit, query_text, query_terms, query_intent), 6),
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
        query_text: str,
        query_terms: list[str],
        query_intent: str,
    ) -> float:
        """计算启发式快速重排分数。"""
        base_score = hit.score or 0.0
        content_text = _normalize_text(hit.content)
        heading_text = _normalize_text(" ".join(hit.heading_path))
        section_title_text = _normalize_text(hit.section_title or "")
        compact_content_text = _normalize_compact_text(hit.content)
        compact_heading_text = _normalize_compact_text(" ".join(hit.heading_path))
        compact_section_title_text = _normalize_compact_text(hit.section_title or "")
        normalized_query_text = _normalize_text(query_text)
        compact_query_text = _normalize_compact_text(query_text)
        filename_title_text = _normalize_text(hit.source_filename_text or hit.source_filename)
        compact_filename_title_text = _normalize_compact_text(
            hit.source_filename_text or hit.source_filename
        )

        keyword_hits = sum(
            1 for term in query_terms if not _is_precise_term(term) and term in content_text
        )
        precise_keyword_hits = sum(
            1
            for term in query_terms
            if _is_precise_term(term)
            and (_normalize_compact_text(term) in compact_content_text or term in content_text)
        )
        heading_hits = sum(
            1
            for term in query_terms
            if not _is_precise_term(term) and (term in heading_text or term in section_title_text)
        )
        precise_heading_hits = sum(
            1
            for term in query_terms
            if _is_precise_term(term)
            and (
                _normalize_compact_text(term) in compact_heading_text
                or _normalize_compact_text(term) in compact_section_title_text
                or term in heading_text
                or term in section_title_text
            )
        )
        filename_hits = sum(
            1 for term in query_terms if not _is_precise_term(term) and term in filename_title_text
        )
        precise_filename_hits = sum(
            1
            for term in query_terms
            if _is_precise_term(term)
            and (
                _normalize_compact_text(term) in compact_filename_title_text
                or term in filename_title_text
            )
        )
        score = (
            base_score
            + keyword_hits * 0.03
            + precise_keyword_hits * 0.08
            + heading_hits * 0.09
            + precise_heading_hits * 0.2
            + filename_hits * 0.15
            + precise_filename_hits * 0.18
        )

        if len(normalized_query_text) >= 6 and normalized_query_text in content_text:
            score += 0.12
        if len(compact_query_text) >= 6 and compact_query_text in compact_content_text:
            score += 0.14
        if len(normalized_query_text) >= 6 and (
            normalized_query_text in heading_text or normalized_query_text in section_title_text
        ):
            score += 0.12
        if len(compact_query_text) >= 6 and (
            compact_query_text in compact_heading_text
            or compact_query_text in compact_section_title_text
        ):
            score += 0.14
        if len(normalized_query_text) >= 4 and normalized_query_text in filename_title_text:
            score += 0.2
        if len(compact_query_text) >= 4 and compact_query_text in compact_filename_title_text:
            score += 0.24

        filename_overlap_ratio = _compute_filename_overlap_ratio(query_terms, filename_title_text)
        if filename_overlap_ratio >= 0.5:
            score += 0.2
        ngram_overlap_ratio = _compute_ngram_overlap_ratio(query_text, filename_title_text)
        if ngram_overlap_ratio >= 0.25:
            score += 0.18

        if query_intent == "structured" and hit.content_type == "table":
            score += 0.08
        if query_intent == "document_location" and heading_hits:
            score += 0.08
        if query_intent == "procedure" and heading_hits:
            score += 0.05

        return score
