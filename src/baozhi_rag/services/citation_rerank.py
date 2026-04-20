"""答案完成后对引用候选做二次重排。"""

from __future__ import annotations

import re
from dataclasses import replace

from baozhi_rag.services.chunk_search import ChunkSearchHit
from baozhi_rag.services.query_intent import QueryIntentService

_NUMERIC_TERM_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:个工作日|工作日|小时|分钟|秒|天|周|月|年|元|万元|次|位|页|条|%)"
)
_TEXT_TERM_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{2,16}")
_STOP_TERMS = {
    "当前",
    "支持",
    "需要",
    "可以",
    "不能",
    "没有",
    "根据",
    "以及",
    "进行",
    "系统",
    "文档",
    "知识库",
    "用户",
    "管理员",
}


class CitationRerankService:
    """根据答案文本和查询意图重排引用候选。"""

    def __init__(self, query_intent_service: QueryIntentService | None = None) -> None:
        self._query_intent_service = query_intent_service or QueryIntentService()

    def rerank_citations(
        self,
        *,
        answer_text: str,
        query_text: str,
        candidate_chunks: list[ChunkSearchHit],
    ) -> list[ChunkSearchHit]:
        """对引用候选做二次重排。"""
        if len(candidate_chunks) <= 1:
            return candidate_chunks

        target_document_types = self._query_intent_service.detect_document_types(query_text)
        numeric_terms = self._extract_numeric_terms(answer_text)
        named_terms = self._extract_named_terms(answer_text)

        rescored_hits = [
            replace(
                hit,
                score=round(
                    self._score_hit(
                        hit=hit,
                        answer_text=answer_text,
                        numeric_terms=numeric_terms,
                        named_terms=named_terms,
                        target_document_types=target_document_types,
                    ),
                    6,
                ),
            )
            for hit in candidate_chunks
        ]
        return sorted(
            rescored_hits,
            key=lambda item: (item.score or 0.0, -item.chunk_index),
            reverse=True,
        )

    def _score_hit(
        self,
        *,
        hit: ChunkSearchHit,
        answer_text: str,
        numeric_terms: list[str],
        named_terms: list[str],
        target_document_types: list[str],
    ) -> float:
        """计算单条引用候选的证据充分度分数。"""
        compact_searchable = self._compact(hit.searchable_text or hit.content)
        compact_answer = self._compact(answer_text)
        numeric_hits = sum(
            1
            for term in numeric_terms
            if self._compact(term) in compact_searchable
        )
        named_hits = sum(
            1
            for term in named_terms
            if self._compact(term) in compact_searchable
        )
        document_type_match = 1.0 if not target_document_types or hit.document_type in target_document_types else 0.0
        filename_match = 0.0
        filename_title = self._compact(hit.source_filename_text or hit.source_filename)
        if filename_title and filename_title in compact_answer:
            filename_match = 0.5
        elif any(self._compact(term) in filename_title for term in named_terms):
            filename_match = 0.3

        return (
            (hit.score or 0.0)
            + 0.4 * numeric_hits
            + 0.3 * named_hits
            + 0.2 * document_type_match
            + 0.1 * filename_match
        )

    def _extract_numeric_terms(self, answer_text: str) -> list[str]:
        """抽取答案中的数字量词短语。"""
        return list(dict.fromkeys(match.strip() for match in _NUMERIC_TERM_PATTERN.findall(answer_text)))

    def _extract_named_terms(self, answer_text: str) -> list[str]:
        """抽取答案中的命名短语。"""
        terms: list[str] = []
        for match in _TEXT_TERM_PATTERN.findall(answer_text):
            cleaned_term = match.strip()
            if cleaned_term in _STOP_TERMS or len(cleaned_term) < 2:
                continue
            if cleaned_term.isdigit():
                continue
            terms.append(cleaned_term)
        return list(dict.fromkeys(terms))

    def _compact(self, text: str) -> str:
        """把文本压缩为去空白的小写形式。"""
        return re.sub(r"\s+", "", text).lower()
