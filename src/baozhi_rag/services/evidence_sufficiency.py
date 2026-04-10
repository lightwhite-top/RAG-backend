"""证据充分性判断服务。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baozhi_rag.services.chunk_search import ChunkSearchHit


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """证据充分性判断结果。"""

    sufficient: bool
    reason_code: str
    citation_count: int
    top_score: float | None = None


class EvidenceSufficiencyService:
    """基于命中数量与排序得分做轻量证据充分性判断。"""

    _CONFLICT_TOKEN_PAIRS = [
        ("支持", "不支持"),
        ("允许", "禁止"),
        ("启用", "禁用"),
        ("开启", "关闭"),
        ("可以", "不可"),
        ("enabled", "disabled"),
        ("allow", "deny"),
        ("supported", "unsupported"),
    ]

    def __init__(self, *, min_score: float = 0.015, min_citation_count: int = 1) -> None:
        self._min_score = min_score
        self._min_citation_count = max(1, min_citation_count)

    def assess(self, hits: list[ChunkSearchHit]) -> EvidenceAssessment:
        """判断当前证据是否足以支撑回答。"""
        citation_count = len(hits)
        if citation_count < self._min_citation_count:
            return EvidenceAssessment(
                sufficient=False,
                reason_code="no_evidence",
                citation_count=citation_count,
                top_score=None,
            )

        top_score = hits[0].score
        if top_score is not None and top_score < self._min_score:
            return EvidenceAssessment(
                sufficient=False,
                reason_code="low_score",
                citation_count=citation_count,
                top_score=top_score,
            )

        if self._has_conflicting_evidence(hits):
            return EvidenceAssessment(
                sufficient=False,
                reason_code="conflicting_evidence",
                citation_count=citation_count,
                top_score=top_score,
            )

        return EvidenceAssessment(
            sufficient=True,
            reason_code="sufficient",
            citation_count=citation_count,
            top_score=top_score,
        )

    def _has_conflicting_evidence(self, hits: list[ChunkSearchHit]) -> bool:
        """基于明显冲突词检测前几条证据是否相互矛盾。"""
        candidate_texts = [
            self._normalize_text(hit.content) for hit in hits[:3] if hit.content.strip()
        ]
        if len(candidate_texts) < 2:
            return False

        for left_text in candidate_texts:
            for right_text in candidate_texts:
                if left_text == right_text:
                    continue
                for positive, negative in self._CONFLICT_TOKEN_PAIRS:
                    if positive in left_text and negative in right_text:
                        return True
                    if negative in left_text and positive in right_text:
                        return True
        return False

    def _normalize_text(self, text: str) -> str:
        """把文本规整为便于冲突检测的单行小写形式。"""
        return re.sub(r"\s+", " ", text).strip().lower()
