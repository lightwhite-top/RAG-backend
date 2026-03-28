"""领域词抽取服务。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baozhi_rag.domain.term_dictionary import DomainTermDictionary, load_domain_dictionary


@dataclass(frozen=True, slots=True)
class TermMatchResult:
    """单段文本的领域词匹配结果。"""

    merged_terms: list[str]


class MaximumMatchingTermMatcher:
    """基于领域词典执行最大匹配并输出合并后的领域词。"""

    def __init__(self, dictionary: DomainTermDictionary) -> None:
        """初始化领域词匹配服务。"""
        self._dictionary = dictionary

    def extract_terms(self, text: str) -> TermMatchResult:
        """从文本中抽取去重后的领域词项。"""
        fmm_terms = self._forward_maximum_matching(text)
        bmm_terms = self._backward_maximum_matching(text)
        merged_terms = list(dict.fromkeys([*fmm_terms, *bmm_terms]))
        return TermMatchResult(merged_terms=merged_terms)

    def _forward_maximum_matching(self, text: str) -> list[str]:
        """执行正向最大匹配。"""
        matched_terms: list[str] = []
        cursor = 0

        while cursor < len(text):
            matched_term = self._match_from_left(text=text, start=cursor)
            if matched_term is None:
                cursor += 1
                continue
            matched_terms.append(matched_term)
            cursor += len(matched_term)

        return matched_terms

    def _backward_maximum_matching(self, text: str) -> list[str]:
        """执行逆向最大匹配。"""
        matched_terms: list[str] = []
        cursor = len(text)

        while cursor > 0:
            matched_term = self._match_from_right(text=text, end=cursor)
            if matched_term is None:
                cursor -= 1
                continue
            matched_terms.append(matched_term)
            cursor -= len(matched_term)

        matched_terms.reverse()
        return matched_terms

    def _match_from_left(self, text: str, start: int) -> str | None:
        """从左向右尝试匹配最长词项。"""
        max_length = min(self._dictionary.max_term_length, len(text) - start)
        for length in range(max_length, 0, -1):
            candidate = text[start : start + length]
            if self._dictionary.contains(candidate):
                return candidate
        return None

    def _match_from_right(self, text: str, end: int) -> str | None:
        """从右向左尝试匹配最长词项。"""
        max_length = min(self._dictionary.max_term_length, end)
        for length in range(max_length, 0, -1):
            candidate = text[end - length : end]
            if self._dictionary.contains(candidate):
                return candidate
        return None


def build_default_term_matcher(
    dictionary_path: Path | None = None,
) -> MaximumMatchingTermMatcher:
    """构造默认领域词匹配服务。"""
    return MaximumMatchingTermMatcher(load_domain_dictionary(dictionary_path))
