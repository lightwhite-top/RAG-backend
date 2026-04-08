"""领域词典。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DOMAIN_DICTIONARY_PATH = Path("data/domain_dictionary.txt")


@dataclass(frozen=True, slots=True)
class DomainTermDictionary:
    """封装领域词典与匹配辅助信息。"""

    terms: frozenset[str]
    max_term_length: int

    @classmethod
    def from_terms(cls, terms: set[str] | frozenset[str]) -> DomainTermDictionary:
        """基于词项集合构造领域词典。

        参数:
            terms: 已去重的领域词项集合。

        返回:
            词典对象；空词典时最大词长为 0。
        """
        normalized_terms = frozenset(term.strip() for term in terms if term.strip())
        max_term_length = max((len(term) for term in normalized_terms), default=0)
        return cls(terms=normalized_terms, max_term_length=max_term_length)

    @classmethod
    def from_file(
        cls,
        dictionary_path: Path,
        base_terms: set[str] | frozenset[str] | None = None,
    ) -> DomainTermDictionary:
        """从文本文件加载领域词典。

        参数:
            dictionary_path: 词典文件路径，按行存储词项，支持 `#` 注释。
            base_terms: 可选基础词典，会与文件中的词项合并。

        返回:
            合并后的领域词典对象。

        异常:
            OSError: 当词典文件无法读取时抛出。
        """
        loaded_terms = set(base_terms or set())
        loaded_terms.update(_load_terms_from_text_file(dictionary_path))
        return cls.from_terms(loaded_terms)

    def contains(self, term: str) -> bool:
        """判断词项是否存在于词典中。"""
        return term in self.terms


def _load_terms_from_text_file(dictionary_path: Path) -> set[str]:
    """从纯文本词典文件加载词项集合。

    参数:
        dictionary_path: 词典文件路径，按行存储词项，支持 `#` 注释。

    返回:
        已去除空行和注释后的词项集合。

    异常:
        OSError: 当词典文件无法读取时抛出。
    """
    loaded_terms: set[str] = set()
    for line in dictionary_path.read_text(encoding="utf-8").splitlines():
        normalized_line = line.strip()
        if not normalized_line or normalized_line.startswith("#"):
            continue
        loaded_terms.add(normalized_line)
    return loaded_terms


def load_domain_dictionary(dictionary_path: Path | None = None) -> DomainTermDictionary:
    """加载领域词典。"""
    resolved_dictionary_path = dictionary_path or DEFAULT_DOMAIN_DICTIONARY_PATH
    return DomainTermDictionary.from_file(resolved_dictionary_path)
