"""金融保险领域词典。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DOMAIN_TERMS = frozenset(
    {
        "保险",
        "保险单",
        "保单",
        "投保",
        "投保人",
        "被保险人",
        "受益人",
        "保险责任",
        "责任免除",
        "免责条款",
        "等待期",
        "犹豫期",
        "现金价值",
        "保费",
        "保额",
        "续保",
        "退保",
        "理赔",
        "核保",
        "核赔",
        "免赔",
        "免赔额",
        "给付",
        "赔付",
        "宽限期",
        "保单年度",
        "保障期限",
        "保险期间",
        "生效日",
        "失效日",
        "复效",
        "健康告知",
        "如实告知",
        "主险",
        "附加险",
        "特约",
        "意外险",
        "医疗险",
        "住院医疗",
        "门急诊",
        "重疾险",
        "重大疾病",
        "疾病保险",
        "重大疾病保险",
        "寿险",
        "年金险",
        "既往症",
        "观察期",
        "保险金",
        "保全",
        "回访",
        "保单借款",
        "费率",
        "赔案",
        "报案",
        "出险",
    }
)


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
        for line in dictionary_path.read_text(encoding="utf-8").splitlines():
            normalized_line = line.strip()
            if not normalized_line or normalized_line.startswith("#"):
                continue
            loaded_terms.add(normalized_line)
        return cls.from_terms(loaded_terms)

    def contains(self, term: str) -> bool:
        """判断词项是否存在于词典中。"""
        return term in self.terms


def load_domain_dictionary(dictionary_path: Path | None = None) -> DomainTermDictionary:
    """加载默认金融保险词典，可选叠加外部词典。"""
    if dictionary_path is None:
        return DomainTermDictionary.from_terms(DEFAULT_DOMAIN_TERMS)
    return DomainTermDictionary.from_file(dictionary_path, base_terms=DEFAULT_DOMAIN_TERMS)
