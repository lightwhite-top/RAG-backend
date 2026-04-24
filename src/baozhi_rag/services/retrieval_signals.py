"""检索相关的轻量信号提取工具。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_VERSION_PATTERN = re.compile(r"(?:^|[\s_-])v(?P<rank>\d+)\b", re.IGNORECASE)
_LEADING_SERIAL_PATTERN = re.compile(r"^\d+[_\-\s]+")
_SAMPLE_PATTERN = re.compile(r"样例\s*(?P<index>\d+)", re.IGNORECASE)
_FILENAME_WITH_EXTENSION_PATTERN = re.compile(
    r"(?P<filename>(?<![\w\u4e00-\u9fff])[\w\u4e00-\u9fff][\w\u4e00-\u9fff\-\s（）()【】\[\]·_.]*?\.(?:docx|pdf|doc|xlsx|xls|pptx|ppt|md|txt))(?![\w\u4e00-\u9fff])",
    re.IGNORECASE,
)
_FILENAME_TOKEN_PATTERN = re.compile(
    r"[\w\u4e00-\u9fff\-（）()【】\[\]·_.]+\.(?:docx|pdf|doc|xlsx|xls|pptx|ppt|md|txt)",
    re.IGNORECASE,
)
_NUMBER_WITH_UNIT_PATTERN = re.compile(
    r"(?P<number>\d+)\s*(?P<unit>小时|天|年|月|位|分钟)",
    re.IGNORECASE,
)
_OLD_VERSION_MARKERS = ("旧版", "老版", "历史版本", "原版", "之前版本")
_LATEST_VERSION_MARKERS = ("新版", "最新", "现行", "当前版本", "目前版本")
_QUERY_STOPWORD_PATTERN = re.compile(
    r"(什么是|是什么|必须|包含|需要|如何|怎么|多少|几个|几天|几位|几小时|新版|旧版|最新|当前|现行|规则|要求|规定|记为|为|是|的|了|吗|呢|么|可|内|后|前)",
    re.IGNORECASE,
)
_QUERY_FILENAME_STRIP_CHARS = "\"'“”‘’《》"


@dataclass(frozen=True, slots=True)
class QueryFileAnchor:
    """查询中显式点名的目标文件锚点。"""

    raw_filename: str
    normalized_filename: str
    normalized_title: str
    file_family_key: str | None
    version_rank: int | None = None


def normalize_text(text: str) -> str:
    """把文本规整为便于信号提取的单行形式。"""
    return " ".join(text.split()).strip().lower()


def normalize_filename_title(filename: str) -> str:
    """把文件名归一化为更适合检索与排序的标题文本。"""
    stem = Path(filename).stem.strip()
    if not stem:
        return ""

    without_serial = _LEADING_SERIAL_PATTERN.sub("", stem)
    normalized_title = without_serial.replace("_", " ")
    normalized_title = re.sub(r"\s+", " ", normalized_title).strip(" -_")
    return normalized_title or stem


def extract_version_rank(filename_or_title: str) -> int:
    """从文件名或标题中提取版本号，未命中时返回 0。"""
    matches = list(_VERSION_PATTERN.finditer(normalize_filename_title(filename_or_title)))
    if not matches:
        return 0
    return int(matches[-1].group("rank"))


def build_version_chain_key(filename_or_title: str) -> str | None:
    """构造版本链聚合键，用于同主题多版本文件去重。"""
    normalized_title = normalize_filename_title(filename_or_title)
    if not normalized_title or extract_version_rank(normalized_title) <= 0:
        return None

    chain_key = _VERSION_PATTERN.sub("", normalized_title)
    chain_key = re.sub(r"\s+", " ", chain_key).strip(" -_")
    return chain_key.lower() or None


def build_file_family_key(filename_or_title: str) -> str | None:
    """构造文件族键，供显式文件锚点与版本链匹配使用。"""
    version_chain_key = build_version_chain_key(filename_or_title)
    if version_chain_key is not None:
        return version_chain_key

    normalized_title = normalize_filename_title(filename_or_title)
    return normalized_title.lower() or None


def extract_query_file_anchor(query_text: str) -> QueryFileAnchor | None:
    """从查询文本中提取显式点名的目标文件锚点。"""
    match = _FILENAME_WITH_EXTENSION_PATTERN.search(query_text)
    if match is None:
        return None

    matched_text = match.group("filename").strip().strip(_QUERY_FILENAME_STRIP_CHARS)
    filename_tokens = _FILENAME_TOKEN_PATTERN.findall(matched_text)
    raw_filename = Path(filename_tokens[-1] if filename_tokens else matched_text).name.strip()
    if not raw_filename or "." not in raw_filename:
        return None

    normalized_title = normalize_filename_title(raw_filename)
    extracted_version_rank = extract_version_rank(raw_filename)
    explicit_version_rank = extracted_version_rank or _extract_query_version_rank(query_text)
    return QueryFileAnchor(
        raw_filename=raw_filename,
        normalized_filename=raw_filename.casefold(),
        normalized_title=normalized_title,
        file_family_key=build_file_family_key(raw_filename),
        version_rank=explicit_version_rank if explicit_version_rank > 0 else None,
    )


def matches_query_file_anchor(anchor: QueryFileAnchor, filename_or_title: str) -> bool:
    """判断候选文件是否满足查询中的显式文件锚点。"""
    normalized_candidate_filename = Path(filename_or_title).name.strip().casefold()
    if normalized_candidate_filename == anchor.normalized_filename:
        return True

    candidate_family_key = build_file_family_key(filename_or_title)
    if anchor.file_family_key is None or candidate_family_key != anchor.file_family_key:
        return False
    if anchor.version_rank is None:
        return (
            normalize_filename_title(filename_or_title).casefold()
            == anchor.normalized_title.casefold()
        )
    return extract_version_rank(filename_or_title) == anchor.version_rank


def belongs_to_query_file_family(anchor: QueryFileAnchor, filename_or_title: str) -> bool:
    """判断候选文件是否属于查询显式点名文件的同一文件族。"""
    candidate_family_key = build_file_family_key(filename_or_title)
    return anchor.file_family_key is not None and candidate_family_key == anchor.file_family_key


def query_prefers_old_version(query_text: str) -> bool:
    """判断查询是否显式偏向旧版本内容。"""
    normalized_query = normalize_text(query_text)
    return any(marker in normalized_query for marker in _OLD_VERSION_MARKERS)


def query_requests_latest_version(query_text: str) -> bool:
    """判断查询是否显式偏向最新版内容。"""
    normalized_query = normalize_text(query_text)
    return any(marker in normalized_query for marker in _LATEST_VERSION_MARKERS)


def extract_exact_search_terms(text: str) -> list[str]:
    """提取适合精确匹配的数字/编号锚点。"""
    normalized_text = " ".join(text.split()).strip()
    if not normalized_text:
        return []

    exact_terms: list[str] = []
    for match in _SAMPLE_PATTERN.finditer(normalized_text):
        sample_index = match.group("index")
        exact_terms.extend([f"样例 {sample_index}", f"样例{sample_index}"])

    for match in _NUMBER_WITH_UNIT_PATTERN.finditer(normalized_text):
        number = match.group("number")
        unit = match.group("unit")
        exact_terms.extend([f"{number} {unit}", f"{number}{unit}"])

    return _deduplicate_terms(exact_terms)


def extract_query_terms(text: str) -> list[str]:
    """从查询文本中提取更适合快速重排和精确召回的关键词。"""
    normalized_text = normalize_text(text)
    if not normalized_text:
        return []

    query_terms: list[str] = []
    query_terms.extend(_extract_basic_terms(normalized_text))
    query_terms.extend(extract_exact_search_terms(text))
    query_terms.extend(_extract_focus_phrases(text))
    return _deduplicate_terms(query_terms)


def _extract_basic_terms(normalized_text: str) -> list[str]:
    """提取基础切分词，保留数字类单字符信号。"""
    return [
        term
        for term in re.split(r"[\s,.;:!?/\\|()\[\]{}<>，。；：！？、]+", normalized_text)
        if len(term) >= 2 or term.isdigit()
    ]


def _extract_focus_phrases(text: str) -> list[str]:
    """通过剔除常见问句停用词，保留更聚焦的短语。"""
    normalized_text = " ".join(text.split()).strip()
    if not normalized_text:
        return []

    simplified_text = _QUERY_STOPWORD_PATTERN.sub(" ", normalized_text)
    focus_phrases = [
        item.strip()
        for item in re.split(r"[\s,.;:!?/\\|()\[\]{}<>，。；：！？、]+", simplified_text)
        if len(item.strip()) >= 2
    ]
    return focus_phrases


def _deduplicate_terms(terms: list[str]) -> list[str]:
    """按出现顺序去重并清理空白。"""
    deduplicated_terms: list[str] = []
    seen_signatures: set[str] = set()
    for raw_term in terms:
        normalized_term = " ".join(raw_term.split()).strip()
        if not normalized_term:
            continue
        signature = normalized_term.casefold()
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduplicated_terms.append(normalized_term)
    return deduplicated_terms


def _extract_query_version_rank(query_text: str) -> int:
    """从查询全文中提取显式版本号，未命中时返回 0。"""
    matches = list(_VERSION_PATTERN.finditer(query_text))
    if not matches:
        return 0
    return int(matches[-1].group("rank"))
