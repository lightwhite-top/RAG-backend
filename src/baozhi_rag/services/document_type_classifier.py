"""基于文件名规则的文档类型与版本链元数据分类器。"""

from __future__ import annotations

import re
from pathlib import Path

DOCUMENT_TYPE_POLICY = "policy"
DOCUMENT_TYPE_MATRIX = "matrix"
DOCUMENT_TYPE_PRICE = "price"
DOCUMENT_TYPE_VERSION_CHAIN = "version_chain"
DOCUMENT_TYPE_MANUAL = "manual"
DOCUMENT_TYPE_CASE = "case"
DOCUMENT_TYPE_COMMUNICATION = "communication"
DOCUMENT_TYPE_OCR_SCAN = "ocr_scan"

ALL_DOCUMENT_TYPES = {
    DOCUMENT_TYPE_POLICY,
    DOCUMENT_TYPE_MATRIX,
    DOCUMENT_TYPE_PRICE,
    DOCUMENT_TYPE_VERSION_CHAIN,
    DOCUMENT_TYPE_MANUAL,
    DOCUMENT_TYPE_CASE,
    DOCUMENT_TYPE_COMMUNICATION,
    DOCUMENT_TYPE_OCR_SCAN,
}

_OCR_SCAN_PATTERN = re.compile(r"(扫描|ocr|样例|水印|盖章|低分辨率)", re.IGNORECASE)
_VERSION_CHAIN_PATTERN = re.compile(r"(版本链|[_\s-]v\d+)", re.IGNORECASE)
_MATRIX_PATTERN = re.compile(r"(矩阵|对照表|速查表|映射表)", re.IGNORECASE)
_PRICE_PATTERN = re.compile(r"(定价|价格|套餐|月费|折扣|价目)", re.IGNORECASE)
_COMMUNICATION_PATTERN = re.compile(r"(邮件|群聊|纪要|工单|交接)", re.IGNORECASE)
_MANUAL_PATTERN = re.compile(r"(手册|帮助中心|接口说明|使用说明|模板|上传限制)", re.IGNORECASE)
_CASE_PATTERN = re.compile(r"(案例|题库|样例)", re.IGNORECASE)
_POLICY_PATTERN = re.compile(r"(制度|规则|规范|策略|办法|流程|规定)", re.IGNORECASE)
_VERSION_GROUP_PATTERN = re.compile(
    r"^\d+_(?P<group>.+?)(?:_版本链(?:[_\s-]*v\d+)?|_(?:正式版|旧版|新版|历史版))$",
    re.IGNORECASE,
)


def classify(source_filename: str) -> str:
    """根据文件名规则判定文档类型。

    参数:
        source_filename: 上传时的原始文件名。

    返回:
        8 类之一；未命中任何规则时回退到 `policy`。
    """
    normalized_name = Path(source_filename).name.strip()
    lowered_name = normalized_name.lower()

    if lowered_name.endswith(".pdf") and _OCR_SCAN_PATTERN.search(normalized_name):
        return DOCUMENT_TYPE_OCR_SCAN
    if _VERSION_CHAIN_PATTERN.search(normalized_name):
        return DOCUMENT_TYPE_VERSION_CHAIN
    if _MATRIX_PATTERN.search(normalized_name):
        return DOCUMENT_TYPE_MATRIX
    if _PRICE_PATTERN.search(normalized_name):
        return DOCUMENT_TYPE_PRICE
    if _COMMUNICATION_PATTERN.search(normalized_name):
        return DOCUMENT_TYPE_COMMUNICATION
    if _MANUAL_PATTERN.search(normalized_name):
        return DOCUMENT_TYPE_MANUAL
    if _CASE_PATTERN.search(normalized_name):
        return DOCUMENT_TYPE_CASE
    if _POLICY_PATTERN.search(normalized_name):
        return DOCUMENT_TYPE_POLICY
    return DOCUMENT_TYPE_POLICY


def extract_version_rank(source_filename: str) -> int:
    """从文件名中提取版本优先级，越大表示越新。"""
    normalized_name = Path(source_filename).stem.lower()
    version_match = re.search(r"版本链[\s_-]*v(\d+)", normalized_name)
    if version_match is not None:
        return int(version_match.group(1))
    if "新版" in normalized_name or "正式版" in normalized_name or "现行" in normalized_name:
        return 100
    if "旧版" in normalized_name or "历史" in normalized_name:
        return 0
    return 0


def extract_version_chain_group(source_filename: str) -> str | None:
    """从文件名中提取版本链主题名。

    参数:
        source_filename: 上传时的原始文件名。

    返回:
        版本链主题名；非版本链文件返回 `None`。
    """
    stem = Path(source_filename).stem
    matched = _VERSION_GROUP_PATTERN.match(stem)
    if matched is None:
        return None
    normalized_group = matched.group("group").replace("_", " ").strip()
    return normalized_group or None
