"""查询意图识别服务。"""

from __future__ import annotations

import re

from baozhi_rag.services.document_type_classifier import (
    DOCUMENT_TYPE_COMMUNICATION,
    DOCUMENT_TYPE_MANUAL,
    DOCUMENT_TYPE_MATRIX,
    DOCUMENT_TYPE_OCR_SCAN,
    DOCUMENT_TYPE_PRICE,
    DOCUMENT_TYPE_VERSION_CHAIN,
)


class QueryIntentService:
    """根据规则识别查询主意图。"""

    _PRECISE_DOCUMENT_LOCATION_PATTERN = re.compile(
        r"(样例\s*\d+|案例\s*\d+|版本\s*[vV]?\s*\d+|第\s*\d+\s*页|"
        r"(?:处理时限|审批阈值|外发链接).*\d+(?:\.\d+)?\s*(?:小时|分钟|秒|天|周|月|年|元|万元|次|位|页|条|%)|"
        r"超过\s*\d+(?:\.\d+)?\s*(?:小时|分钟|秒|天|周|月|年|元|万元|次|位|页|条|%)\s*记为(?:超时|逾期)|"
        r"默认\s*\d+(?:\.\d+)?\s*(?:小时|分钟|秒|天|周|月|年|元|万元|次|位|页|条|%)\s*失效)",
        re.IGNORECASE,
    )
    _STRUCTURED_INTENT_PATTERN = re.compile(
        r"(表格|图片|截图|配图|图示|流程图|印章|ocr)",
        re.IGNORECASE,
    )
    _COMPARISON_INTENT_PATTERN = re.compile(
        r"(区别|对比|比较|不同|差异|vs\b)",
        re.IGNORECASE,
    )
    _PROCEDURE_INTENT_PATTERN = re.compile(
        r"(如何|怎么|步骤|流程|顺序|过程)",
        re.IGNORECASE,
    )
    _DEFINITION_INTENT_PATTERN = re.compile(
        r"(是什么|什么是|含义|定义|解释|说明一下)",
        re.IGNORECASE,
    )
    _DOCUMENT_LOCATION_INTENT_PATTERN = re.compile(
        r"(在哪|位置|哪一章|哪一节|哪一段|章节|标题|哪个文件|原文|包含什么|有哪些|支持哪些)",
        re.IGNORECASE,
    )
    _PRICE_PATTERN = re.compile(r"(价格|月费|套餐|定价|折扣|价目)", re.IGNORECASE)
    _MATRIX_PATTERN = re.compile(r"(矩阵|对照表|速查表|映射表)", re.IGNORECASE)
    _OCR_PATTERN = re.compile(
        r"(样例\s*\d+|扫描|盖章|水印|低分辨率|ocr|"
        r"处理时限.*\d+\s*小时|超过.*\d+\s*小时.*超时|外发链接.*\d+\s*小时.*失效|审批阈值.*\d+\s*元)",
        re.IGNORECASE,
    )
    _COMMUNICATION_PATTERN = re.compile(r"(邮件|群聊|纪要|工单|交接)", re.IGNORECASE)
    _MANUAL_PATTERN = re.compile(
        r"(手册|帮助中心|接口|使用说明|支持哪些|包含什么|有哪些|模板)", re.IGNORECASE
    )
    _LATEST_VERSION_PATTERN = re.compile(r"(最新|新版|正式版|现行|当前)", re.IGNORECASE)
    _OLDEST_VERSION_PATTERN = re.compile(r"(旧版|历史|以前|之前)", re.IGNORECASE)

    def detect(self, query_text: str) -> str:
        """识别当前查询的主意图。"""
        if self._STRUCTURED_INTENT_PATTERN.search(query_text):
            return "structured"
        if self._PRECISE_DOCUMENT_LOCATION_PATTERN.search(query_text):
            return "document_location"
        if self._COMPARISON_INTENT_PATTERN.search(query_text):
            return "comparison"
        if self._PROCEDURE_INTENT_PATTERN.search(query_text):
            return "procedure"
        if self._DEFINITION_INTENT_PATTERN.search(query_text):
            return "definition"
        if self._DOCUMENT_LOCATION_INTENT_PATTERN.search(query_text):
            return "document_location"
        return "general"

    def detect_document_types(self, query_text: str) -> list[str]:
        """识别当前查询优先命中的文档类型集合。"""
        detected_types: list[str] = []
        if self._PRICE_PATTERN.search(query_text):
            detected_types.extend([DOCUMENT_TYPE_PRICE, DOCUMENT_TYPE_VERSION_CHAIN])
        if self._MATRIX_PATTERN.search(query_text):
            detected_types.append(DOCUMENT_TYPE_MATRIX)
        if self._OCR_PATTERN.search(query_text):
            detected_types.append(DOCUMENT_TYPE_OCR_SCAN)
        if self._COMMUNICATION_PATTERN.search(query_text):
            detected_types.append(DOCUMENT_TYPE_COMMUNICATION)
        if self._MANUAL_PATTERN.search(query_text):
            detected_types.append(DOCUMENT_TYPE_MANUAL)

        deduplicated_types: list[str] = []
        seen_types: set[str] = set()
        for document_type in detected_types:
            if document_type in seen_types:
                continue
            seen_types.add(document_type)
            deduplicated_types.append(document_type)
        return deduplicated_types

    def detect_version_preference(self, query_text: str) -> str:
        """识别查询偏向最新版本、旧版本还是不限版本。"""
        if self._OLDEST_VERSION_PATTERN.search(query_text):
            return "oldest"
        if self._LATEST_VERSION_PATTERN.search(query_text):
            return "latest"
        return "latest"
