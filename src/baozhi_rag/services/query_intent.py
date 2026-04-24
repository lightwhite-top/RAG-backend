"""查询意图识别服务。"""

from __future__ import annotations

import re


class QueryIntentService:
    """根据规则识别查询主意图。"""

    _STRUCTURED_INTENT_PATTERN = re.compile(
        r"(表格|图片|图像|截图|配图|图示|流程图|印章|ocr|影像|扫描|扫描件)",
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
        r"(在哪|位置|哪一章|哪一节|哪一段|章节|标题|哪个文件|原文)",
        re.IGNORECASE,
    )

    def detect(self, query_text: str) -> str:
        """识别当前查询的主意图。"""
        if self._STRUCTURED_INTENT_PATTERN.search(query_text):
            return "structured"
        if self._COMPARISON_INTENT_PATTERN.search(query_text):
            return "comparison"
        if self._PROCEDURE_INTENT_PATTERN.search(query_text):
            return "procedure"
        if self._DEFINITION_INTENT_PATTERN.search(query_text):
            return "definition"
        if self._DOCUMENT_LOCATION_INTENT_PATTERN.search(query_text):
            return "document_location"
        return "general"
