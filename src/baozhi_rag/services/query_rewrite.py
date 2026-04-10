"""查询改写服务。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from baozhi_rag.services.llm import ChatMessage


@dataclass(frozen=True, slots=True)
class QueryRewriteResult:
    """查询改写结果。"""

    original_query: str
    rewritten_query: str
    rewrite_applied: bool


class QueryRewriteService:
    """对明显依赖上文的短追问做轻量规则改写。"""

    _REWRITE_TRIGGER_PATTERN = re.compile(
        r"(这个|那个|它|其|刚才|上面|上一|前面|继续|再说|第二列|第一列|第二个|上一个|这里|那里)"
    )

    def __init__(self, *, max_rewrite_query_length: int = 24) -> None:
        self._max_rewrite_query_length = max(1, max_rewrite_query_length)

    def rewrite(
        self,
        messages: list[ChatMessage],
        *,
        original_query: str,
    ) -> QueryRewriteResult:
        """根据最近一轮用户问题对当前追问做规则改写。"""
        if not self._should_rewrite_query(original_query):
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=original_query,
                rewrite_applied=False,
            )

        previous_user_query = self._find_previous_user_query(
            messages,
            current_query=original_query,
        )
        if not previous_user_query:
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=original_query,
                rewrite_applied=False,
            )

        rewritten_query = f"{previous_user_query}。补充问题：{original_query}".strip()
        if rewritten_query == original_query:
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=original_query,
                rewrite_applied=False,
            )

        return QueryRewriteResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            rewrite_applied=True,
        )

    def _should_rewrite_query(self, query_text: str) -> bool:
        """判断当前查询是否更像需要结合上文的短追问。"""
        normalized_query = query_text.strip()
        if not normalized_query:
            return False
        if len(normalized_query) <= self._max_rewrite_query_length:
            return True
        return bool(self._REWRITE_TRIGGER_PATTERN.search(normalized_query))

    def _find_previous_user_query(
        self,
        messages: list[ChatMessage],
        *,
        current_query: str,
    ) -> str:
        """查找当前查询之前最近的一条用户问题。"""
        seen_current = False
        for message in reversed(messages):
            if message.role != "user":
                continue
            if not seen_current and message.content == current_query:
                seen_current = True
                continue
            if seen_current:
                return message.content.strip()
        return ""
