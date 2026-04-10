from __future__ import annotations

from baozhi_rag.services.llm import ChatMessage
from baozhi_rag.services.query_rewrite import QueryRewriteService


def test_query_rewrite_service_rewrites_short_followup() -> None:
    service = QueryRewriteService()

    result = service.rewrite(
        [
            ChatMessage(role="user", content="接口限流窗口是什么意思"),
            ChatMessage(role="assistant", content="上一轮回答"),
            ChatMessage(role="user", content="那认证错误码呢"),
        ],
        original_query="那认证错误码呢",
    )

    assert result.rewrite_applied is True
    assert result.rewritten_query == "接口限流窗口是什么意思。补充问题：那认证错误码呢"


def test_query_rewrite_service_keeps_query_when_no_previous_user_message() -> None:
    service = QueryRewriteService()

    result = service.rewrite(
        [ChatMessage(role="user", content="这个配置项是什么意思")],
        original_query="这个配置项是什么意思",
    )

    assert result.rewrite_applied is False
    assert result.rewritten_query == "这个配置项是什么意思"
