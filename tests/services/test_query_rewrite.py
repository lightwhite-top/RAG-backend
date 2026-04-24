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


def test_query_rewrite_service_keeps_short_self_contained_query() -> None:
    service = QueryRewriteService()

    result = service.rewrite(
        [
            ChatMessage(role="user", content="上一轮问题"),
            ChatMessage(role="assistant", content="上一轮回答"),
            ChatMessage(role="user", content="数据库巡检时间窗是什么"),
        ],
        original_query="数据库巡检时间窗是什么",
    )

    assert result.rewrite_applied is False
    assert result.rewritten_query == "数据库巡检时间窗是什么"


def test_query_rewrite_service_includes_previous_assistant_for_structure_followup() -> None:
    service = QueryRewriteService()

    result = service.rewrite(
        [
            ChatMessage(
                role="user",
                content="先看 01_财务报销制度_V1.docx，其中“单笔差旅自动审批上限”是什么？",
            ),
            ChatMessage(role="assistant", content="单笔差旅自动审批上限是1500元。"),
            ChatMessage(role="user", content="那第二列那个规则具体怎么写？"),
        ],
        original_query="那第二列那个规则具体怎么写？",
    )

    assert result.rewrite_applied is True
    assert "上一轮回答：单笔差旅自动审批上限是1500元。" in result.rewritten_query
