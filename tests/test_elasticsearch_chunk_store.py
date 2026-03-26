"""Elasticsearch chunk store 测试。"""

from __future__ import annotations

from typing import Any, cast

from baozhi_rag.infra.retrieval.elasticsearch_chunk_store import ElasticsearchChunkStore
from baozhi_rag.services.chunk_search import ChunkSearchRequest


def test_build_search_query_contains_fulltext_and_term_boosts() -> None:
    """混合检索查询应同时包含全文和词项匹配。"""
    request = ChunkSearchRequest(
        query_text="免赔额责任",
        size=5,
        fmm_terms=["免赔额"],
        bmm_terms=["责任免除"],
        merged_terms=["免赔额", "责任免除"],
    )

    query = ElasticsearchChunkStore.build_search_query(request)
    should_queries = cast(list[dict[str, Any]], cast(dict[str, Any], query["bool"])["should"])

    assert len(should_queries) == 4
    assert should_queries[0]["match"]["content"]["query"] == "免赔额责任"
    assert should_queries[1]["constant_score"]["filter"]["terms"]["merged_terms"] == [
        "免赔额",
        "责任免除",
    ]
    assert should_queries[2]["constant_score"]["filter"]["terms"]["fmm_terms"] == ["免赔额"]
    assert should_queries[3]["constant_score"]["filter"]["terms"]["bmm_terms"] == ["责任免除"]
