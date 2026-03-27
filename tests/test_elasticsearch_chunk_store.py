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
        query_embedding=None,
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


def test_build_search_query_uses_script_score_when_query_embedding_present() -> None:
    """启用查询向量时应构造语义混合检索查询。"""
    request = ChunkSearchRequest(
        query_text="免赔额责任",
        size=5,
        fmm_terms=["免赔额"],
        bmm_terms=["责任免除"],
        merged_terms=["免赔额", "责任免除"],
        query_embedding=[0.1, 0.2, 0.3],
    )

    query = ElasticsearchChunkStore.build_search_query(request)
    script_score = cast(dict[str, Any], query["script_score"])
    bool_query = cast(dict[str, Any], cast(dict[str, Any], script_score["query"])["bool"])

    assert bool_query["minimum_should_match"] == 0
    assert bool_query["filter"] == [{"exists": {"field": "content_embedding"}}]
    assert script_score["script"]["params"]["query_vector"] == [0.1, 0.2, 0.3]
    assert "cosineSimilarity" in script_score["script"]["source"]


def test_build_mappings_contains_dense_vector_when_embedding_enabled() -> None:
    """启用向量检索时应为索引补充 dense_vector 字段。"""
    store = ElasticsearchChunkStore(
        index_name="document_chunks",
        url="http://127.0.0.1:9200",
        api_key=None,
        username=None,
        password=None,
        verify_certs=False,
        embedding_enabled=True,
        embedding_dimensions=1024,
    )

    mappings = store._build_mappings()
    properties = cast(dict[str, Any], mappings["properties"])

    assert properties["content_embedding"] == {
        "type": "dense_vector",
        "dims": 1024,
        "index": False,
    }
