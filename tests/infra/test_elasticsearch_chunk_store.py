from __future__ import annotations

from baozhi_rag.infra.retrieval.elasticsearch_chunk_store import ElasticsearchChunkStore
from baozhi_rag.services.chunk_search import ChunkSearchRequest


def test_elasticsearch_search_query_includes_structure_queries_for_document_location() -> None:
    request = ChunkSearchRequest(
        query_text="这个内容在哪一章",
        size=5,
        merged_terms=[],
        query_embedding=[0.1],
        query_intent="document_location",
    )

    query = ElasticsearchChunkStore.build_search_query(request)
    should_queries = query["bool"]["should"]

    assert any("section_title" in item.get("match", {}) for item in should_queries)
    assert any("heading_path" in item.get("match", {}) for item in should_queries)


def test_elasticsearch_search_query_boosts_table_for_structured_intent() -> None:
    request = ChunkSearchRequest(
        query_text="表格里第二列是什么意思",
        size=5,
        merged_terms=[],
        query_embedding=[0.1],
        query_intent="structured",
    )

    query = ElasticsearchChunkStore.build_search_query(request)
    should_queries = query["bool"]["should"]

    assert any(
        item.get("constant_score", {}).get("filter") == {"term": {"content_type": "table"}}
        for item in should_queries
    )


def test_elasticsearch_parse_hit_preserves_structure_fields() -> None:
    hit = ElasticsearchChunkStore._parse_hit(  # pyright: ignore[reportPrivateUsage]
        {
            "_score": 1.23,
            "_source": {
                "chunk_id": "chunk-1",
                "file_id": "file-1",
                "chunk_type": "text",
                "segment_id": "seg-1",
                "source_filename": "guide.md",
                "storage_key": "guide.md",
                "uploader_user_id": "user-1",
                "visibility_scope": "global",
                "chunk_index": 0,
                "char_count": 12,
                "heading_path": ["第一章", "配置说明"],
                "section_title": "配置说明",
                "content_type": "table",
                "content": "表格内容",
                "merged_terms": ["配置"],
                "image_assets": [],
            },
        }
    )

    assert hit.heading_path == ["第一章", "配置说明"]
    assert hit.section_title == "配置说明"
    assert hit.content_type == "table"
