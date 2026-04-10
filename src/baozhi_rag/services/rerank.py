"""检索 chunk 与图片候选的重排服务。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from baozhi_rag.services.chunk_search import ChunkSearchHit
from baozhi_rag.services.document_chunking import ChunkImageAsset
from baozhi_rag.services.llm import ChatMessage, StructuredOutputModelClient


def _normalize_text_for_matching(text: str) -> str:
    """把文本规整为便于关键词匹配的单行形式。"""
    return " ".join(text.split()).lower()


def _extract_query_terms(text: str) -> list[str]:
    """从查询上下文中提取简单关键词。"""
    normalized_text = _normalize_text_for_matching(text)
    return [
        term
        for term in re.split(r"[\s,.;:!?/\\|()\[\]{}<>，。；：！？、]+", normalized_text)
        if len(term) >= 2
    ]


class ChunkRerankService:
    """使用通用重排模型对 chunk 候选做二次排序。"""

    def __init__(
        self,
        *,
        client: StructuredOutputModelClient,
        model_name: str | None,
    ) -> None:
        self._client = client
        self._model_name = (model_name or "").strip()

    def rerank(
        self,
        *,
        query_text: str,
        hits: list[ChunkSearchHit],
    ) -> list[ChunkSearchHit]:
        """按查询文本对混合检索候选进行重排。"""
        if len(hits) <= 1 or not self._model_name:
            return hits

        try:
            ordered_ids = self._invoke_model(
                query_text=query_text,
                hits=hits,
            )
        except Exception:
            return self._fallback_rerank(query_text=query_text, hits=hits)

        if not ordered_ids:
            return self._fallback_rerank(query_text=query_text, hits=hits)
        return self._reorder_hits(hits, ordered_ids)

    def _invoke_model(
        self,
        *,
        query_text: str,
        hits: list[ChunkSearchHit],
    ) -> list[str]:
        """调用重排模型输出 chunk_id 顺序。"""
        payload = {
            "query_text": query_text,
            "candidates": [
                {
                    "chunk_id": hit.chunk_id,
                    "chunk_type": hit.chunk_type,
                    "segment_id": hit.segment_id,
                    "content": hit.content,
                    "score": hit.score,
                }
                for hit in hits
            ],
        }
        response = self._client.complete_json(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "你是检索候选重排助手。"
                        "请只返回 JSON。"
                        'JSON 结构必须为 {"ordered_ids":["..."]}。'
                        "ordered_ids 中只能填写输入候选里的 chunk_id，按相关性从高到低排序。"
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                ),
            ],
            model_name=self._model_name,
            temperature=0.0,
        )
        ordered_ids = response.get("ordered_ids", [])
        if not isinstance(ordered_ids, list):
            return []
        return [str(item).strip() for item in ordered_ids if str(item).strip()]

    def _fallback_rerank(
        self,
        *,
        query_text: str,
        hits: list[ChunkSearchHit],
    ) -> list[ChunkSearchHit]:
        """模型不可用时，退化为简单关键词得分排序。"""
        query_terms = _extract_query_terms(query_text)
        if not query_terms:
            return hits

        def _score(hit: ChunkSearchHit) -> tuple[int, float]:
            normalized_content = _normalize_text_for_matching(hit.content)
            keyword_hits = sum(1 for term in query_terms if term in normalized_content)
            return (keyword_hits, hit.score or 0.0)

        return sorted(hits, key=_score, reverse=True)

    def _reorder_hits(
        self,
        hits: list[ChunkSearchHit],
        ordered_ids: list[str],
    ) -> list[ChunkSearchHit]:
        """按模型返回顺序重排 chunk，未覆盖项按原顺序补齐。"""
        hit_map = {hit.chunk_id: hit for hit in hits}
        ordered_hits: list[ChunkSearchHit] = []
        seen_ids: set[str] = set()

        for chunk_id in ordered_ids:
            hit = hit_map.get(chunk_id)
            if hit is None or chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            ordered_hits.append(hit)

        ordered_hits.extend(hit for hit in hits if hit.chunk_id not in seen_ids)
        return ordered_hits


class ImageRerankService:
    """使用通用重排模型对图片候选做二次排序。"""

    def __init__(
        self,
        *,
        client: StructuredOutputModelClient,
        model_name: str | None,
        max_returned_images: int = 3,
    ) -> None:
        self._client = client
        self._model_name = (model_name or "").strip()
        self._max_returned_images = max_returned_images

    def rerank(
        self,
        *,
        user_query: str,
        retrieval_query: str,
        block_text: str,
        context_sections: Sequence[str],
        image_assets: list[ChunkImageAsset],
    ) -> list[ChunkImageAsset]:
        """按当前回答块上下文重排图片候选。"""
        if len(image_assets) <= 1:
            return image_assets
        if not self._model_name:
            return self._fallback_rerank(
                user_query=user_query,
                retrieval_query=retrieval_query,
                block_text=block_text,
                context_sections=context_sections,
                image_assets=image_assets,
            )[: self._max_returned_images]

        try:
            ordered_keys = self._invoke_model(
                user_query=user_query,
                retrieval_query=retrieval_query,
                block_text=block_text,
                context_sections=context_sections,
                image_assets=image_assets,
            )
        except Exception:
            return self._fallback_rerank(
                user_query=user_query,
                retrieval_query=retrieval_query,
                block_text=block_text,
                context_sections=context_sections,
                image_assets=image_assets,
            )[: self._max_returned_images]

        if not ordered_keys:
            return self._fallback_rerank(
                user_query=user_query,
                retrieval_query=retrieval_query,
                block_text=block_text,
                context_sections=context_sections,
                image_assets=image_assets,
            )[: self._max_returned_images]

        ordered_assets = self._reorder_assets(image_assets, ordered_keys)
        return ordered_assets[: self._max_returned_images]

    def _invoke_model(
        self,
        *,
        user_query: str,
        retrieval_query: str,
        block_text: str,
        context_sections: Sequence[str],
        image_assets: list[ChunkImageAsset],
    ) -> list[str]:
        """调用重排模型输出图片键顺序。"""
        payload = {
            "user_query": user_query,
            "retrieval_query": retrieval_query,
            "block_text": block_text,
            "context_sections": [section for section in context_sections if section.strip()],
            "candidates": [
                {
                    "candidate_key": self._build_asset_key(asset),
                    "segment_id": asset.segment_id,
                    "asset_id": asset.asset_id,
                    "summary": asset.summary,
                    "ocr_text": asset.ocr_text,
                    "image_type": asset.image_type,
                }
                for asset in image_assets
            ],
        }
        response = self._client.complete_json(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "你是图片候选重排助手。"
                        "请只返回 JSON。"
                        'JSON 结构必须为 {"ordered_keys":["..."]}。'
                        "ordered_keys 中只能填写输入候选里的 candidate_key，按相关性从高到低排序。"
                        "判断标准是当前回答片段最应该配哪张图，而不是只看用户问题和图片的字面相似度。"
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                ),
            ],
            model_name=self._model_name,
            temperature=0.0,
        )
        ordered_keys = response.get("ordered_keys", [])
        if not isinstance(ordered_keys, list):
            return []
        return [str(item).strip() for item in ordered_keys if str(item).strip()]

    def _fallback_rerank(
        self,
        *,
        user_query: str,
        retrieval_query: str,
        block_text: str,
        context_sections: Sequence[str],
        image_assets: list[ChunkImageAsset],
    ) -> list[ChunkImageAsset]:
        """模型不可用时，退化为简单关键词覆盖排序。"""
        context_text = " ".join(
            item for item in [user_query, retrieval_query, block_text, *context_sections] if item
        )
        query_terms = _extract_query_terms(context_text)
        if not query_terms:
            return image_assets

        def _score(asset: ChunkImageAsset) -> int:
            candidate_text = _normalize_text_for_matching(
                " ".join(
                    item
                    for item in [asset.summary, asset.ocr_text, asset.image_type]
                    if item.strip()
                )
            )
            return sum(1 for term in query_terms if term in candidate_text)

        return sorted(image_assets, key=_score, reverse=True)

    def _reorder_assets(
        self,
        image_assets: list[ChunkImageAsset],
        ordered_keys: list[str],
    ) -> list[ChunkImageAsset]:
        """按模型返回顺序重排图片资产。"""
        asset_map = {self._build_asset_key(asset): asset for asset in image_assets}
        ordered_assets: list[ChunkImageAsset] = []
        seen_keys: set[str] = set()

        for candidate_key in ordered_keys:
            asset = asset_map.get(candidate_key)
            if asset is None or candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)
            ordered_assets.append(asset)

        ordered_assets.extend(
            asset for asset in image_assets if self._build_asset_key(asset) not in seen_keys
        )
        return ordered_assets

    def _build_asset_key(self, asset: ChunkImageAsset) -> str:
        """构造图片候选的稳定键。"""
        if asset.segment_id:
            return f"{asset.segment_id}::{asset.asset_id}"
        return asset.asset_id or asset.source_anchor
