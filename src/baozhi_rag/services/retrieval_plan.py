"""检索计划与多 lane 编排规则。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _normalize_optional_text(value: str | None) -> str | None:
    """归一化可选文本，空白字符串视为 None。"""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


@dataclass(frozen=True, slots=True)
class RetrievalLaneSeed:
    """上游传入的 lane 种子输入。"""

    lane_id: str
    source_strategy: str = "original"
    lexical_query_text: str | None = None
    embedding_source_text: str | None = None
    expansion_id: str | None = None
    source_query: str | None = None
    strategy_label: str | None = None
    lane_weight: float | None = None
    lexical_candidate_size: int | None = None
    vector_candidate_size: int | None = None
    lexical_rrf_weight: float | None = None
    vector_rrf_weight: float | None = None
    query_intent: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalLanePlan:
    """单条检索 lane 的执行计划。"""

    lane_id: str
    query_text: str
    lane_weight: float
    result_size: int
    lexical_candidate_size: int
    vector_candidate_size: int
    lexical_rrf_weight: float | None = None
    vector_rrf_weight: float | None = None
    query_intent: str = "general"
    source_strategy: str = "original"
    lexical_query_text: str | None = None
    embedding_source_text: str | None = None
    expansion_id: str | None = None
    source_query: str | None = None
    strategy_label: str | None = None

    def __post_init__(self) -> None:
        """归一化 lane 的查询与来源元信息。"""
        lane_query_text = self.query_text
        lexical_query_text = _normalize_optional_text(self.lexical_query_text) or lane_query_text
        embedding_source_text = (
            _normalize_optional_text(self.embedding_source_text) or lexical_query_text
        )
        source_strategy = _normalize_optional_text(self.source_strategy) or "original"
        source_query = _normalize_optional_text(self.source_query) or lane_query_text
        strategy_label = _normalize_optional_text(self.strategy_label) or source_strategy
        expansion_id = _normalize_optional_text(self.expansion_id)

        object.__setattr__(self, "source_strategy", source_strategy)
        object.__setattr__(self, "lexical_query_text", lexical_query_text)
        object.__setattr__(self, "embedding_source_text", embedding_source_text)
        object.__setattr__(self, "source_query", source_query)
        object.__setattr__(self, "strategy_label", strategy_label)
        object.__setattr__(self, "expansion_id", expansion_id)


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """完整检索计划。"""

    mode: str
    query_text: str
    query_intent: str
    result_size: int
    lanes: list[RetrievalLanePlan]


class RetrievalPlanner:
    """根据模式和查询意图生成多 lane 检索计划。"""

    def __init__(
        self,
        *,
        lexical_candidate_size: int,
        vector_candidate_size: int,
        lexical_rrf_weight: float,
        vector_rrf_weight: float,
    ) -> None:
        self._lexical_candidate_size = max(1, lexical_candidate_size)
        self._vector_candidate_size = max(1, vector_candidate_size)
        self._lexical_rrf_weight = max(0.0, lexical_rrf_weight)
        self._vector_rrf_weight = max(0.0, vector_rrf_weight)

    def build(
        self,
        *,
        query_text: str,
        query_intent: str,
        result_size: int,
        mode: str,
        lane_seeds: Sequence[RetrievalLaneSeed] | None = None,
    ) -> RetrievalPlan:
        """构造检索计划。"""
        candidate_scale = 2 if mode == "chat" else 1
        lexical_candidate_size = self._lexical_candidate_size * candidate_scale
        vector_candidate_size = self._vector_candidate_size * candidate_scale

        lanes: list[RetrievalLanePlan] = []
        if lane_seeds:
            lanes = self._build_seed_lanes(
                lane_seeds=lane_seeds,
                query_text=query_text,
                query_intent=query_intent,
                result_size=result_size,
                lexical_candidate_size=lexical_candidate_size,
                vector_candidate_size=vector_candidate_size,
            )
        if not lanes:
            lanes = [
                RetrievalLanePlan(
                    lane_id="hybrid-default",
                    query_text=query_text,
                    lane_weight=1.0,
                    result_size=result_size,
                    lexical_candidate_size=lexical_candidate_size,
                    vector_candidate_size=vector_candidate_size,
                    query_intent=query_intent,
                    source_strategy="original",
                    source_query=query_text,
                    strategy_label="original",
                ),
                RetrievalLanePlan(
                    lane_id="lexical-focus",
                    query_text=query_text,
                    lane_weight=0.9,
                    result_size=result_size,
                    lexical_candidate_size=int(lexical_candidate_size * 1.25),
                    vector_candidate_size=0,
                    lexical_rrf_weight=self._lexical_rrf_weight * 1.25,
                    vector_rrf_weight=0.0,
                    query_intent=query_intent,
                    source_strategy="original",
                    source_query=query_text,
                    strategy_label="original",
                ),
                RetrievalLanePlan(
                    lane_id="semantic-focus",
                    query_text=query_text,
                    lane_weight=0.9,
                    result_size=result_size,
                    lexical_candidate_size=0,
                    vector_candidate_size=int(vector_candidate_size * 1.25),
                    lexical_rrf_weight=0.0,
                    vector_rrf_weight=self._vector_rrf_weight * 1.2,
                    query_intent=query_intent,
                    source_strategy="original",
                    source_query=query_text,
                    strategy_label="original",
                ),
            ]

            if query_intent in {"document_location", "structured", "procedure"}:
                lanes.append(
                    RetrievalLanePlan(
                        lane_id="structure-focus",
                        query_text=query_text,
                        lane_weight=0.85,
                        result_size=result_size,
                        lexical_candidate_size=int(lexical_candidate_size * 1.4),
                        vector_candidate_size=max(1, int(vector_candidate_size * 0.6)),
                        lexical_rrf_weight=self._lexical_rrf_weight * 1.3,
                        vector_rrf_weight=self._vector_rrf_weight * 0.8,
                        query_intent=query_intent,
                        source_strategy="original",
                        source_query=query_text,
                        strategy_label="original",
                    )
                )

        return RetrievalPlan(
            mode=mode,
            query_text=query_text,
            query_intent=query_intent,
            result_size=result_size,
            lanes=lanes,
        )

    def _build_seed_lanes(
        self,
        *,
        lane_seeds: Sequence[RetrievalLaneSeed],
        query_text: str,
        query_intent: str,
        result_size: int,
        lexical_candidate_size: int,
        vector_candidate_size: int,
    ) -> list[RetrievalLanePlan]:
        """根据上游 seed 输入构造 lane 计划。"""
        lanes: list[RetrievalLanePlan] = []
        for index, lane_seed in enumerate(lane_seeds, start=1):
            lane_id = _normalize_optional_text(lane_seed.lane_id) or f"seed-{index}"
            lexical_query_text = (
                _normalize_optional_text(lane_seed.lexical_query_text) or query_text
            )
            embedding_source_text = (
                _normalize_optional_text(lane_seed.embedding_source_text) or lexical_query_text
            )
            source_query = _normalize_optional_text(lane_seed.source_query) or query_text
            source_strategy = _normalize_optional_text(lane_seed.source_strategy) or "original"
            strategy_label = _normalize_optional_text(lane_seed.strategy_label) or source_strategy
            lane_weight = max(
                0.0,
                lane_seed.lane_weight if lane_seed.lane_weight is not None else 1.0,
            )
            lane_lexical_size = self._resolve_candidate_size(
                lane_seed.lexical_candidate_size,
                lexical_candidate_size,
            )
            lane_vector_size = self._resolve_candidate_size(
                lane_seed.vector_candidate_size,
                vector_candidate_size,
            )
            lexical_rrf_weight = lane_seed.lexical_rrf_weight
            vector_rrf_weight = lane_seed.vector_rrf_weight
            if lexical_rrf_weight is None and lane_lexical_size > 0:
                lexical_rrf_weight = self._lexical_rrf_weight
            if vector_rrf_weight is None and lane_vector_size > 0:
                vector_rrf_weight = self._vector_rrf_weight

            lanes.append(
                RetrievalLanePlan(
                    lane_id=lane_id,
                    query_text=lexical_query_text,
                    lane_weight=lane_weight,
                    result_size=result_size,
                    lexical_candidate_size=lane_lexical_size,
                    vector_candidate_size=lane_vector_size,
                    lexical_rrf_weight=lexical_rrf_weight,
                    vector_rrf_weight=vector_rrf_weight,
                    query_intent=lane_seed.query_intent or query_intent,
                    source_strategy=source_strategy,
                    lexical_query_text=lexical_query_text,
                    embedding_source_text=embedding_source_text,
                    expansion_id=lane_seed.expansion_id,
                    source_query=source_query,
                    strategy_label=strategy_label,
                )
            )
        return lanes

    def _resolve_candidate_size(self, override_value: int | None, default_value: int) -> int:
        """合并 lane 的候选规模配置。"""
        if override_value is None:
            return max(0, default_value)
        return max(0, override_value)
