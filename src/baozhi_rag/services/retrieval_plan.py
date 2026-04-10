"""检索计划与多 lane 编排规则。"""

from __future__ import annotations

from dataclasses import dataclass


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
    ) -> RetrievalPlan:
        """构造检索计划。"""
        candidate_scale = 2 if mode == "chat" else 1
        lexical_candidate_size = self._lexical_candidate_size * candidate_scale
        vector_candidate_size = self._vector_candidate_size * candidate_scale

        lanes = [
            RetrievalLanePlan(
                lane_id="hybrid-default",
                query_text=query_text,
                lane_weight=1.0,
                result_size=result_size,
                lexical_candidate_size=lexical_candidate_size,
                vector_candidate_size=vector_candidate_size,
                query_intent=query_intent,
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
                )
            )

        return RetrievalPlan(
            mode=mode,
            query_text=query_text,
            query_intent=query_intent,
            result_size=result_size,
            lanes=lanes,
        )
