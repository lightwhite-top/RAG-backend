"""统一查询扩展策略与检索计划编排。"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from baozhi_rag.infra.llm.openai_compatible_client import (
    OpenAICompatibleLlmClient,
    OpenAICompatibleLlmError,
)
from baozhi_rag.services.llm import ChatMessage
from baozhi_rag.services.retrieval_plan import (
    RetrievalLanePlan,
    RetrievalLaneSeed,
    RetrievalPlan,
    RetrievalPlanner,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExpansionPlannerRequest:
    """扩展检索编排输入。

    参数:
        query_text: 原始查询文本。
        query_intent: 查询意图标签。
        mode: 检索模式，例如 search/chat。
        size: 上层请求的最终返回条数。
    """

    query_text: str
    query_intent: str
    mode: str
    size: int


@dataclass(frozen=True, slots=True)
class ExpansionCandidate:
    """扩展策略生成的规范化候选。

    参数:
        strategy_name: 候选来源策略名称。
        query_text: lane 的词法查询文本。
        embedding_source_text: lane 的向量查询文本。
        lane_weight: lane 融合权重。
        vector_only: 是否仅用于向量检索输入。
    """

    strategy_name: str
    query_text: str
    embedding_source_text: str
    lane_weight: float
    vector_only: bool = False


@dataclass(frozen=True, slots=True)
class ExpansionLaneInput:
    """可直接转换为 RetrievalLanePlan 的输入对象。

    参数:
        lane_id: lane 唯一标识。
        query_text: lane 查询文本（用于 trace 与显示）。
        lexical_query_text: lane 的词法查询文本。
        embedding_source_text: lane 的向量查询文本。
        source_strategy: lane 来源策略。
        source_query: lane 对应的原始问题。
        strategy_label: 人类可读策略标签。
        lane_weight: lane 融合权重。
        lexical_candidate_size: 词法候选池大小。
        vector_candidate_size: 向量候选池大小。
    """

    lane_id: str
    query_text: str
    lexical_query_text: str
    embedding_source_text: str
    source_strategy: str
    source_query: str
    strategy_label: str
    lane_weight: float
    lexical_candidate_size: int
    vector_candidate_size: int


class RetrievalExpansionStrategy(Protocol):
    """统一扩展策略协议。"""

    strategy_name: str

    def expand(self, request: ExpansionPlannerRequest) -> list[ExpansionCandidate]:
        """按请求生成扩展候选。"""


class MultiQueryExpansionStrategy:
    """MQE（Multi-Query Expansion）策略。"""

    strategy_name = "mqe"

    def __init__(
        self,
        *,
        client: OpenAICompatibleLlmClient,
        enabled: bool,
        model_name: str,
        generate_count: int,
        lane_weight: float,
        allowed_modes: Sequence[str],
        max_query_length: int,
    ) -> None:
        """初始化 MQE 策略。

        参数:
            client: OpenAI 兼容客户端。
            enabled: 是否启用该策略。
            model_name: 结构化生成模型名。
            generate_count: 目标扩展条数。
            lane_weight: lane 融合权重。
            allowed_modes: 允许启用的模式列表。
            max_query_length: 单条扩展查询最大长度。
        """
        self._client = client
        self._enabled = enabled
        self._model_name = model_name.strip()
        self._generate_count = max(1, generate_count)
        self._lane_weight = max(0.0, lane_weight)
        self._allowed_modes = _normalize_mode_set(allowed_modes)
        self._max_query_length = max(8, max_query_length)

    def expand(self, request: ExpansionPlannerRequest) -> list[ExpansionCandidate]:
        """生成多个扩展查询；失败时返回空列表。"""
        if not self._enabled or self._lane_weight <= 0.0:
            return []
        if not self._is_mode_allowed(request.mode):
            return []
        if not self._model_name:
            LOGGER.warning("mqe_expand_skipped reason=model_missing")
            return []

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "你是检索扩展助手。请输出 JSON 对象，必须包含 queries 字段。"
                    "queries 是字符串数组，数组元素应该是与原问题语义相关的可检索改写。"
                    "不要解释，不要输出 JSON 之外内容。"
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"原始问题：{request.query_text}\n"
                    f"查询意图：{request.query_intent}\n"
                    f"请生成不超过 {self._generate_count} 条扩展查询。"
                ),
            ),
        ]

        try:
            payload = self._client.complete_json(
                messages,
                model_name=self._model_name,
                temperature=0.2,
            )
        except OpenAICompatibleLlmError as exc:
            LOGGER.warning("mqe_expand_failed fallback=original_query error=%s", type(exc).__name__)
            return []
        except Exception:
            LOGGER.exception("mqe_expand_failed fallback=original_query")
            return []

        normalized_queries = self._normalize_queries(
            payload=payload,
            original_query=request.query_text,
        )
        return [
            ExpansionCandidate(
                strategy_name=self.strategy_name,
                query_text=query_text,
                embedding_source_text=query_text,
                lane_weight=self._lane_weight,
                vector_only=False,
            )
            for query_text in normalized_queries
        ]

    def _is_mode_allowed(self, mode: str) -> bool:
        """判断当前模式是否允许启用 MQE。"""
        return mode.strip().lower() in self._allowed_modes

    def _normalize_queries(
        self,
        *,
        payload: Mapping[str, object],
        original_query: str,
    ) -> list[str]:
        """归一化 MQE 输出，执行去空、去重和长度控制。"""
        raw_values = self._extract_query_values(payload)
        original_signature = _to_signature(original_query)
        seen_signatures = {original_signature} if original_signature else set()
        normalized_queries: list[str] = []

        for raw_value in raw_values:
            raw_query = self._coerce_query_text(raw_value)
            normalized_query = _normalize_text(raw_query, max_length=self._max_query_length)
            if not normalized_query:
                continue

            signature = _to_signature(normalized_query)
            if not signature or signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            normalized_queries.append(normalized_query)

            if len(normalized_queries) >= self._generate_count:
                break

        return normalized_queries

    @staticmethod
    def _extract_query_values(payload: Mapping[str, object]) -> list[object]:
        """从模型 JSON 结果中提取候选数组。"""
        for key in ("queries", "expanded_queries", "query_candidates"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        data_value = payload.get("data")
        if isinstance(data_value, Mapping):
            nested_queries = data_value.get("queries")
            if isinstance(nested_queries, list):
                return nested_queries
        return []

    @staticmethod
    def _coerce_query_text(raw_value: object) -> str:
        """把模型输出值转换为查询文本。"""
        if isinstance(raw_value, str):
            return raw_value
        if isinstance(raw_value, Mapping):
            for key in ("query", "text", "value"):
                value = raw_value.get(key)
                if isinstance(value, str):
                    return value
        return ""


class HydeExpansionStrategy:
    """HyDE（Hypothetical Document Embedding）策略。"""

    strategy_name = "hyde"

    def __init__(
        self,
        *,
        client: OpenAICompatibleLlmClient,
        enabled: bool,
        model_name: str,
        lane_weight: float,
        allowed_modes: Sequence[str],
        max_document_length: int,
    ) -> None:
        """初始化 HyDE 策略。

        参数:
            client: OpenAI 兼容客户端。
            enabled: 是否启用该策略。
            model_name: 结构化生成模型名。
            lane_weight: lane 融合权重。
            allowed_modes: 允许启用的模式列表。
            max_document_length: 假设文档最大长度。
        """
        self._client = client
        self._enabled = enabled
        self._model_name = model_name.strip()
        self._lane_weight = max(0.0, lane_weight)
        self._allowed_modes = _normalize_mode_set(allowed_modes)
        self._max_document_length = max(64, max_document_length)

    def expand(self, request: ExpansionPlannerRequest) -> list[ExpansionCandidate]:
        """生成 HyDE 假设文档；失败时返回空列表。"""
        if not self._enabled or self._lane_weight <= 0.0:
            return []
        if not self._is_mode_allowed(request.mode):
            return []
        if not self._model_name:
            LOGGER.warning("hyde_expand_skipped reason=model_missing")
            return []

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "你是 HyDE 生成助手。请输出 JSON 对象，包含 hypothetical_document 字段。"
                    "内容应是可用于向量检索的简洁假设答案，避免无关扩写。"
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"原始问题：{request.query_text}\n"
                    f"查询意图：{request.query_intent}\n"
                    "请生成一段可用于语义检索的假设文档。"
                ),
            ),
        ]

        try:
            payload = self._client.complete_json(
                messages,
                model_name=self._model_name,
                temperature=0.1,
            )
        except OpenAICompatibleLlmError as exc:
            LOGGER.warning(
                "hyde_expand_failed fallback=original_query error=%s", type(exc).__name__
            )
            return []
        except Exception:
            LOGGER.exception("hyde_expand_failed fallback=original_query")
            return []

        hypothetical_text = self._extract_hypothetical_text(payload)
        normalized_text = _normalize_text(
            hypothetical_text,
            max_length=self._max_document_length,
        )
        if not normalized_text:
            return []

        return [
            ExpansionCandidate(
                strategy_name=self.strategy_name,
                query_text=request.query_text,
                embedding_source_text=normalized_text,
                lane_weight=self._lane_weight,
                vector_only=True,
            )
        ]

    def _is_mode_allowed(self, mode: str) -> bool:
        """判断当前模式是否允许启用 HyDE。"""
        return mode.strip().lower() in self._allowed_modes

    @staticmethod
    def _extract_hypothetical_text(payload: Mapping[str, object]) -> str:
        """从模型 JSON 结果中提取假设文档内容。"""
        for key in (
            "hypothetical_document",
            "hypothetical_answer",
            "answer",
            "document",
            "text",
        ):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return ""


class RetrievalExpansionOrchestrator:
    """扩展策略统一编排器。"""

    def __init__(
        self,
        *,
        enabled: bool,
        lexical_candidate_size: int,
        vector_candidate_size: int,
        candidate_pool_multiplier: float,
        strategies: Sequence[RetrievalExpansionStrategy],
    ) -> None:
        """初始化扩展编排器。

        参数:
            enabled: 是否启用统一扩展框架。
            lexical_candidate_size: 基础词法候选池大小。
            vector_candidate_size: 基础向量候选池大小。
            candidate_pool_multiplier: 扩展 lane 候选池倍率。
            strategies: 参与编排的扩展策略集合。
        """
        self._enabled = enabled
        self._lexical_candidate_size = max(0, lexical_candidate_size)
        self._vector_candidate_size = max(0, vector_candidate_size)
        self._candidate_pool_multiplier = max(0.1, candidate_pool_multiplier)
        self._strategies = list(strategies)

    def build_lane_inputs(
        self,
        *,
        query_text: str,
        query_intent: str,
        mode: str,
        size: int,
    ) -> list[ExpansionLaneInput]:
        """按统一入口构造可供 planner 消费的扩展 lane 输入。"""
        if not self._enabled:
            return []

        request = ExpansionPlannerRequest(
            query_text=query_text,
            query_intent=query_intent,
            mode=mode,
            size=size,
        )
        base_lexical_candidate_size = self._resolve_mode_candidate_size(
            base_size=self._lexical_candidate_size,
            mode=mode,
        )
        base_vector_candidate_size = self._resolve_mode_candidate_size(
            base_size=self._vector_candidate_size,
            mode=mode,
        )

        lane_inputs: list[ExpansionLaneInput] = []
        strategy_lane_index: dict[str, int] = {}
        seen_query_signatures: set[str] = set()

        for strategy in self._strategies:
            candidates = strategy.expand(request)
            for candidate in candidates:
                lexical_query_text = _normalize_text(candidate.query_text, max_length=2048)
                embedding_source_text = _normalize_text(
                    candidate.embedding_source_text,
                    max_length=4096,
                )
                if not lexical_query_text or not embedding_source_text:
                    continue
                signature = _to_signature(f"{lexical_query_text}|{embedding_source_text}")
                if not signature or signature in seen_query_signatures:
                    continue

                seen_query_signatures.add(signature)
                lane_index = strategy_lane_index.get(candidate.strategy_name, 0) + 1
                strategy_lane_index[candidate.strategy_name] = lane_index

                lexical_candidate_size = (
                    0
                    if candidate.vector_only
                    else self._scale_candidate_size(base_lexical_candidate_size)
                )
                vector_candidate_size = self._scale_candidate_size(base_vector_candidate_size)
                if lexical_candidate_size <= 0 and vector_candidate_size <= 0:
                    continue

                lane_inputs.append(
                    ExpansionLaneInput(
                        lane_id=f"exp-{candidate.strategy_name}-{lane_index}",
                        query_text=lexical_query_text,
                        lexical_query_text=lexical_query_text,
                        embedding_source_text=embedding_source_text,
                        source_strategy=candidate.strategy_name,
                        source_query=query_text,
                        strategy_label=candidate.strategy_name,
                        lane_weight=max(0.0, candidate.lane_weight),
                        lexical_candidate_size=lexical_candidate_size,
                        vector_candidate_size=vector_candidate_size,
                    )
                )

        return lane_inputs

    def _resolve_mode_candidate_size(self, *, base_size: int, mode: str) -> int:
        """根据检索模式计算基础候选池大小。"""
        candidate_scale = 2 if mode.strip().lower() == "chat" else 1
        return max(0, base_size * candidate_scale)

    def _scale_candidate_size(self, candidate_size: int) -> int:
        """按倍率缩放候选池大小。"""
        if candidate_size <= 0:
            return 0
        scaled_size = int(round(candidate_size * self._candidate_pool_multiplier))
        return max(1, scaled_size)


class ExpansionAwareRetrievalPlanner(RetrievalPlanner):
    """在默认 RetrievalPlanner 之上叠加 expansion lane。"""

    def __init__(
        self,
        *,
        lexical_candidate_size: int,
        vector_candidate_size: int,
        lexical_rrf_weight: float,
        vector_rrf_weight: float,
        expansion_orchestrator: RetrievalExpansionOrchestrator,
    ) -> None:
        """初始化带 expansion 编排能力的 planner。"""
        super().__init__(
            lexical_candidate_size=lexical_candidate_size,
            vector_candidate_size=vector_candidate_size,
            lexical_rrf_weight=lexical_rrf_weight,
            vector_rrf_weight=vector_rrf_weight,
        )
        self._expansion_orchestrator = expansion_orchestrator

    def build(
        self,
        *,
        query_text: str,
        query_intent: str,
        result_size: int,
        mode: str,
        lane_seeds: Sequence[RetrievalLaneSeed] | None = None,
    ) -> RetrievalPlan:
        """构造计划并附加扩展 lane。"""
        base_plan = super().build(
            query_text=query_text,
            query_intent=query_intent,
            result_size=result_size,
            mode=mode,
            lane_seeds=lane_seeds,
        )

        expansion_lane_inputs = self._expansion_orchestrator.build_lane_inputs(
            query_text=query_text,
            query_intent=query_intent,
            mode=mode,
            size=result_size,
        )
        if not expansion_lane_inputs:
            return base_plan

        merged_lanes = list(base_plan.lanes)
        lane_ids = {lane.lane_id for lane in merged_lanes}
        for lane_input in expansion_lane_inputs:
            if lane_input.lane_id in lane_ids or lane_input.lane_weight <= 0.0:
                continue

            merged_lanes.append(
                RetrievalLanePlan(
                    lane_id=lane_input.lane_id,
                    query_text=lane_input.query_text,
                    lane_weight=lane_input.lane_weight,
                    result_size=result_size,
                    lexical_candidate_size=lane_input.lexical_candidate_size,
                    vector_candidate_size=lane_input.vector_candidate_size,
                    query_intent=query_intent,
                    source_strategy=lane_input.source_strategy,
                    lexical_query_text=lane_input.lexical_query_text,
                    embedding_source_text=lane_input.embedding_source_text,
                    expansion_id=lane_input.lane_id,
                    source_query=lane_input.source_query,
                    strategy_label=lane_input.strategy_label,
                )
            )
            lane_ids.add(lane_input.lane_id)

        return RetrievalPlan(
            mode=base_plan.mode,
            query_text=base_plan.query_text,
            query_intent=base_plan.query_intent,
            result_size=base_plan.result_size,
            lanes=merged_lanes,
        )


def _normalize_mode_set(modes: Sequence[str]) -> set[str]:
    """把模式列表归一化为去重后的小写集合。"""
    normalized_modes = {mode.strip().lower() for mode in modes if mode.strip()}
    return normalized_modes or {"search", "chat"}


def _normalize_text(text: str, *, max_length: int) -> str:
    """清理文本空白并执行长度限制。"""
    normalized_text = " ".join(text.split()).strip()
    if not normalized_text:
        return ""
    return normalized_text[:max_length].strip()


def _to_signature(text: str) -> str:
    """把文本转换为大小写不敏感的去重签名。"""
    return text.strip().casefold()
