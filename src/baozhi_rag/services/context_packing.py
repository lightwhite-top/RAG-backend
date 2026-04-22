"""证据上下文打包服务。"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from baozhi_rag.services.evidence_sufficiency import EvidenceAssessment

if TYPE_CHECKING:
    from baozhi_rag.services.chat import ChatCitation


_CJK_CHAR_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ASCII_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")


@dataclass(frozen=True, slots=True)
class ContextPackingBudgetConfig:
    """定义证据上下文可使用的预算边界。

    参数:
        max_prompt_tokens: 当前模型单轮提示词窗口上限。
        occupied_prompt_tokens: 构建证据上下文前，已经被系统提示词、固定指令等占用的 token 数。
        reserved_answer_tokens: 需要为模型回答预留的 token 数。
        reserved_margin_tokens: 额外保守预留的安全边界，避免近似估算过于乐观。
        max_evidence_tokens: 可选的证据 token 上限；若为空，则按剩余窗口自动推导。
        min_evidence_tokens: 证据最小可用 token 下限，仅用于结果观测，不会强制突破窗口约束。
        approx_chars_per_token: 字符与 token 的保守近似比例，用于字符裁剪兜底。
    """

    max_prompt_tokens: int
    occupied_prompt_tokens: int = 0
    reserved_answer_tokens: int = 0
    reserved_margin_tokens: int = 0
    max_evidence_tokens: int | None = None
    min_evidence_tokens: int = 0
    approx_chars_per_token: int = 4


@dataclass(frozen=True, slots=True)
class ContextPackingBudgetSnapshot:
    """记录一次上下文打包时推导出的预算快照。"""

    max_prompt_tokens: int
    occupied_prompt_tokens: int
    history_tokens: int
    scaffold_tokens: int
    requested_evidence_tokens: int
    available_prompt_tokens: int
    available_evidence_tokens: int
    available_evidence_chars: int
    min_evidence_tokens: int
    approx_chars_per_token: int


@dataclass(frozen=True, slots=True)
class ContextPackingResult:
    """返回打包后的提示词以及本次预算消耗情况。"""

    prompt: str
    budget_snapshot: ContextPackingBudgetSnapshot | None
    included_citation_ids: list[str]
    dropped_citation_ids: list[str]
    history_tokens: int
    evidence_tokens: int
    truncated_by_budget: bool


class ContextPackingService:
    """负责把检索证据打包成模型可消费的上下文。"""

    def __init__(self, *, max_context_chars: int = 1200) -> None:
        self._max_context_chars = max_context_chars

    def build_context_prompt(
        self,
        *,
        retrieval_query: str,
        citations: list[ChatCitation],
        budget_config: ContextPackingBudgetConfig | None = None,
        history_messages: Sequence[object] | None = None,
    ) -> str:
        """把检索结果格式化为模型可消费的证据上下文。

        参数:
            retrieval_query: 当前轮最终用于检索的查询文本。
            citations: 检索命中的证据列表。
            budget_config: 可选的预算配置；为空时保持旧版按字符截断的行为。
            history_messages: 可选的历史消息列表，用于估算已占用的提示词预算。
        返回:
            可直接塞入 system prompt 的证据上下文。
        """

        return self.build_context_prompt_result(
            retrieval_query=retrieval_query,
            citations=citations,
            budget_config=budget_config,
            history_messages=history_messages,
        ).prompt

    def build_context_prompt_result(
        self,
        *,
        retrieval_query: str,
        citations: list[ChatCitation],
        budget_config: ContextPackingBudgetConfig | None = None,
        history_messages: Sequence[object] | None = None,
    ) -> ContextPackingResult:
        """构建带预算信息的证据上下文。

        参数:
            retrieval_query: 当前轮最终用于检索的查询文本。
            citations: 检索命中的证据列表。
            budget_config: 可选的预算配置；为空时保持旧版按字符截断的行为。
            history_messages: 可选的历史消息列表，用于估算历史占用并裁剪证据。
        返回:
            包含最终提示词、预算快照及证据裁剪结果的对象。
        """

        normalized_query = retrieval_query.strip()
        if not citations:
            return ContextPackingResult(
                prompt=self.build_no_knowledge_context_prompt(retrieval_query=normalized_query),
                budget_snapshot=(
                    self.calculate_budget(
                        retrieval_query=normalized_query,
                        citations=[],
                        budget_config=budget_config,
                        history_messages=history_messages,
                    )
                    if budget_config is not None
                    else None
                ),
                included_citation_ids=[],
                dropped_citation_ids=[],
                history_tokens=self._estimate_messages_token_count(history_messages),
                evidence_tokens=0,
                truncated_by_budget=False,
            )

        if budget_config is None:
            return self._build_legacy_context_prompt(
                retrieval_query=normalized_query,
                citations=citations,
                history_messages=history_messages,
            )

        budget_snapshot = self.calculate_budget(
            retrieval_query=normalized_query,
            citations=citations,
            budget_config=budget_config,
            history_messages=history_messages,
        )
        return self._build_budgeted_context_prompt(
            retrieval_query=normalized_query,
            citations=citations,
            budget_snapshot=budget_snapshot,
        )

    def calculate_budget(
        self,
        *,
        retrieval_query: str,
        citations: Sequence[ChatCitation],
        budget_config: ContextPackingBudgetConfig,
        history_messages: Sequence[object] | None = None,
    ) -> ContextPackingBudgetSnapshot:
        """计算当前证据上下文可用预算。

        参数:
            retrieval_query: 当前轮最终用于检索的查询文本。
            citations: 当前待打包的证据列表，仅用于估算脚手架开销。
            budget_config: 预算配置。
            history_messages: 可选的历史消息列表，用于估算占用的 token。
        返回:
            一次预算推导的快照，供调用方裁剪证据或记录调试信息。
        """

        history_tokens = self._estimate_messages_token_count(history_messages)
        scaffold_tokens = self._estimate_scaffold_tokens(
            retrieval_query=retrieval_query,
            citations=citations,
        )
        occupied_prompt_tokens = max(budget_config.occupied_prompt_tokens, 0)
        available_prompt_tokens = max(
            budget_config.max_prompt_tokens
            - occupied_prompt_tokens
            - history_tokens
            - max(budget_config.reserved_answer_tokens, 0)
            - max(budget_config.reserved_margin_tokens, 0),
            0,
        )
        requested_evidence_tokens = (
            max(budget_config.max_evidence_tokens, 0)
            if budget_config.max_evidence_tokens is not None
            else available_prompt_tokens
        )
        available_evidence_tokens = max(
            min(requested_evidence_tokens, available_prompt_tokens) - scaffold_tokens,
            0,
        )
        approx_chars_per_token = max(budget_config.approx_chars_per_token, 1)

        return ContextPackingBudgetSnapshot(
            max_prompt_tokens=budget_config.max_prompt_tokens,
            occupied_prompt_tokens=occupied_prompt_tokens,
            history_tokens=history_tokens,
            scaffold_tokens=scaffold_tokens,
            requested_evidence_tokens=requested_evidence_tokens,
            available_prompt_tokens=available_prompt_tokens,
            available_evidence_tokens=available_evidence_tokens,
            available_evidence_chars=available_evidence_tokens * approx_chars_per_token,
            min_evidence_tokens=max(budget_config.min_evidence_tokens, 0),
            approx_chars_per_token=approx_chars_per_token,
        )

    def build_no_knowledge_context_prompt(self, *, retrieval_query: str) -> str:
        """构造未命中知识库时的模型约束提示。"""

        return "\n\n".join(
            [
                f"用户当前问题：{retrieval_query}",
                "当前轮未检索到可引用的知识库证据。",
                (
                    "你仍然需要回答，但必须遵守以下约束："
                    "对问候、身份说明、能力介绍、通用概念解释，可以直接给出简洁中文回答；"
                    "对涉及具体规则、权限、金额计算、合规要求或业务结论等高风险问题，"
                    "不得伪造依据或给出超出证据的确定性承诺。"
                ),
                (
                    "如果当前问题需要系统记录、业务规则或知识库材料支撑，"
                    "必须明确说明“当前没有检索到可支撑结论的知识库材料”，"
                    "只能提供一般性说明，并建议补充材料或转人工核实。"
                ),
                "当前没有证据可引用，不要输出 [1][2] 这类引用编号。",
            ]
        )

    def build_evidence_guidance_prompt(
        self,
        *,
        evidence_assessment: EvidenceAssessment | None,
    ) -> str:
        """根据证据充分性判断补充保守回答约束。"""

        if evidence_assessment is None or evidence_assessment.sufficient:
            return ""
        return "\n".join(
            [
                "当前证据充分性判断：证据不足。",
                "回答时必须保持保守，只能给出一般性说明，不得输出超出证据的确定性结论。",
                f"不足原因：{evidence_assessment.reason_code}",
            ]
        )

    def estimate_token_count(self, text: str) -> int:
        """对文本做保守 token 近似估算。

        说明:
            当前仓库尚未统一接入 tokenizer，因此这里优先采用保守近似：
            - 中文字符按 1 token 估算；
            - 英文/数字串按每 3 个字符约 1 token 向上取整；
            - 标点额外计入，避免低估。
            同时保留通用字符兜底 `len(text) / 4`，最终取更保守的一侧。
        """

        normalized = " ".join(text.split())
        if not normalized:
            return 0

        cjk_count = len(_CJK_CHAR_PATTERN.findall(normalized))
        ascii_token_count = sum(
            math.ceil(len(match.group(0)) / 3) for match in _ASCII_WORD_PATTERN.finditer(normalized)
        )
        punctuation_count = len(_PUNCTUATION_PATTERN.findall(normalized))
        char_fallback = math.ceil(len(normalized) / 4)
        return max(cjk_count + ascii_token_count + punctuation_count, char_fallback, 1)

    def truncate_content(self, content: str, *, max_chars: int | None = None) -> str:
        """限制单条证据进入提示词的长度，避免上下文失控。

        参数:
            content: 原始证据正文。
            max_chars: 可选的字符上限；为空时沿用默认单条证据上限。
        返回:
            归一化并按上限截断后的正文。
        """

        normalized = " ".join(content.split())
        limit = self._max_context_chars if max_chars is None else max(max_chars, 0)
        if len(normalized) <= limit:
            return normalized
        if limit <= 3:
            return "." * limit
        return f"{normalized[:limit]}..."

    def _build_legacy_context_prompt(
        self,
        *,
        retrieval_query: str,
        citations: list[ChatCitation],
        history_messages: Sequence[object] | None,
    ) -> ContextPackingResult:
        """保持旧版按单条证据字符截断的兼容行为。"""

        sections = self._build_prompt_prefix(retrieval_query)
        for index, citation in enumerate(citations, start=1):
            sections.append(
                self._format_citation_section(
                    index=index,
                    citation=citation,
                    content=self.truncate_content(citation.content),
                )
            )
        sections.append(self._build_prompt_suffix())

        return ContextPackingResult(
            prompt="\n\n".join(sections),
            budget_snapshot=None,
            included_citation_ids=[citation.citation_id for citation in citations],
            dropped_citation_ids=[],
            history_tokens=self._estimate_messages_token_count(history_messages),
            evidence_tokens=sum(
                self.estimate_token_count(self.truncate_content(citation.content))
                for citation in citations
            ),
            truncated_by_budget=False,
        )

    def _build_budgeted_context_prompt(
        self,
        *,
        retrieval_query: str,
        citations: list[ChatCitation],
        budget_snapshot: ContextPackingBudgetSnapshot,
    ) -> ContextPackingResult:
        """按预算裁剪证据正文并构造最终提示词。"""

        sections = self._build_prompt_prefix(retrieval_query)
        included_citation_ids: list[str] = []
        remaining_citation_ids = [citation.citation_id for citation in citations]
        remaining_tokens = budget_snapshot.available_evidence_tokens
        truncated_by_budget = False
        section_joiner_tokens = self.estimate_token_count("\n\n")

        for index, citation in enumerate(citations, start=1):
            prefix = self._build_citation_prefix(index=index, citation=citation)
            prefix_tokens = self.estimate_token_count(prefix)
            if remaining_tokens <= prefix_tokens:
                truncated_by_budget = True
                break

            content_budget_chars = min(
                self._max_context_chars,
                max((remaining_tokens - prefix_tokens) * budget_snapshot.approx_chars_per_token, 0),
            )
            if content_budget_chars <= 0:
                truncated_by_budget = True
                break

            content = self.truncate_content(citation.content, max_chars=content_budget_chars)
            citation_section = self._format_citation_section(
                index=index,
                citation=citation,
                content=content,
            )
            citation_tokens = self.estimate_token_count(citation_section) + section_joiner_tokens

            while citation_tokens > remaining_tokens and content_budget_chars > 3:
                # 近似 token 估算不可避免存在偏差，超预算时继续按字符收缩做保守兜底。
                content_budget_chars = max(
                    content_budget_chars - budget_snapshot.approx_chars_per_token,
                    3,
                )
                content = self.truncate_content(citation.content, max_chars=content_budget_chars)
                citation_section = self._format_citation_section(
                    index=index,
                    citation=citation,
                    content=content,
                )
                citation_tokens = (
                    self.estimate_token_count(citation_section) + section_joiner_tokens
                )

            if citation_tokens > remaining_tokens:
                truncated_by_budget = True
                break

            sections.append(citation_section)
            included_citation_ids.append(citation.citation_id)
            remaining_citation_ids.pop(0)
            remaining_tokens = max(remaining_tokens - citation_tokens, 0)

        sections.append(self._build_prompt_suffix())
        evidence_tokens = max(budget_snapshot.available_evidence_tokens - remaining_tokens, 0)
        if not included_citation_ids and citations:
            truncated_by_budget = True

        return ContextPackingResult(
            prompt="\n\n".join(sections),
            budget_snapshot=budget_snapshot,
            included_citation_ids=included_citation_ids,
            dropped_citation_ids=remaining_citation_ids,
            history_tokens=budget_snapshot.history_tokens,
            evidence_tokens=evidence_tokens,
            truncated_by_budget=truncated_by_budget,
        )

    def _build_prompt_prefix(self, retrieval_query: str) -> list[str]:
        """构造上下文提示词开头，便于预算估算与正文拼装保持一致。"""

        return [
            f"用户当前问题：{retrieval_query}",
            "以下是可引用的知识库证据：",
        ]

    def _build_prompt_suffix(self) -> str:
        """构造上下文提示词结尾约束。"""

        return "\n".join(
            [
                "请只基于上述证据回答，不要引用未提供的外部知识。",
                "如果证据已经包含足以回答当前问题的事实，请先直接给出结论，再补充必要条件。",
                "不要在已有证据时误报“未检索到”或“未提及”。",
            ]
        )

    def _build_citation_prefix(self, *, index: int, citation: ChatCitation) -> str:
        """构造单条证据在正文之前的元信息区域。"""

        score_text = str(citation.score) if citation.score is not None else "null"
        return "\n".join(
            [
                f"[{index}] 文件：{citation.source_filename}",
                f"chunk_id：{citation.chunk_id}",
                f"chunk_index：{citation.chunk_index}",
                f"score：{score_text}",
                "内容：",
            ]
        )

    def _format_citation_section(
        self,
        *,
        index: int,
        citation: ChatCitation,
        content: str,
    ) -> str:
        """把单条证据组装成最终提示词片段。"""

        return f"{self._build_citation_prefix(index=index, citation=citation)}{content}"

    def _estimate_scaffold_tokens(
        self,
        *,
        retrieval_query: str,
        citations: Sequence[ChatCitation],
    ) -> int:
        """估算除证据正文外的固定提示词开销。"""

        prompt_prefix_text = "\n\n".join(self._build_prompt_prefix(retrieval_query))
        prompt_suffix_text = self._build_prompt_suffix()
        return (
            self.estimate_token_count(prompt_prefix_text)
            + self.estimate_token_count(prompt_suffix_text)
            + self.estimate_token_count("\n\n")
        )

    def _estimate_messages_token_count(self, messages: Sequence[object] | None) -> int:
        """估算历史消息列表占用的 token。"""

        if not messages:
            return 0

        total_tokens = 0
        for message in messages:
            content = self._extract_message_content(message)
            if not content:
                continue
            total_tokens += self.estimate_token_count(content)
        return total_tokens

    def _extract_message_content(self, message: object) -> str:
        """从消息对象中抽取可估算 token 的文本内容。"""

        if isinstance(message, str):
            return message

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return str(content) if content is not None else ""
