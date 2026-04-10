"""证据上下文打包服务。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from baozhi_rag.services.evidence_sufficiency import EvidenceAssessment

if TYPE_CHECKING:
    from baozhi_rag.services.chat import ChatCitation


class ContextPackingService:
    """负责把检索证据打包成模型可消费的上下文。"""

    def __init__(self, *, max_context_chars: int = 1200) -> None:
        self._max_context_chars = max_context_chars

    def build_context_prompt(
        self,
        *,
        retrieval_query: str,
        citations: list[ChatCitation],
    ) -> str:
        """把检索结果格式化为模型可消费的证据上下文。"""
        if not citations:
            return self.build_no_knowledge_context_prompt(retrieval_query=retrieval_query)

        sections = [f"用户当前问题：{retrieval_query}", "以下是可引用的知识库证据："]

        for index, citation in enumerate(citations, start=1):
            sections.append(
                "\n".join(
                    [
                        f"[{index}] 文件：{citation.source_filename}",
                        f"chunk_id：{citation.chunk_id}",
                        f"chunk_index：{citation.chunk_index}",
                        f"score：{citation.score if citation.score is not None else 'null'}",
                        f"内容：{self.truncate_content(citation.content)}",
                    ]
                )
            )

        sections.append("请只基于上述证据回答，不要引用未提供的外部知识。")
        return "\n\n".join(sections)

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

    def truncate_content(self, content: str) -> str:
        """限制单条证据进入提示词的长度，避免上下文失控。"""
        normalized = " ".join(content.split())
        if len(normalized) <= self._max_context_chars:
            return normalized
        return f"{normalized[: self._max_context_chars]}..."
