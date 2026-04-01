"""大模型能力抽象。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """标准化聊天消息。"""

    role: Literal["system", "user", "assistant"]
    content: str


class EmbeddingModelClient(Protocol):
    """文本向量模型客户端抽象。"""

    def ensure_ready(self) -> None:
        """校验客户端是否可用于后续模型调用。"""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """把输入文本批量转换为向量。"""
        ...


class ChatModelClient(Protocol):
    """聊天模型客户端抽象。"""

    def ensure_ready(self) -> None:
        """校验客户端是否可用于后续模型调用。"""
        ...

    def complete_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> str:
        """执行一次聊天补全并返回文本结果。"""
        ...

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """执行一次流式聊天补全，并按顺序返回文本增量。"""
        ...
