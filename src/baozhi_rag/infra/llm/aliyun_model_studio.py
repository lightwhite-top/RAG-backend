"""阿里云百炼 OpenAI 兼容客户端封装。"""

from __future__ import annotations

from typing import Any, cast

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.llm import ChatMessage

try:  # pragma: no cover - 是否安装依赖取决于运行环境
    from openai import OpenAI as ImportedOpenAIClient  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - 测试环境可通过可选导入绕过
    OPENAI_CLIENT_CLASS: Any | None = None
    OPENAI_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - 导入成功路径不需要单独覆盖
    OPENAI_CLIENT_CLASS = ImportedOpenAIClient
    OPENAI_IMPORT_ERROR = None


class AlibabaModelStudioError(AppError):
    """阿里云百炼客户端异常。"""

    default_message = "阿里云百炼调用失败"
    default_error_code = "bailian_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class AlibabaModelStudioDependencyError(AlibabaModelStudioError):
    """百炼客户端依赖缺失。"""

    default_message = "百炼客户端依赖缺失"
    default_error_code = "bailian_dependency_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class AlibabaModelStudioConfigurationError(AlibabaModelStudioError):
    """百炼客户端配置异常。"""

    default_message = "百炼客户端配置异常"
    default_error_code = "bailian_configuration_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class AlibabaModelStudioInvocationError(AlibabaModelStudioError):
    """百炼模型调用异常。"""

    default_message = "百炼模型调用失败"
    default_error_code = "bailian_invocation_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class AlibabaModelStudioClient:
    """统一封装阿里云百炼的 Embedding 与 Chat 能力。"""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        timeout_seconds: float,
        embedding_model: str,
        embedding_dimensions: int,
        embedding_batch_size: int,
        chat_model: str | None,
    ) -> None:
        """初始化百炼客户端。

        参数:
            api_key: DashScope API Key。
            base_url: 百炼 OpenAI 兼容接口地址。
            timeout_seconds: 单次调用超时时间。
            embedding_model: 向量模型名称。
            embedding_dimensions: 向量维度。
            embedding_batch_size: 单次批量向量化最大文本数。
            chat_model: 预留的聊天模型名称；后续接入对话生成时复用。

        返回:
            None。

        异常:
            ValueError: 当超时时间、向量维度或批大小非法时抛出。
        """
        if timeout_seconds <= 0:
            msg = "百炼客户端超时时间必须大于 0"
            raise ValueError(msg)
        if not embedding_model.strip():
            msg = "百炼向量模型名称不能为空"
            raise ValueError(msg)
        if embedding_dimensions <= 0:
            msg = "向量维度必须大于 0"
            raise ValueError(msg)
        if embedding_batch_size <= 0:
            msg = "向量批大小必须大于 0"
            raise ValueError(msg)

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._embedding_batch_size = embedding_batch_size
        self._chat_model = chat_model
        self._client: Any | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> AlibabaModelStudioClient:
        """基于应用配置创建百炼客户端。"""
        return cls(
            api_key=settings.bailian_api_key,
            base_url=settings.bailian_base_url,
            timeout_seconds=settings.bailian_timeout_seconds,
            embedding_model=settings.chunk_embedding_model,
            embedding_dimensions=settings.chunk_embedding_dimensions,
            embedding_batch_size=settings.chunk_embedding_batch_size,
            chat_model=settings.bailian_chat_model,
        )

    def ensure_ready(self) -> None:
        """校验客户端可初始化且关键配置完整。"""
        self._validate_api_key()
        self._get_client()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """调用百炼向量模型生成文本向量。

        参数:
            texts: 待向量化文本列表。

        返回:
            与输入文本顺序一一对应的浮点向量列表。

        异常:
            AlibabaModelStudioConfigurationError: 当向量模型未配置时抛出。
            AlibabaModelStudioInvocationError: 当模型调用失败或响应不完整时抛出。
        """
        if not texts:
            return []
        self._validate_api_key()
        client = self._get_client()
        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._embedding_batch_size):
            batch = texts[start : start + self._embedding_batch_size]
            try:
                response = client.embeddings.create(
                    model=self._embedding_model,
                    input=batch,
                    dimensions=self._embedding_dimensions,
                    encoding_format="float",
                )
            except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
                msg = "调用百炼向量模型失败"
                raise AlibabaModelStudioInvocationError(msg) from exc

            data = sorted(getattr(response, "data", []), key=lambda item: item.index)
            if len(data) != len(batch):
                msg = "百炼向量模型返回数量异常"
                raise AlibabaModelStudioInvocationError(msg)

            embeddings.extend([self._extract_embedding(item.embedding) for item in data])
        return embeddings

    def complete_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> str:
        """调用百炼聊天模型完成一次对话补全。

        参数:
            messages: 标准化消息列表。
            temperature: 可选采样温度。

        返回:
            首个候选回复文本。

        异常:
            AlibabaModelStudioConfigurationError: 当聊天模型未配置时抛出。
            AlibabaModelStudioInvocationError: 当模型调用失败或响应为空时抛出。
        """
        if not self._chat_model:
            msg = "未配置百炼聊天模型"
            raise AlibabaModelStudioConfigurationError(msg)

        self._validate_api_key()
        try:
            response = self._get_client().chat.completions.create(
                model=self._chat_model,
                messages=[
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in messages
                ],
                temperature=temperature,
            )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            msg = "调用百炼聊天模型失败"
            raise AlibabaModelStudioInvocationError(msg) from exc

        choices = getattr(response, "choices", [])
        if not choices:
            msg = "百炼聊天模型未返回候选结果"
            raise AlibabaModelStudioInvocationError(msg)

        content = cast(str | None, getattr(choices[0].message, "content", None))
        if not content:
            msg = "百炼聊天模型返回空内容"
            raise AlibabaModelStudioInvocationError(msg)
        return content

    def _get_client(self) -> Any:
        """延迟初始化 OpenAI 兼容客户端。"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        """创建 OpenAI 兼容客户端实例。"""
        if OPENAI_CLIENT_CLASS is None:
            msg = "未安装 openai 依赖，无法启用阿里云百炼模型"
            raise AlibabaModelStudioDependencyError(msg) from OPENAI_IMPORT_ERROR

        return cast(
            Any,
            OPENAI_CLIENT_CLASS(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout_seconds,
            ),
        )

    def _validate_api_key(self) -> None:
        """校验 API Key 配置。"""
        if self._api_key:
            return
        msg = "未配置 DASHSCOPE_API_KEY，无法调用阿里云百炼模型"
        raise AlibabaModelStudioConfigurationError(msg)

    @staticmethod
    def _extract_embedding(raw_embedding: Any) -> list[float]:
        """标准化单个向量结果。"""
        if not isinstance(raw_embedding, list):
            msg = "百炼向量模型返回格式非法"
            raise AlibabaModelStudioInvocationError(msg)
        return [float(value) for value in raw_embedding]
