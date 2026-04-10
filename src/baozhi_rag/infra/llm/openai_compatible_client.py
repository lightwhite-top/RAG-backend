"""OpenAI 兼容大模型客户端封装。"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

from fastapi import status

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.services.llm import ChatMessage

if TYPE_CHECKING:
    from baozhi_rag.core.config import Settings

try:  # pragma: no cover - 是否安装依赖取决于运行环境
    from openai import OpenAI as ImportedOpenAIClient  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - 测试环境可通过可选导入绕过
    openai_client_class: type[Any] | None = None
    openai_import_error: Exception | None = exc
else:  # pragma: no cover - 导入成功路径不需要单独覆盖
    openai_client_class = ImportedOpenAIClient
    openai_import_error = None

LOGGER = logging.getLogger(__name__)


class OpenAICompatibleLlmError(AppError):
    """OpenAI 兼容大模型客户端异常。"""

    default_message = "大模型服务调用失败"
    default_error_code = "llm_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class OpenAICompatibleLlmDependencyError(OpenAICompatibleLlmError):
    """大模型客户端依赖缺失。"""

    default_message = "大模型客户端依赖缺失"
    default_error_code = "llm_dependency_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class OpenAICompatibleLlmConfigurationError(OpenAICompatibleLlmError):
    """大模型客户端配置异常。"""

    default_message = "大模型客户端配置异常"
    default_error_code = "llm_configuration_error"
    default_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class OpenAICompatibleLlmInvocationError(OpenAICompatibleLlmError):
    """大模型调用异常。"""

    default_message = "调用大模型失败"
    default_error_code = "llm_invocation_error"
    default_status_code = status.HTTP_502_BAD_GATEWAY


class OpenAICompatibleLlmClient:
    """统一封装 OpenAI 兼容接口的 Embedding 与 Chat 能力。"""

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
        """初始化大模型客户端。"""
        if timeout_seconds <= 0:
            msg = "大模型客户端超时时间必须大于 0"
            raise ValueError(msg)
        if not embedding_model.strip():
            msg = "向量大模型名称不能为空"
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
    def from_settings(cls, settings: Settings) -> OpenAICompatibleLlmClient:
        """基于应用配置创建大模型客户端。"""
        return cls(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            embedding_model=settings.chunk_embedding_model,
            embedding_dimensions=settings.chunk_embedding_dimensions,
            embedding_batch_size=settings.chunk_embedding_batch_size,
            chat_model=settings.llm_chat_model,
        )

    @classmethod
    def from_embedding_settings(cls, settings: Settings) -> OpenAICompatibleLlmClient:
        """基于应用配置创建向量化专用客户端。"""
        return cls(
            api_key=settings.resolved_chunk_embedding_llm_api_key,
            base_url=settings.resolved_chunk_embedding_llm_base_url,
            timeout_seconds=settings.resolved_chunk_embedding_llm_timeout_seconds,
            embedding_model=settings.chunk_embedding_model,
            embedding_dimensions=settings.chunk_embedding_dimensions,
            embedding_batch_size=settings.chunk_embedding_batch_size,
            chat_model=settings.llm_chat_model,
        )

    def ensure_ready(self) -> None:
        """校验客户端可初始化且关键配置完整。"""
        self._validate_api_key()
        self._get_client()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """调用向量大模型生成文本向量。"""
        if not texts:
            return []
        self._validate_api_key()
        client = self._get_client()
        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._embedding_batch_size):
            # 这里按当前上游的单次批量限制切片，避免大批次请求直接被上游拒绝。
            batch = texts[start : start + self._embedding_batch_size]
            try:
                response = client.embeddings.create(
                    model=self._embedding_model,
                    input=batch,
                    dimensions=self._embedding_dimensions,
                    encoding_format="float",
                )
            except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
                self._log_upstream_failure(
                    operation="embeddings.create",
                    model_name=self._embedding_model,
                    exc=exc,
                )
                msg = "调用向量大模型失败"
                raise OpenAICompatibleLlmInvocationError(msg) from exc

            data = sorted(getattr(response, "data", []), key=lambda item: item.index)
            if len(data) != len(batch):
                msg = "向量大模型返回数量异常"
                raise OpenAICompatibleLlmInvocationError(msg)

            embeddings.extend([self._extract_embedding(item.embedding) for item in data])
        return embeddings

    def complete_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> str:
        """调用聊天大模型完成一次对话补全。"""
        if not self._chat_model:
            msg = "未配置聊天大模型"
            raise OpenAICompatibleLlmConfigurationError(msg)

        self._validate_api_key()
        try:
            response = self._get_client().chat.completions.create(
                **self._build_chat_completion_request(
                    messages,
                    temperature=temperature,
                )
            )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            self._log_upstream_failure(
                operation="chat.completions.create",
                model_name=self._chat_model,
                exc=exc,
            )
            msg = "调用聊天大模型失败"
            raise OpenAICompatibleLlmInvocationError(msg) from exc

        choices = getattr(response, "choices", [])
        if not choices:
            msg = "聊天大模型未返回候选结果"
            raise OpenAICompatibleLlmInvocationError(msg)

        content = getattr(choices[0].message, "content", None)
        if not isinstance(content, str) or not content:
            msg = "聊天大模型返回空内容"
            raise OpenAICompatibleLlmInvocationError(msg)
        return content

    def recognize_image(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        model_name: str,
    ) -> dict[str, str]:
        """调用多模态模型识别文档图片并返回结构化结果。"""
        self._validate_api_key()
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            msg = "未配置图片识别模型名称"
            raise OpenAICompatibleLlmConfigurationError(msg)

        request_payload = {
            "model": normalized_model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是文档图片识别助手。请识别图片中的文字和图示语义，只返回 JSON。"
                        "JSON 必须包含三个字符串字段：ocr_text、summary、image_type。"
                        "image_type 仅允许 diagram、stamp、handwriting、screenshot、mixed、unknown。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请识别这张文档图片。"
                                "ocr_text 填写图片中的可见文字；"
                                "summary 用中文概括图片语义；"
                                "image_type 选择最贴近的类别。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._build_image_data_url(
                                    image_bytes=image_bytes,
                                    content_type=content_type,
                                )
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.0,
        }

        try:
            response = self._get_client().chat.completions.create(**request_payload)
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            self._log_upstream_failure(
                operation="chat.completions.create_image_recognition",
                model_name=normalized_model_name,
                exc=exc,
            )
            msg = "调用图片识别模型失败"
            raise OpenAICompatibleLlmInvocationError(msg) from exc

        choices = getattr(response, "choices", [])
        if not choices:
            msg = "图片识别模型未返回候选结果"
            raise OpenAICompatibleLlmInvocationError(msg)

        content = getattr(choices[0].message, "content", None)
        if not isinstance(content, str) or not content.strip():
            msg = "图片识别模型返回空内容"
            raise OpenAICompatibleLlmInvocationError(msg)

        parsed_content = self._parse_json_object(content)
        return {
            "ocr_text": str(parsed_content.get("ocr_text", "")).strip(),
            "summary": str(parsed_content.get("summary", "")).strip(),
            "image_type": str(parsed_content.get("image_type", "unknown")).strip() or "unknown",
        }

    def complete_json(
        self,
        messages: list[ChatMessage],
        *,
        model_name: str,
        temperature: float | None = 0.0,
    ) -> dict[str, object]:
        """调用指定模型并强制返回 JSON 对象。"""
        self._validate_api_key()
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            msg = "未配置结构化输出模型名称"
            raise OpenAICompatibleLlmConfigurationError(msg)

        try:
            response = self._get_client().chat.completions.create(
                model=normalized_model_name,
                messages=[
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in messages
                ],
                temperature=self._normalize_temperature(temperature) or 0.0,
            )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            self._log_upstream_failure(
                operation="chat.completions.create_json",
                model_name=normalized_model_name,
                exc=exc,
            )
            msg = "调用结构化输出模型失败"
            raise OpenAICompatibleLlmInvocationError(msg) from exc

        choices = getattr(response, "choices", [])
        if not choices:
            msg = "结构化输出模型未返回候选结果"
            raise OpenAICompatibleLlmInvocationError(msg)

        content = getattr(choices[0].message, "content", None)
        if not isinstance(content, str) or not content.strip():
            msg = "结构化输出模型返回空内容"
            raise OpenAICompatibleLlmInvocationError(msg)
        return self._parse_json_object(content)

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """调用聊天大模型执行流式对话补全。"""
        if not self._chat_model:
            msg = "未配置聊天大模型"
            raise OpenAICompatibleLlmConfigurationError(msg)

        self._validate_api_key()
        try:
            response = self._get_client().chat.completions.create(
                **self._build_chat_completion_request(
                    messages,
                    temperature=temperature,
                    stream=True,
                )
            )
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            self._log_upstream_failure(
                operation="chat.completions.create",
                model_name=self._chat_model,
                exc=exc,
            )
            msg = "调用聊天大模型失败"
            raise OpenAICompatibleLlmInvocationError(msg) from exc

        try:
            for chunk in response:
                # 不同 SDK 版本的流式 chunk 结构可能略有差异，统一在一个方法里做兼容提取。
                delta_text = self._extract_stream_delta(chunk)
                if delta_text:
                    yield delta_text
        except Exception as exc:  # pragma: no cover - 第三方异常类型不稳定
            self._log_upstream_failure(
                operation="chat.completions.stream_read",
                model_name=self._chat_model,
                exc=exc,
            )
            msg = "读取聊天大模型流式结果失败"
            raise OpenAICompatibleLlmInvocationError(msg) from exc

    def _get_client(self) -> Any:
        """延迟初始化 OpenAI 兼容客户端。"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        """创建 OpenAI 兼容客户端实例。"""
        if openai_client_class is None:
            msg = "未安装 openai 依赖，无法启用大模型客户端"
            raise OpenAICompatibleLlmDependencyError(msg) from openai_import_error

        return openai_client_class(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout_seconds,
        )

    def _build_chat_completion_request(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None,
        stream: bool = False,
    ) -> dict[str, object]:
        """构造聊天补全请求参数，并兼容上游对 `temperature` 的严格校验。"""
        request_payload: dict[str, object] = {
            "model": self._chat_model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
        }
        normalized_temperature = self._normalize_temperature(temperature)
        if normalized_temperature is not None:
            request_payload["temperature"] = normalized_temperature
        if stream:
            request_payload["stream"] = True
        return request_payload

    def _validate_api_key(self) -> None:
        """校验 API Key 配置。"""
        if self._api_key:
            return
        msg = "未配置模型服务 API Key，无法调用聊天或向量化能力"
        raise OpenAICompatibleLlmConfigurationError(msg)

    def _log_upstream_failure(
        self,
        *,
        operation: str,
        model_name: str,
        exc: Exception,
    ) -> None:
        """记录上游大模型调用失败的关键上下文，便于排查模型、权限或网关问题。"""
        LOGGER.exception(
            "llm_upstream_failed operation=%s model=%s base_url=%s timeout_seconds=%s upstream=%s",
            operation,
            model_name,
            self._base_url,
            self._timeout_seconds,
            self._summarize_exception(exc),
        )

    @classmethod
    def _summarize_exception(cls, exc: Exception) -> str:
        """提取第三方异常中的稳定字段，避免日志里只有泛化错误。"""
        parts = [f"type={type(exc).__name__}"]

        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            parts.append(f"status_code={status_code}")

        error_code = getattr(exc, "code", None)
        if error_code:
            parts.append(f"code={error_code}")

        request_id = getattr(exc, "request_id", None)
        if request_id:
            parts.append(f"request_id={request_id}")

        raw_message = str(exc).strip()
        if raw_message:
            parts.append(f"message={cls._truncate_log_text(raw_message)}")

        body = getattr(exc, "body", None)
        if body is not None:
            parts.append(f"body={cls._truncate_log_text(cls._serialize_log_value(body))}")

        return " ".join(parts)

    @staticmethod
    def _serialize_log_value(value: object) -> str:
        """把异常对象中的 body 等复杂字段转换为便于日志检索的文本。"""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)

    @staticmethod
    def _truncate_log_text(value: str, *, max_length: int = 500) -> str:
        """限制日志字段长度，避免上游返回整页错误体时污染日志。"""
        normalized = " ".join(value.split())
        if len(normalized) <= max_length:
            return normalized
        return f"{normalized[:max_length]}..."

    @staticmethod
    def _as_object_list(value: object) -> list[object] | None:
        """鎶婂姩鎬佸€煎綊涓€鍖栦负鍙亶鍘嗙殑 `list[object]`銆?"""
        if not isinstance(value, list):
            return None
        return cast(list[object], value)

    @staticmethod
    def _get_optional_attribute(value: object, attribute_name: str) -> object | None:
        """浠ュ畨鍏ㄧ被鍨嬭繑鍥炶鍙栧姩鎬佸璞＄殑鍙€夊睘鎬с€?"""
        return getattr(value, attribute_name, None)

    @staticmethod
    def _extract_stream_text_piece(value: object) -> str | None:
        """浠庢祦寮忓唴瀹瑰垎娈靛璞′腑鎻愬彇绾枃鏈墖娈点€?"""
        if isinstance(value, Mapping):
            mapping_value = cast(Mapping[object, object], value)
            text_value = mapping_value.get("text")
            if isinstance(text_value, str):
                return text_value
            return None

        text_value = OpenAICompatibleLlmClient._get_optional_attribute(value, "text")
        if isinstance(text_value, str):
            return text_value
        return None

    @staticmethod
    def _extract_embedding(raw_embedding: object) -> list[float]:
        """标准化单个向量结果。"""
        embedding_values = OpenAICompatibleLlmClient._as_object_list(raw_embedding)
        if embedding_values is None:
            msg = "向量大模型返回格式非法"
            raise OpenAICompatibleLlmInvocationError(msg)

        normalized_embedding: list[float] = []
        for value in embedding_values:
            if isinstance(value, bool):
                normalized_embedding.append(float(value))
                continue
            if isinstance(value, int | float | str):
                normalized_embedding.append(float(value))
                continue

            msg = "向量大模型返回格式非法"
            raise OpenAICompatibleLlmInvocationError(msg)
        return normalized_embedding

    @staticmethod
    def _normalize_temperature(temperature: float | None) -> float | None:
        """把温度参数归一化为真正的浮点数；未传时直接省略。"""
        if temperature is None:
            return None
        return float(temperature)

    @staticmethod
    def _extract_stream_delta(chunk: object) -> str:
        """从流式 chunk 中提取文本增量。"""
        choices = OpenAICompatibleLlmClient._as_object_list(
            OpenAICompatibleLlmClient._get_optional_attribute(chunk, "choices")
        )
        if not choices:
            return ""

        delta = OpenAICompatibleLlmClient._get_optional_attribute(choices[0], "delta")
        if delta is None:
            return ""

        content = OpenAICompatibleLlmClient._get_optional_attribute(delta, "content")
        if isinstance(content, str):
            return content

        content_items = OpenAICompatibleLlmClient._as_object_list(content)
        if content_items is None:
            return ""

        # 某些兼容实现会把内容拆成分段对象列表，这里统一拼回纯文本。
        pieces: list[str] = []
        for item in content_items:
            text_value = OpenAICompatibleLlmClient._extract_stream_text_piece(item)
            if text_value is not None:
                pieces.append(text_value)
        return "".join(pieces)

    @staticmethod
    def _build_image_data_url(*, image_bytes: bytes, content_type: str) -> str:
        """把图片字节编码为多模态模型可读取的 data URL。"""
        encoded_bytes = base64.b64encode(image_bytes).decode("ascii")
        normalized_content_type = content_type.strip() or "application/octet-stream"
        return f"data:{normalized_content_type};base64,{encoded_bytes}"

    @staticmethod
    def _parse_json_object(raw_content: str) -> dict[str, object]:
        """从模型返回文本中提取 JSON 对象。"""
        text = raw_content.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start_index = text.find("{")
            end_index = text.rfind("}")
            if start_index < 0 or end_index < start_index:
                msg = "图片识别模型未返回合法 JSON"
                raise OpenAICompatibleLlmInvocationError(msg) from None
            try:
                parsed = json.loads(text[start_index : end_index + 1])
            except json.JSONDecodeError as exc:
                msg = "图片识别模型未返回合法 JSON"
                raise OpenAICompatibleLlmInvocationError(msg) from exc

        if not isinstance(parsed, dict):
            msg = "图片识别模型返回的 JSON 结构非法"
            raise OpenAICompatibleLlmInvocationError(msg)
        return cast(dict[str, object], parsed)
