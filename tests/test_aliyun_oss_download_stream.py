"""阿里云 OSS 下载流式兼容性测试。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from baozhi_rag.infra.storage import aliyun_oss_file_store
from baozhi_rag.infra.storage.aliyun_oss_file_store import AliyunOssFileStore


@dataclass(slots=True)
class _FakeGetObjectRequest:
    """模拟 SDK 的获取对象请求。"""

    bucket: str
    key: str


class _FakeOssModule:
    """为测试提供最小化的 OSS 模块替身。"""

    GetObjectRequest = _FakeGetObjectRequest


class _FakeResponse:
    """包装下载响应体的测试替身。"""

    def __init__(self, body: object) -> None:
        self.body = body


class _FakeClient:
    """返回固定响应的 OSS 客户端替身。"""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.requests: list[_FakeGetObjectRequest] = []

    def get_object(self, request: _FakeGetObjectRequest) -> _FakeResponse:
        self.requests.append(request)
        return self._response


class _IterBytesOnlyBody:
    """模拟 `StreamBodyReader` 的流式读取接口。"""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.iter_called = False
        self.closed = False

    def read(self) -> bytes:
        return b"".join(self._chunks)

    def iter_bytes(self) -> Iterator[bytes]:
        self.iter_called = True
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True


def test_download_file_uses_iter_bytes_for_stream_body_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当响应体的 `read()` 不接收大小参数时，应改走 `iter_bytes()` 流式下载。"""
    body = _IterBytesOnlyBody([b"hello", b"-", b"world"])
    client = _FakeClient(_FakeResponse(body))
    store = AliyunOssFileStore(
        region="cn-hangzhou",
        endpoint="oss-cn-hangzhou.aliyuncs.com",
        bucket_name="light-rag",
        access_key_id="test-id",
        access_key_secret="test-secret",
    )
    store._client = client
    monkeypatch.setattr(aliyun_oss_file_store, "ALIYUN_OSS_MODULE", _FakeOssModule())
    monkeypatch.setattr(aliyun_oss_file_store, "ALIYUN_OSS_IMPORT_ERROR", None)

    local_path = tmp_path / "downloads" / "document.docx"
    storage_key = "knowledge-files/raw/3a/example.docx"

    store.download_file(storage_key=storage_key, local_path=local_path)

    assert local_path.read_bytes() == b"hello-world"
    assert body.iter_called is True
    assert body.closed is True
    assert client.requests == [_FakeGetObjectRequest(bucket="light-rag", key=storage_key)]
