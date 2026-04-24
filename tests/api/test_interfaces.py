from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.domain.knowledge_file_errors import KnowledgeFileNotFoundError
from baozhi_rag.domain.user_errors import (
    AccessTokenInvalidError,
    RegistrationCodeSendTooFrequentError,
)
from baozhi_rag.services.chat import ChatBlockAsset, ChatContentBlock, ChatStreamEvent
from baozhi_rag.services.document_chunking import ChunkImageAsset


def _parse_sse_events(body: str) -> list[tuple[str, dict[str, object]]]:
    """把测试里的 SSE 文本流解析为事件列表，便于断言最终事件状态。"""
    events: list[tuple[str, dict[str, object]]] = []
    for chunk in body.split("\n\n"):
        normalized_chunk = chunk.strip()
        if not normalized_chunk:
            continue
        event_name = ""
        event_data: dict[str, object] = {}
        for line in normalized_chunk.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ").strip()
            if line.startswith("data: "):
                event_data = json.loads(line.removeprefix("data: "))
        if event_name:
            events.append((event_name, event_data))
    return events


def test_health_live_returns_service_metadata(
    build_harness: Iterator,
) -> None:
    with build_harness() as harness:
        response = harness.client.get("/health/live")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "success"
    assert payload["data"] == {
        "status": "ok",
        "service": harness.settings.app_name,
        "environment": harness.settings.app_env,
        "version": harness.settings.version,
    }
    assert response.headers["X-Request-ID"] == payload["request_id"]


def test_root_and_auth_me_return_authenticated_context(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        root_response = harness.client.get("/")
        me_response = harness.client.get("/auth/me")

    assert root_response.status_code == 200
    assert root_response.json()["data"] == {
        "service": harness.settings.app_name,
        "environment": harness.settings.app_env,
        "version": harness.settings.version,
        "docs_url": "/docs",
    }
    assert me_response.status_code == 200
    assert me_response.json()["data"]["id"] == standard_user.id
    assert me_response.headers["X-Request-ID"] == me_response.json()["request_id"]


def test_auth_me_requires_token(build_harness: Iterator) -> None:
    with build_harness() as harness:
        response = harness.client.get("/auth/me")

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"
    assert response.json()["message"] == "未提供有效的身份凭证"


def test_send_registration_code_success(build_harness: Iterator) -> None:
    with build_harness() as harness:
        response = harness.client.post(
            "/auth/register/code",
            json={"email": "new-user@example.com"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["expires_in"] == 600
    assert payload["data"]["resend_after"] == 60


def test_send_registration_code_validates_email_format(build_harness: Iterator) -> None:
    with build_harness() as harness:
        response = harness.client.post(
            "/auth/register/code",
            json={"email": "not-an-email"},
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "request_validation_error"
    assert payload["message"] == "请求参数校验失败"
    assert payload["details"][0]["message"] == "邮箱格式不正确"


def test_send_registration_code_surfaces_rate_limit_error(build_harness: Iterator) -> None:
    with build_harness() as harness:
        harness.auth_service.send_registration_code_error = RegistrationCodeSendTooFrequentError()

        response = harness.client.post(
            "/auth/register/code",
            json={"email": "new-user@example.com"},
        )

    assert response.status_code == 429
    assert response.json()["code"] == "registration_code_send_too_frequent"


def test_register_returns_user_payload(build_harness: Iterator) -> None:
    with build_harness() as harness:
        response = harness.client.post(
            "/auth/register",
            json={
                "email": "new-user@example.com",
                "password": "Password123",
                "username": "New User",
                "verification_code": "123456",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["email"] == "new-user@example.com"
    assert payload["data"]["role"] == "user"


def test_login_returns_access_token(build_harness: Iterator) -> None:
    with build_harness() as harness:
        response = harness.client.post(
            "/auth/login",
            json={"email": "user@example.com", "password": "Password123"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["access_token"] == "token-123"
    assert payload["data"]["token_type"] == "Bearer"
    assert payload["data"]["user"]["id"] == "user-1"


def test_login_requires_password(build_harness: Iterator) -> None:
    with build_harness() as harness:
        response = harness.client.post(
            "/auth/login",
            json={"email": "user@example.com"},
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "request_validation_error"
    assert payload["details"][0]["message"] == "字段不能为空"


def test_invalid_token_returns_401(build_harness: Iterator) -> None:
    with build_harness() as harness:
        harness.auth_service.token_error = AccessTokenInvalidError()

        response = harness.client.get(
            "/auth/me",
            headers={"Authorization": "Bearer invalid-token"},
        )

    assert response.status_code == 401
    assert response.json()["code"] == "token_invalid"


def test_update_me_returns_updated_profile(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.patch(
            "/auth/me",
            json={"username": "Updated Name"},
        )

    assert response.status_code == 200
    assert response.json()["data"]["username"] == "Updated Name"


def test_change_password_returns_success(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.patch(
            "/auth/password",
            json={
                "current_password": "Password123",
                "new_password": "Password456",
            },
        )

    assert response.status_code == 200
    assert response.json()["state"] == "success"
    assert response.json()["data"] is None


def test_file_list_endpoints_return_paginated_data(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        global_response = harness.client.get("/files/global")
        mine_response = harness.client.get("/files/mine")

    assert global_response.status_code == 200
    assert global_response.json()["meta"] == {"page": 1, "page_size": 20, "total": 1}
    assert global_response.json()["data"]["items"][0]["file_id"] == "file-1"
    assert global_response.json()["data"]["items"][0]["extension"] == "docx"
    assert global_response.json()["data"]["items"][0]["file_url"].endswith("/files/file-1/content")
    assert mine_response.status_code == 200
    assert mine_response.json()["data"]["items"][0]["uploader_user_id"] == standard_user.id
    assert mine_response.json()["data"]["items"][0]["extension"] == "docx"
    assert mine_response.json()["data"]["items"][0]["file_url"].endswith("/files/file-1/content")


def test_delete_my_all_files_returns_summary_and_uses_current_user(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.delete("/files/mine")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "success"
    assert payload["data"] == {"deleted_file_count": 1, "deleted_task_count": 1}
    assert response.headers["X-Request-ID"] == payload["request_id"]
    assert harness.file_delete_service.delete_all_called_user_ids == [standard_user.id]


def test_admin_delete_all_files_returns_summary(
    build_harness: Iterator,
    admin_user,
) -> None:
    with build_harness(current_user=admin_user) as harness:
        response = harness.client.delete("/admin/files")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "success"
    assert payload["data"] == {"deleted_file_count": 1, "deleted_task_count": 1}
    assert harness.file_delete_service.delete_all_globally_called_user_ids == [admin_user.id]


def test_file_content_endpoint_streams_same_origin_file_bytes(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.get("/files/file-1/content")

    assert response.status_code == 200
    assert response.content == b"fake-docx-bytes"
    assert response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def test_file_content_endpoint_supports_unicode_filename_header(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        harness.knowledge_file_repository.files[0] = replace(
            harness.knowledge_file_repository.files[0],
            original_filename="中文资料.docx",
        )
        response = harness.client.get("/files/file-1/content")

    assert response.status_code == 200
    assert response.content == b"fake-docx-bytes"
    assert response.headers["content-disposition"].startswith('inline; filename="download.docx"; ')
    assert (
        "filename*=UTF-8''%E4%B8%AD%E6%96%87%E8%B5%84%E6%96%99.docx"
        in response.headers["content-disposition"]
    )


def test_image_asset_endpoints_stream_same_origin_image_bytes(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        content_response = harness.client.get("/files/image-assets/img-1/content")
        preview_response = harness.client.get("/files/image-assets/img-1/preview")

    assert content_response.status_code == 200
    assert content_response.content == b"fake-image-bytes"
    assert content_response.headers["content-type"].startswith("image/png")
    assert preview_response.status_code == 200
    assert preview_response.content == b"fake-thumb-bytes"
    assert preview_response.headers["content-type"].startswith("image/png")


def test_openapi_extension_schema_is_nullable_string_instead_of_anyof(
    build_harness: Iterator,
) -> None:
    with build_harness() as harness:
        response = harness.client.get("/openapi.json")

    assert response.status_code == 200
    payload = response.json()

    file_extension_schema = payload["components"]["schemas"]["KnowledgeFileItem"]["properties"][
        "extension"
    ]
    task_extension_schema = payload["components"]["schemas"]["UploadTaskItem"]["properties"][
        "extension"
    ]
    file_required_fields = payload["components"]["schemas"]["KnowledgeFileItem"]["required"]
    task_required_fields = payload["components"]["schemas"]["UploadTaskItem"]["required"]

    assert file_extension_schema["type"] == "string"
    assert file_extension_schema["nullable"] is True
    assert file_extension_schema["example"] == "pdf"
    assert "anyOf" not in file_extension_schema
    assert "default" not in file_extension_schema
    assert "extension" in file_required_fields

    assert task_extension_schema["type"] == "string"
    assert task_extension_schema["nullable"] is True
    assert task_extension_schema["example"] == "pdf"
    assert "anyOf" not in task_extension_schema
    assert "default" not in task_extension_schema
    assert "extension" in task_required_fields


def test_upload_returns_accepted_task_summary(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.post(
            "/files/upload",
            files={
                "files": (
                    "policy.docx",
                    b"fake-docx-bytes",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["data"]["file_count"] == 1
    assert payload["data"]["tasks"][0]["task_id"] == "task-1"
    assert payload["data"]["tasks"][0]["extension"] == "docx"


def test_upload_requires_files_field(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.post("/files/upload")

    assert response.status_code == 422
    assert response.json()["code"] == "request_validation_error"


def test_upload_rejects_unsupported_document_type_immediately(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.post(
            "/files/upload",
            files={
                "files": (
                    "notes.txt",
                    b"plain text",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "unsupported_document_type"


def test_upload_task_endpoints_return_task_state(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        list_response = harness.client.get("/files/upload-tasks")
        detail_response = harness.client.get("/files/upload-tasks/task-1")
        retry_response = harness.client.post("/files/upload-tasks/task-1/retry")

    assert list_response.status_code == 200
    assert list_response.json()["data"]["task_count"] == 1
    assert list_response.json()["data"]["tasks"][0]["extension"] == "docx"
    assert detail_response.status_code == 200
    assert detail_response.json()["data"]["task_id"] == "task-1"
    assert detail_response.json()["data"]["extension"] == "docx"
    assert retry_response.status_code == 200
    assert retry_response.json()["data"]["task_id"] == "task-1"
    assert retry_response.json()["data"]["extension"] == "docx"


def test_delete_file_propagates_not_found_error(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        harness.file_delete_service.delete_error = KnowledgeFileNotFoundError()

        response = harness.client.delete("/files/file-404")

    assert response.status_code == 404
    assert response.json()["code"] == "knowledge_file_not_found"


def test_search_chunks_returns_hits(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.get("/search/chunks", params={"q": "deductible"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["query"] == "deductible"
    assert payload["data"]["size"] == 1
    assert payload["data"]["hits"][0]["chunk_id"] == "file-1-chunk-0"
    assert payload["data"]["hits"][0]["heading_path"] == ["产品说明", "免赔规则"]
    assert payload["data"]["hits"][0]["section_title"] == "免赔规则"
    assert payload["data"]["hits"][0]["content_type"] == "paragraph"


def test_search_chunks_validates_size(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.get("/search/chunks", params={"q": "deductible", "size": 99})

    assert response.status_code == 422
    assert response.json()["code"] == "request_validation_error"


def test_chat_completion_returns_structured_response(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.post(
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is a deductible?"}],
                "retrieval_size": 4,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["assistant_message"]["message_id"]
    assert payload["data"]["assistant_message"]["citations"][0]["id"] == "cit-1"
    assert payload["data"]["assistant_message"]["citations"][0]["file_url"].endswith(
        "/files/file-1/content"
    )
    assert payload["data"]["trace"]["model"] == harness.settings.llm_chat_model
    assert payload["data"]["trace"]["query_intent"] == "definition"
    assert payload["data"]["trace"]["retrieval_mode"] == "chat"
    assert payload["data"]["trace"]["lane_count"] == 3
    assert payload["data"]["trace"]["evidence_sufficient"] is True
    assert harness.chat_service.last_complete_request is not None
    assert harness.chat_service.last_complete_request["retrieval_size"] == 4


def test_chat_completion_accepts_legacy_temperature_field_for_compatibility(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.post(
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is a deductible?"}],
                "retrieval_size": 4,
                "temperature": 1.4,
            },
        )

    assert response.status_code == 200
    assert harness.chat_service.last_complete_request is not None
    assert harness.chat_service.last_complete_request["temperature"] == 1.4


def test_chat_completion_returns_image_assets_in_citation_and_content_blocks(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        image_asset = ChunkImageAsset(
            segment_id="seg-1",
            asset_id="img-1",
            asset_index=1,
            source_anchor="p:1:image:1",
            content_type="image/png",
            extension=".png",
            image_bytes=None,
            storage_key="knowledge-files/user-1/file-1/assets/images/img-1.png",
            thumbnail_storage_key="knowledge-files/user-1/file-1/assets/thumbnails/img-1.png",
            summary="A simple process diagram",
            ocr_text="apply -> review -> approve",
            image_type="diagram",
        )
        citation = replace(
            harness.chat_service.complete_result.citations[0],
            image_assets=[image_asset],
        )
        harness.chat_service.complete_result = replace(
            harness.chat_service.complete_result,
            citations=[citation],
            content_blocks=[
                ChatContentBlock(
                    block_id="blk-1",
                    block_type="markdown",
                    text="A deductible is the amount paid by the insured first.",
                    citation_ids=["cit-1"],
                    sequence=1,
                ),
                ChatContentBlock(
                    block_id="blk-1-imgs",
                    block_type="image_gallery",
                    text="",
                    citation_ids=["cit-1"],
                    sequence=2,
                    files_assets=[
                        ChatBlockAsset(
                            asset_id="img-1",
                            display_name="diagram",
                            storage_key="knowledge-files/user-1/file-1/assets/images/img-1.png",
                            content_type="image/png",
                            extension="png",
                            preview_storage_key=(
                                "knowledge-files/user-1/file-1/assets/thumbnails/img-1.png"
                            ),
                            source_anchor="p:1:image:1",
                            summary="A simple process diagram",
                            ocr_text="apply -> review -> approve",
                        )
                    ],
                ),
            ],
        )

        response = harness.client.post(
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is a deductible?"}],
                "retrieval_size": 4,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    citation_item = payload["data"]["assistant_message"]["citations"][0]
    assert citation_item["image_assets"][0]["image_url"].endswith(
        "/files/image-assets/img-1/content"
    )
    assert citation_item["image_assets"][0]["thumbnail_url"].endswith(
        "/files/image-assets/img-1/preview"
    )
    assert (
        payload["data"]["assistant_message"]["content_blocks"][1]["block_type"] == "image_gallery"
    )
    assert (
        payload["data"]["assistant_message"]["content_blocks"][1]["files_assets"][0]["summary"]
        == "A simple process diagram"
    )
    assert (
        payload["data"]["assistant_message"]["content_blocks"][1]["files_assets"][0]["preview_url"]
        is not None
    )
    assert payload["data"]["assistant_message"]["content_blocks"][1]["files_assets"][0][
        "url"
    ].endswith("/files/image-assets/img-1/content")
    assert payload["data"]["assistant_message"]["content_blocks"][1]["files_assets"][0][
        "preview_url"
    ].endswith("/files/image-assets/img-1/preview")


def test_chat_completion_session_mode_returns_persisted_message_identity(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.post(
            "/chat/completions",
            json={
                "session_id": "sess-1",
                "messages": [{"role": "user", "content": "What is a deductible?"}],
                "retrieval_size": 4,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["assistant_message"]["message_id"] == "msg-session-2"
    assert payload["data"]["assistant_message"]["session_id"] == "sess-1"
    assert payload["data"]["assistant_message"]["sequence_no"] == 2
    assert harness.conversation_chat_service.last_complete_request is not None
    assert harness.conversation_chat_service.last_complete_request["retrieval_size"] == 4


def test_chat_completion_session_stream_prepares_request_before_streaming(
    build_harness: Iterator,
    standard_user,
) -> None:
    with (
        build_harness(current_user=standard_user) as harness,
        harness.client.stream(
            "POST",
            "/chat/completions",
            json={
                "session_id": "sess-1",
                "messages": [{"role": "user", "content": "What is a deductible?"}],
                "retrieval_size": 4,
                "stream": True,
            },
        ) as response,
    ):
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: message.start" in body
    assert harness.conversation_chat_service.last_prepare_stream_request is not None
    assert harness.conversation_chat_service.last_stream_request is not None
    assert (
        harness.conversation_chat_service.last_stream_request["prepared_current_message"]
        is not None
    )


def test_chat_completion_stream_emits_message_error_event(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        harness.chat_service.stream_events = [
            ChatStreamEvent(
                event="context",
                data={
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "rewrite_applied": False,
                    "citations": [],
                },
            ),
            AppError(
                message="stream failed",
                error_code="stream_failed",
                status_code=500,
            ),
        ]

        with harness.client.stream(
            "POST",
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is a deductible?"}],
                "retrieval_size": 4,
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: message.start" in body
    assert "event: message.error" in body
    assert '"code":"stream_failed"' in body


def test_chat_completion_stream_emits_message_error_when_first_event_raises(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        harness.chat_service.stream_events = [
            AppError(
                message="stream failed before first event",
                error_code="stream_failed_early",
                status_code=500,
            )
        ]

        with harness.client.stream(
            "POST",
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is a deductible?"}],
                "retrieval_size": 4,
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert "event: message.error" not in body
    assert '"code":"stream_failed_early"' in body


def test_chat_completion_stream_emits_image_gallery_block_delta(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        harness.chat_service.stream_events = [
            ChatStreamEvent(
                event="context",
                data={
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "rewrite_applied": False,
                    "citations": [
                        {
                            "id": "cit-1",
                            "chunk_id": "file-1-chunk-0",
                            "file_id": "file-1",
                            "source_filename": "policy.docx",
                            "storage_key": "knowledge-files/user-1/file-1/policy.docx",
                            "chunk_index": 0,
                            "char_count": 42,
                            "content": "Deductible details and coverage notes.",
                            "snippet": "Deductible details and coverage notes.",
                            "merged_terms": ["deductible"],
                            "score": 0.91,
                            "heading_path": [],
                            "section_title": None,
                            "content_type": "paragraph",
                            "source_anchor": "chunk:0",
                            "image_assets": [
                                {
                                    "segment_id": "seg-1",
                                    "asset_id": "img-1",
                                    "source_anchor": "p:1:image:1",
                                    "storage_key": "knowledge-files/user-1/file-1/assets/images/img-1.png",
                                    "thumbnail_storage_key": "knowledge-files/user-1/file-1/assets/thumbnails/img-1.png",
                                    "image_type": "diagram",
                                    "summary": "A simple process diagram",
                                    "ocr_text": "apply -> review -> approve",
                                }
                            ],
                        }
                    ],
                },
            ),
            ChatStreamEvent(
                event="delta",
                data={
                    "delta_type": "insert",
                    "after_block_id": "blk-1",
                    "block": {
                        "block_id": "blk-1-imgs",
                        "block_type": "image_gallery",
                        "text": "",
                        "citation_ids": ["cit-1"],
                        "sequence": 2,
                        "files_assets": [
                            {
                                "asset_id": "img-1",
                                "display_name": "diagram",
                                "source_anchor": "p:1:image:1",
                                "storage_key": "knowledge-files/user-1/file-1/assets/images/img-1.png",
                                "preview_storage_key": "knowledge-files/user-1/file-1/assets/thumbnails/img-1.png",
                                "content_type": "image/png",
                                "extension": "png",
                                "summary": "A simple process diagram",
                                "ocr_text": "apply -> review -> approve",
                            }
                        ],
                    },
                },
            ),
            ChatStreamEvent(
                event="done",
                data={
                    "answer": "Hello",
                    "plain_text": "Hello",
                    "content_blocks": [
                        {
                            "block_id": "blk-1-imgs",
                            "block_type": "image_gallery",
                            "text": "",
                            "citation_ids": ["cit-1"],
                            "sequence": 2,
                            "files_assets": [
                                {
                                    "asset_id": "img-1",
                                    "display_name": "diagram",
                                    "source_anchor": "p:1:image:1",
                                    "storage_key": "knowledge-files/user-1/file-1/assets/images/img-1.png",
                                    "preview_storage_key": "knowledge-files/user-1/file-1/assets/thumbnails/img-1.png",
                                    "content_type": "image/png",
                                    "extension": "png",
                                    "summary": "A simple process diagram",
                                    "ocr_text": "apply -> review -> approve",
                                }
                            ],
                        }
                    ],
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "citations": [],
                    "finish_reason": "stop",
                    "rewrite_applied": False,
                },
            ),
        ]

        with harness.client.stream(
            "POST",
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is a deductible?"}],
                "retrieval_size": 4,
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: message.delta" in body
    assert '"delta_type":"insert"' in body
    assert '"block_type":"image_gallery"' in body
    assert '"files_assets"' in body
    assert '"http://testserver/files/image-assets/img-1/content"' in body
    assert '"http://testserver/files/image-assets/img-1/preview"' in body


def test_chat_completion_stream_emits_source_file_block_delta(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        harness.chat_service.stream_events = [
            ChatStreamEvent(
                event="context",
                data={
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "rewrite_applied": False,
                    "citations": [
                        {
                            "id": "cit-1",
                            "chunk_id": "file-1-chunk-0",
                            "file_id": "file-1",
                            "source_filename": "policy.docx",
                            "storage_key": "knowledge-files/user-1/file-1/policy.docx",
                            "chunk_index": 0,
                            "char_count": 42,
                            "content": "Deductible details and coverage notes.",
                            "snippet": "Deductible details and coverage notes.",
                            "merged_terms": ["deductible"],
                            "score": 0.91,
                            "heading_path": [],
                            "section_title": None,
                            "content_type": "paragraph",
                            "source_anchor": "chunk:0",
                            "file_content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            "extension": "docx",
                            "size": 256,
                            "image_assets": [],
                        }
                    ],
                },
            ),
            ChatStreamEvent(
                event="delta",
                data={
                    "delta_type": "insert",
                    "after_block_id": "blk-1",
                    "block": {
                        "block_id": "blk-1-files",
                        "block_type": "source_file",
                        "text": "",
                        "citation_ids": ["cit-1"],
                        "sequence": 2,
                        "files_assets": [
                            {
                                "asset_id": "file-1",
                                "display_name": "policy.docx",
                                "storage_key": "knowledge-files/user-1/file-1/policy.docx",
                                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                "extension": "docx",
                                "size": 256,
                            }
                        ],
                    },
                },
            ),
            ChatStreamEvent(
                event="done",
                data={
                    "answer": "Hello",
                    "plain_text": "Hello",
                    "content_blocks": [
                        {
                            "block_id": "blk-1-files",
                            "block_type": "source_file",
                            "text": "",
                            "citation_ids": ["cit-1"],
                            "sequence": 2,
                            "files_assets": [
                                {
                                    "asset_id": "file-1",
                                    "display_name": "policy.docx",
                                    "storage_key": "knowledge-files/user-1/file-1/policy.docx",
                                    "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    "extension": "docx",
                                    "size": 256,
                                }
                            ],
                        }
                    ],
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "citations": [],
                    "finish_reason": "stop",
                    "rewrite_applied": False,
                },
            ),
        ]

        with harness.client.stream(
            "POST",
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is a deductible?"}],
                "retrieval_size": 4,
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"block_type":"source_file"' in body
    assert '"http://testserver/files/file-1/content"' in body


def test_chat_session_endpoints_return_expected_payloads(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        create_response = harness.client.post("/chat/sessions", json={"title": "理赔等待期咨询"})
        list_response = harness.client.get("/chat/sessions")
        detail_response = harness.client.get("/chat/sessions/sess-1")
        update_response = harness.client.patch(
            "/chat/sessions/sess-1",
            json={"title": "更新后的标题", "status": "archived"},
        )
        history_response = harness.client.get("/chat/sessions/sess-1/messages")
        delete_response = harness.client.delete("/chat/sessions/sess-1")

    assert create_response.status_code == 200
    assert create_response.json()["data"]["session"]["session_id"] == "sess-1"
    assert list_response.status_code == 200
    assert list_response.json()["meta"] == {"page": 1, "page_size": 20, "total": 1}
    assert detail_response.status_code == 200
    assert detail_response.json()["data"]["session"]["title"] == "理赔等待期咨询"
    assert update_response.status_code == 200
    assert history_response.status_code == 200
    assert history_response.json()["data"]["items"][1]["assistant_message"]["message_id"] == "msg-2"
    assert history_response.json()["data"]["items"][1]["assistant_message"]["citations"][0][
        "file_url"
    ].endswith("/files/file-1/content")
    assert delete_response.status_code == 200
    assert harness.chat_session_service.deleted_session_ids == ["sess-1"]


def test_chat_session_history_normalizes_legacy_image_assets_to_files_assets(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        updated_assistant_message = replace(
            harness.chat_session_service.history_result.items[1],
            content_blocks=[
                {
                    "block_id": "blk-1-imgs",
                    "block_type": "image_gallery",
                    "text": "",
                    "citation_ids": ["cit-1"],
                    "sequence": 2,
                    "image_assets": [
                        {
                            "segment_id": "seg-1",
                            "asset_id": "img-1",
                            "source_anchor": "p:1:image:1",
                            "storage_key": "knowledge-files/user-1/file-1/assets/images/img-1.png",
                            "thumbnail_storage_key": "knowledge-files/user-1/file-1/assets/thumbnails/img-1.png",
                            "image_type": "diagram",
                            "summary": "A simple process diagram",
                            "ocr_text": "apply -> review -> approve",
                        }
                    ],
                }
            ],
        )
        harness.chat_session_service.history_result = replace(
            harness.chat_session_service.history_result,
            items=[
                harness.chat_session_service.history_result.items[0],
                updated_assistant_message,
            ],
        )

        response = harness.client.get("/chat/sessions/sess-1/messages")

    assert response.status_code == 200
    block = response.json()["data"]["items"][1]["assistant_message"]["content_blocks"][0]
    assert block["block_type"] == "image_gallery"
    assert block["files_assets"][0]["url"].endswith("/files/image-assets/img-1/content")
    assert block["files_assets"][0]["preview_url"].endswith("/files/image-assets/img-1/preview")


def test_chat_session_history_hides_deleted_file_jump_links(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        updated_assistant_message = replace(
            harness.chat_session_service.history_result.items[1],
            content_blocks=[
                {
                    "block_id": "blk-1-files",
                    "block_type": "source_file",
                    "text": "",
                    "citation_ids": ["cit-1"],
                    "sequence": 2,
                    "files_assets": [
                        {
                            "asset_id": "file-1",
                            "display_name": "policy.docx",
                            "storage_key": "knowledge-files/user-1/file-1/policy.docx",
                            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            "extension": "docx",
                        }
                    ],
                },
                {
                    "block_id": "blk-1-imgs",
                    "block_type": "image_gallery",
                    "text": "",
                    "citation_ids": ["cit-1"],
                    "sequence": 3,
                    "files_assets": [
                        {
                            "segment_id": "seg-1",
                            "asset_id": "img-1",
                            "source_anchor": "p:1:image:1",
                            "storage_key": "knowledge-files/user-1/file-1/assets/images/img-1.png",
                            "thumbnail_storage_key": "knowledge-files/user-1/file-1/assets/thumbnails/img-1.png",
                            "image_type": "diagram",
                            "summary": "A simple process diagram",
                            "ocr_text": "apply -> review -> approve",
                        }
                    ],
                },
            ],
        )
        harness.chat_session_service.history_result = replace(
            harness.chat_session_service.history_result,
            items=[
                harness.chat_session_service.history_result.items[0],
                updated_assistant_message,
            ],
        )
        harness.knowledge_file_repository.files = []

        response = harness.client.get("/chat/sessions/sess-1/messages")

    assert response.status_code == 200
    assistant_message = response.json()["data"]["items"][1]["assistant_message"]
    citation = assistant_message["citations"][0]
    assert citation["file_url"] is None
    assert citation["expires_at"] is None
    assert citation["image_assets"] == []
    source_file_block = assistant_message["content_blocks"][0]
    image_gallery_block = assistant_message["content_blocks"][1]
    assert source_file_block["files_assets"][0]["url"] is None
    assert source_file_block["files_assets"][0]["preview_url"] is None
    assert source_file_block["files_assets"][0]["expires_at"] is None
    assert image_gallery_block["files_assets"][0]["url"] is None
    assert image_gallery_block["files_assets"][0]["preview_url"] is None
    assert image_gallery_block["files_assets"][0]["expires_at"] is None


def test_chat_completion_stream_message_end_uses_final_done_state(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        harness.chat_service.stream_events = [
            ChatStreamEvent(
                event="context",
                data={
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "rewrite_applied": False,
                    "query_intent": "definition",
                    "citations": [
                        {
                            "id": "cit-1",
                            "chunk_id": "file-1-chunk-0",
                            "file_id": "file-1",
                            "source_filename": "policy.docx",
                            "storage_key": "knowledge-files/user-1/file-1/policy.docx",
                            "chunk_index": 0,
                            "char_count": 42,
                            "content": "Deductible details and coverage notes.",
                            "snippet": "Deductible details and coverage notes.",
                            "merged_terms": ["deductible"],
                            "score": 0.91,
                            "heading_path": [],
                            "section_title": None,
                            "content_type": "paragraph",
                            "source_anchor": "chunk:0",
                        }
                    ],
                    "retrieval_trace": {
                        "mode": "chat",
                        "lane_count": 3,
                        "final_hit_count": 2,
                        "evidence_sufficient": True,
                        "evidence_reason": "sufficient",
                        "deep_rerank_triggered": False,
                        "lanes": [],
                    },
                    "evidence_assessment": {
                        "sufficient": True,
                        "reason_code": "sufficient",
                    },
                    "applied_retrieval_size": 5,
                    "applied_temperature": 0.0,
                },
            ),
            ChatStreamEvent(
                event="done",
                data={
                    "answer": "No grounded answer.",
                    "plain_text": "No grounded answer.",
                    "content_blocks": [],
                    "original_query": "What is a deductible?",
                    "retrieval_query": "What is a deductible?",
                    "citations": [],
                    "finish_reason": "evidence_insufficient",
                    "rewrite_applied": False,
                    "query_intent": "definition",
                    "retrieval_trace": {
                        "mode": "chat",
                        "lane_count": 1,
                        "final_hit_count": 0,
                        "evidence_sufficient": False,
                        "evidence_reason": "context_budget_exhausted",
                        "deep_rerank_triggered": False,
                        "lanes": [],
                    },
                    "evidence_assessment": {
                        "sufficient": False,
                        "reason_code": "context_budget_exhausted",
                    },
                    "applied_retrieval_size": 3,
                    "applied_temperature": 0.0,
                },
            ),
        ]

        with harness.client.stream(
            "POST",
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is a deductible?"}],
                "retrieval_size": 4,
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    events = _parse_sse_events(body)
    assert events[-1][0] == "message.end"
    end_payload = events[-1][1]
    assistant_message = end_payload["assistant_message"]
    trace = end_payload["trace"]
    assert assistant_message["citations"] == []
    assert assistant_message["finish_reason"] == "evidence_insufficient"
    assert trace["final_hit_count"] == 0
    assert trace["evidence_sufficient"] is False
    assert trace["evidence_reason"] == "context_budget_exhausted"
    assert trace["applied_retrieval_size"] == 3


def test_chat_session_history_propagates_not_found_error(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        harness.chat_session_service.history_error = AppError(
            message="session not found",
            error_code="chat_session_not_found",
            status_code=404,
        )
        response = harness.client.get("/chat/sessions/missing/messages")

    assert response.status_code == 404
    assert response.json()["code"] == "chat_session_not_found"


def test_admin_endpoints_require_admin_role(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.get("/admin/users")

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


def test_admin_crud_endpoints_return_expected_payloads(
    build_harness: Iterator,
    admin_user,
) -> None:
    with build_harness(current_user=admin_user) as harness:
        list_response = harness.client.get("/admin/users")
        get_response = harness.client.get("/admin/users/user-2")
        create_response = harness.client.post(
            "/admin/users",
            json={
                "email": "created@example.com",
                "password": "Password123",
                "username": "Created User",
                "role": "user",
            },
        )
        update_response = harness.client.patch(
            "/admin/users/user-3",
            json={"username": "Updated User", "role": "admin"},
        )
        delete_response = harness.client.delete("/admin/users/user-3")

    assert list_response.status_code == 200
    assert list_response.json()["data"]["items"][0]["id"] == "user-1"
    assert get_response.status_code == 200
    assert get_response.json()["data"]["id"] == "user-2"
    assert create_response.status_code == 200
    assert create_response.json()["data"]["email"] == "created@example.com"
    assert update_response.status_code == 200
    assert update_response.json()["data"]["role"] == "admin"
    assert delete_response.status_code == 200
    assert harness.user_admin_service.deleted_user_ids == ["user-3"]


def test_admin_create_user_requires_email(
    build_harness: Iterator,
    admin_user,
) -> None:
    with build_harness(current_user=admin_user) as harness:
        response = harness.client.post(
            "/admin/users",
            json={
                "password": "Password123",
                "username": "Created User",
                "role": "user",
            },
        )

    assert response.status_code == 422
    assert response.json()["code"] == "request_validation_error"


def test_admin_user_detail_propagates_not_found_error(
    build_harness: Iterator,
    admin_user,
) -> None:
    with build_harness(current_user=admin_user) as harness:
        harness.user_admin_service.get_error = AppError(
            message="user not found",
            error_code="user_not_found",
            status_code=404,
        )

        response = harness.client.get("/admin/users/missing-user")

    assert response.status_code == 404
    assert response.json()["code"] == "user_not_found"


def test_framework_not_found_error_returns_chinese_message(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.get("/missing-route")

    assert response.status_code == 404
    assert response.json()["message"] == "请求的资源不存在"
