from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

from baozhi_rag.core.exceptions import AppError
from baozhi_rag.domain.knowledge_file_errors import KnowledgeFileNotFoundError
from baozhi_rag.domain.user_errors import (
    AccessTokenInvalidError,
    RegistrationCodeSendTooFrequentError,
)
from baozhi_rag.services.chat import ChatContentBlock, ChatStreamEvent
from baozhi_rag.services.document_chunking import ChunkImageAsset


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
    assert mine_response.status_code == 200
    assert mine_response.json()["data"]["items"][0]["uploader_user_id"] == standard_user.id


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


def test_upload_requires_files_field(
    build_harness: Iterator,
    standard_user,
) -> None:
    with build_harness(current_user=standard_user) as harness:
        response = harness.client.post("/files/upload")

    assert response.status_code == 422
    assert response.json()["code"] == "request_validation_error"


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
    assert detail_response.status_code == 200
    assert detail_response.json()["data"]["task_id"] == "task-1"
    assert retry_response.status_code == 200
    assert retry_response.json()["data"]["task_id"] == "task-1"


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
    assert payload["data"]["trace"]["model"] == harness.settings.llm_chat_model
    assert payload["data"]["trace"]["query_intent"] == "definition"
    assert payload["data"]["trace"]["retrieval_mode"] == "chat"
    assert payload["data"]["trace"]["lane_count"] == 3
    assert payload["data"]["trace"]["evidence_sufficient"] is True


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
                    image_assets=[image_asset],
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
    assert citation_item["image_assets"][0]["image_url"].startswith("https://example.com/")
    assert (
        payload["data"]["assistant_message"]["content_blocks"][1]["block_type"] == "image_gallery"
    )
    assert (
        payload["data"]["assistant_message"]["content_blocks"][1]["image_assets"][0]["summary"]
        == "A simple process diagram"
    )


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
                    "delta_type": "content_block",
                    "block": {
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
    assert '"delta_type":"content_block"' in body
    assert '"block_type":"image_gallery"' in body


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
    assert delete_response.status_code == 200
    assert harness.chat_session_service.deleted_session_ids == ["sess-1"]


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
