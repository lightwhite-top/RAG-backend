"""健康检查接口测试。"""

from fastapi.testclient import TestClient


def test_root_returns_service_info(client: TestClient) -> None:
    """根路径应返回服务元信息。"""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["environment"] == "test"


def test_live_returns_ok(client: TestClient) -> None:
    """健康检查应返回服务可用。"""
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
