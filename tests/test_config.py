"""配置加载测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from baozhi_rag.core.config import get_settings, resolve_settings_env_files


def test_resolve_settings_env_files_includes_app_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当基础环境文件声明 APP_ENV 时，应自动叠加对应覆盖文件。"""
    (tmp_path / ".env").write_text(
        "APP_ENV=development\nES_URL=http://127.0.0.1:9200\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.development").write_text(
        "ES_URL=http://122.152.231.116:9200\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("APP_ENV", raising=False)

    env_files = resolve_settings_env_files()

    assert env_files == (".env", ".env.development")


def test_get_settings_loads_env_specific_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """环境覆盖文件中的连接地址应覆盖基础 `.env` 配置。"""
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "APP_ENV=development",
                "JWT_SECRET_KEY=test-secret",
                "REGISTRATION_CODE_SECRET=test-registration-secret",
                "MYSQL_HOST=127.0.0.1",
                "MYSQL_PORT=3306",
                "MYSQL_DATABASE=test_db",
                "MYSQL_USERNAME=test_user",
                "MYSQL_PASSWORD=test_password",
                "ES_URL=http://127.0.0.1:9200",
                "MILVUS_URI=http://127.0.0.1:19530",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.development").write_text(
        "\n".join(
            [
                "ES_URL=http://122.152.231.116:9200",
                "MILVUS_URI=http://122.152.231.116:19530",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ES_URL", raising=False)
    monkeypatch.delenv("MILVUS_URI", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.app_env == "development"
    assert settings.es_url == "http://122.152.231.116:9200"
    assert settings.milvus_uri == "http://122.152.231.116:19530"
    get_settings.cache_clear()
