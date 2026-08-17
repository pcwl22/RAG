from pathlib import Path

import pytest
import yaml

from app.utils.config import (
    get_config_section,
    resolve_queue_provider,
    validate_runtime_config,
)


def test_base_llm_profile_uses_current_provider_schema():
    config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    text_config = config["llm"]["text"]
    provider = text_config["provider"]

    assert provider == "openai_compatible"
    assert text_config[provider]["model_name"] == "deepseek-chat"
    assert text_config[provider]["api_key"] == "${DEEPSEEK_API_KEY}"


def test_frontend_proxy_exposes_readiness_endpoint():
    config_path = Path(__file__).resolve().parents[2] / "rag-frontend" / "nginx" / "default.conf.template"
    config = config_path.read_text(encoding="utf-8")

    assert "location = /health/ready" in config
    assert "proxy_pass ${RAG_API_UPSTREAM}/health/ready;" in config


def test_frontend_proxy_exposes_liveness_endpoint():
    config_path = Path(__file__).resolve().parents[2] / "rag-frontend" / "nginx" / "default.conf.template"
    config = config_path.read_text(encoding="utf-8")

    assert "location = /health/live" in config
    assert "proxy_pass ${RAG_API_UPSTREAM}/health/live;" in config


def test_frontend_csp_allows_oidc_session_iframe():
    config_path = Path(__file__).resolve().parents[2] / "rag-frontend" / "nginx" / "default.conf.template"
    config = config_path.read_text(encoding="utf-8")

    assert "frame-src 'self' ${OIDC_CONNECT_SRC}" in config


def test_frontend_proxy_matches_upload_and_streaming_contracts():
    config_path = Path(__file__).resolve().parents[2] / "rag-frontend" / "nginx" / "default.conf.template"
    config = config_path.read_text(encoding="utf-8")

    assert "client_max_body_size 100m;" in config
    assert "location ^~ /api/v1/chat/stream" in config
    assert "proxy_read_timeout 300s;" in config
    assert "limit_req zone=rag_api_per_ip" in config
    assert "limit_req zone=rag_stream_per_ip" in config


def test_runtime_profiles_pass_strict_validation(monkeypatch):
    monkeypatch.delenv("QUEUE_PROVIDER", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_REQUIRE_TLS", raising=False)
    config_dir = Path(__file__).resolve().parents[1] / "config"

    for name in ("base.yaml", "laptop.yaml"):
        config = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
        if name == "base.yaml":
            config["postgres"]["user"] = "rag_runtime"
            config["postgres"]["password"] = "test-runtime-password"
            config["redis"]["url"] = "redis://:test-redis-password@redis:6379/0"
        validate_runtime_config(config)


def test_base_profile_requires_runtime_postgres_password():
    config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["postgres"]["user"] = "rag_runtime"
    config["postgres"]["password"] = ""

    try:
        validate_runtime_config(config)
    except ValueError as exc:
        assert "POSTGRES_APP_PASSWORD" in str(exc)
    else:
        raise AssertionError("runtime PostgreSQL user without password must fail fast")


def test_base_profile_requires_authenticated_redis_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["postgres"]["user"] = "rag_runtime"
    config["postgres"]["password"] = "test-runtime-password"
    config["redis"]["url"] = "redis://redis:6379/0"

    try:
        validate_runtime_config(config)
    except ValueError as exc:
        assert "Redis password" in str(exc)
    else:
        raise AssertionError("production Redis URL without password must fail fast")


def test_base_profile_can_require_redis_tls(monkeypatch):
    monkeypatch.setenv("REDIS_REQUIRE_TLS", "true")
    config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["postgres"]["user"] = "rag_runtime"
    config["postgres"]["password"] = "test-runtime-password"
    config["redis"]["url"] = "redis://:test-redis-password@redis:6379/0"

    try:
        validate_runtime_config(config)
    except ValueError as exc:
        assert "rediss://" in str(exc)
    else:
        raise AssertionError("REDIS_REQUIRE_TLS must reject redis:// URLs")


def test_root_compose_redis_supports_password_authentication():
    config_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    config = config_path.read_text(encoding="utf-8")

    assert "REDIS_PASSWORD: ${REDIS_PASSWORD:-}" in config
    assert "--requirepass" in config
    assert "REDISCLI_AUTH" in config
    assert "redis-cli -a" not in config
    assert "COMPOSE_REDIS_URL" in config


def test_unknown_queue_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("QUEUE_PROVIDER", "celrey")

    try:
        resolve_queue_provider({})
    except ValueError as exc:
        assert "Unsupported queue provider" in str(exc)
    else:
        raise AssertionError("invalid queue provider must fail fast")


def test_get_config_section_returns_nested_mapping(monkeypatch):
    import app.utils.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_settings",
        lambda: {"rag": {"retrieval": {"top_k": 5}}},
    )

    assert get_config_section("rag", "retrieval") == {"top_k": 5}
    # A missing level yields {} so callers can keep using .get() defaults.
    assert get_config_section("rag", "absent") == {}
    assert get_config_section("absent", "deeper") == {}


def test_get_config_section_rejects_non_mapping_section(monkeypatch):
    """A scalar where a section is expected is a config error, not a default."""
    import app.utils.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_settings",
        lambda: {"rag": {"retrieval": "should-have-been-a-mapping"}},
    )

    with pytest.raises(TypeError, match="rag.retrieval"):
        get_config_section("rag", "retrieval")
