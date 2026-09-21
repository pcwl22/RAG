from pathlib import Path

import pytest
import yaml

from app.utils.config import (
    get_config_section,
    normalize_openai_base_url,
    resolve_object_storage_config,
    resolve_queue_provider,
    validate_runtime_config,
)


def test_base_llm_profile_uses_current_provider_schema():
    config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    text_config = config["llm"]["text"]
    provider = text_config["provider"]

    assert provider == "openai_compatible"
    assert text_config[provider]["model_name"] == "${DEEPSEEK_MODEL}"
    assert text_config[provider]["api_key"] == "${DEEPSEEK_API_KEY}"
    assert text_config[provider]["base_url"] == "${DEEPSEEK_API_URL}"


def test_openai_base_url_normalization_is_explicit_and_safe():
    assert normalize_openai_base_url("https://proxy.example") == "https://proxy.example/v1"
    assert (
        normalize_openai_base_url("https://proxy.example/openai/v1/")
        == "https://proxy.example/openai/v1"
    )

    with pytest.raises(ValueError, match="absolute http://"):
        normalize_openai_base_url("proxy.example")


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

    assert "client_max_body_size 105m;" in config
    assert "location ^~ /api/v1/chat/stream" in config
    assert "proxy_read_timeout 300s;" in config
    assert "limit_req zone=rag_api_per_ip" in config
    assert "limit_req zone=rag_stream_per_ip" in config


def test_frontend_only_trusts_the_configured_ingress_proxy_cidr():
    config_path = Path(__file__).resolve().parents[2] / "rag-frontend" / "nginx" / "default.conf.template"
    config = config_path.read_text(encoding="utf-8")

    assert "real_ip_header X-Forwarded-For;" in config
    assert "real_ip_recursive on;" in config
    assert "set_real_ip_from ${TRUSTED_PROXY_CIDR};" in config


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


def test_runtime_profiles_bound_reranker_attention_cost():
    config_dir = Path(__file__).resolve().parents[1] / "config"

    for name in ("base.yaml", "laptop.yaml"):
        config = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
        max_length = config["reranker"]["max_length"]

        assert 512 <= max_length <= 1024, name


def test_postgres_connection_can_be_overridden_without_editing_yaml(monkeypatch):
    import app.utils.config as config_module

    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "15433")
    monkeypatch.setenv("POSTGRES_DB", "rag_test")
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-only")
    config = {
        "postgres": {
            "host": "localhost",
            "port": 15432,
            "database": "rag_db",
            "user": "configured-user",
            "password": "configured-password",
        }
    }

    config_module._apply_postgres_env_overrides(config)

    assert config["postgres"] == {
        "host": "127.0.0.1",
        "port": "15433",
        "database": "rag_test",
        "user": "postgres",
        "password": "test-only",
    }


def test_laptop_profile_separates_host_port_from_compose_port(monkeypatch):
    import app.utils.config as config_module

    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.delenv("RAG_POSTGRES_PORT", raising=False)
    config = {"postgres": {"port": 15432}}

    config_module._apply_postgres_env_overrides(
        config,
        port_env_key="RAG_POSTGRES_PORT",
    )

    assert config["postgres"]["port"] == 15432

    monkeypatch.setenv("RAG_POSTGRES_PORT", "15433")
    config_module._apply_postgres_env_overrides(
        config,
        port_env_key="RAG_POSTGRES_PORT",
    )
    assert config["postgres"]["port"] == "15433"


def test_workloads_can_select_cpu_or_gpu_without_editing_shared_yaml(monkeypatch):
    import app.utils.config as config_module

    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("RERANKER_DEVICE", "cpu")
    monkeypatch.setenv("PDF_DEVICE", "cpu")
    monkeypatch.setenv("MULTIMODAL_DEVICE", "cpu")
    config = {
        "embedding": {"device": "cuda"},
        "reranker": {"device": "cuda"},
        "document_processing": {"pdf": {"device": "cuda"}},
        "multimodal": {"image": {"device": "cuda"}},
    }

    config_module._apply_accelerator_env_overrides(config)

    assert config["embedding"]["device"] == "cpu"
    assert config["reranker"]["device"] == "cpu"
    assert config["document_processing"]["pdf"]["device"] == "cpu"
    assert config["multimodal"]["image"]["device"] == "cpu"


def test_protected_release_snapshot_overrides_ambient_runner_values(tmp_path, monkeypatch):
    import app.utils.config as config_module

    env_file = tmp_path / "release.env"
    env_file.write_text("DEEPSEEK_MODEL=file-model\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_MODEL", "ambient-model")

    from app.utils.strict_dotenv import apply_release_env_file

    apply_release_env_file(env_file)

    assert config_module.os.environ["DEEPSEEK_MODEL"] == "file-model"


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
    config["postgres"]["sslmode"] = "verify-full"
    config["postgres"]["sslrootcert"] = "/etc/ssl/certs/internal-ca.pem"
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
    config["postgres"]["sslmode"] = "verify-full"
    config["postgres"]["sslrootcert"] = "/etc/ssl/certs/internal-ca.pem"
    config["redis"]["url"] = "redis://:test-redis-password@redis:6379/0"

    try:
        validate_runtime_config(config)
    except ValueError as exc:
        assert "rediss://" in str(exc)
    else:
        raise AssertionError("REDIS_REQUIRE_TLS must reject redis:// URLs")


def test_secure_mode_requires_redis_tls(monkeypatch):
    monkeypatch.setenv("RAG_SECURE_MODE", "true")
    monkeypatch.delenv("REDIS_REQUIRE_TLS", raising=False)
    config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["postgres"]["user"] = "rag_runtime"
    config["postgres"]["password"] = "test-runtime-password"
    config["postgres"]["sslmode"] = "verify-full"
    config["postgres"]["sslrootcert"] = "/etc/ssl/certs/internal-ca.pem"
    config["redis"]["url"] = "redis://:test-redis-password@redis:6379/0"

    with pytest.raises(ValueError, match="rediss://"):
        validate_runtime_config(config)


def test_root_compose_redis_supports_password_authentication():
    config_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    config = config_path.read_text(encoding="utf-8")

    assert "REDIS_PASSWORD: ${REDIS_PASSWORD:-}" in config
    assert "--requirepass" in config
    assert "REDISCLI_AUTH" in config
    assert "redis-cli -a" not in config
    assert "COMPOSE_REDIS_URL" in config


def test_root_compose_postgres_uses_verify_full_tls():
    config_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    config = config_path.read_text(encoding="utf-8")

    assert config.count("POSTGRES_SSLMODE: verify-full") == 3
    assert config.count("postgres_tls:/etc/rag/postgres-tls:ro") == 3
    assert config.count(
        "POSTGRES_SSLROOTCERT: /etc/rag/postgres-tls/server.crt"
    ) == 3
    assert "postgres-cert-init:" in config
    assert 'subjectAltName=DNS:postgres' in config
    assert "ssl=on" in config
    assert "ssl_cert_file=/etc/postgresql/tls/server.crt" in config
    assert "ssl_key_file=/etc/postgresql/tls/server.key" in config


def test_unknown_queue_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("QUEUE_PROVIDER", "celrey")

    try:
        resolve_queue_provider({})
    except ValueError as exc:
        assert "Unsupported queue provider" in str(exc)
    else:
        raise AssertionError("invalid queue provider must fail fast")


def test_reconciliation_environment_override_is_range_checked(monkeypatch):
    monkeypatch.setenv("RAG_ENV", "laptop")
    monkeypatch.setenv("UPLOAD_PRODUCER_TIMEOUT_SECONDS", "0")
    config = {
        "app": {"debug": True},
        "redis": {"enabled": False},
        "queue": {"provider": "memory", "reconciliation": {}},
    }

    with pytest.raises(ValueError, match="UPLOAD_PRODUCER_TIMEOUT_SECONDS"):
        validate_runtime_config(config)


def test_base_memory_mode_allows_redis_to_be_absent(monkeypatch):
    monkeypatch.setenv("RAG_ENV", "base")
    monkeypatch.setenv("QUEUE_PROVIDER", "memory")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("RAG_SECURE_MODE", raising=False)
    config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["postgres"]["user"] = "rag_runtime"
    config["postgres"]["password"] = "test-runtime-password"
    config["postgres"]["sslmode"] = "verify-full"
    config["postgres"]["sslrootcert"] = "/etc/ssl/certs/internal-ca.pem"

    validate_runtime_config(config)


def test_base_profile_enforces_verified_postgres_tls_without_secure_flag(monkeypatch):
    monkeypatch.setenv("RAG_ENV", "base")
    monkeypatch.setenv("QUEUE_PROVIDER", "memory")
    monkeypatch.delenv("RAG_SECURE_MODE", raising=False)
    config = {
        "app": {"debug": False},
        "postgres": {
            "user": "rag_runtime",
            "password": "test-runtime-password",
            "sslmode": "disable",
        },
        "redis": {"enabled": False},
        "rag": {"generation": {"answer_contract": {"enabled": True}}},
        "queue": {"provider": "memory", "reconciliation": {}},
    }

    with pytest.raises(ValueError, match="verify-full"):
        validate_runtime_config(config)


def test_s3_timeout_environment_override_has_an_upper_bound(monkeypatch):
    monkeypatch.setenv("S3_READ_TIMEOUT_SECONDS", "301")

    with pytest.raises(ValueError, match="S3_READ_TIMEOUT_SECONDS"):
        resolve_object_storage_config({"object_storage": {}})


def test_celery_requires_a_second_postgres_connection_for_worker_lease(monkeypatch):
    monkeypatch.setenv("QUEUE_PROVIDER", "celery")
    config = {
        "app": {"debug": True},
        "postgres": {"min_pool_size": 1, "max_pool_size": 1},
        "redis": {"enabled": False},
        "queue": {"provider": "celery", "reconciliation": {"enabled": True}},
    }

    with pytest.raises(ValueError, match="at least 2"):
        validate_runtime_config(config)


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
