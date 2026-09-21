import base64
from pathlib import Path

import pytest

from app.utils.strict_dotenv import apply_release_env_file
from scripts.validate_production_env import (
    load_env_file,
    load_release_env_file,
    resolve_validation_values,
    validate_production_values,
)


def _valid_values() -> dict[str, str]:
    canary_sample = (
        "本文件仅用于受保护生产发布的租户隔离和检索验证。"
        "它不包含真实用户、案件或业务数据，且每次发布都会写入随机验证标记。"
    ).encode()
    return {
        "RELEASE_ID": "123e4567-e89b-42d3-a456-426614174000",
        "CANARY_OIDC_TOKEN_URL": "https://id.corp.test/realms/rag/protocol/openid-connect/token",
        "CANARY_OIDC_CLIENT_AUTH_METHOD": "client_secret_post",
        "CANARY_PRIMARY_CLIENT_ID": "rag-release-canary-primary",
        "CANARY_PRIMARY_CLIENT_SECRET": "primary-canary-secret-1234",
        "CANARY_SECONDARY_CLIENT_ID": "rag-release-canary-secondary",
        "CANARY_SECONDARY_CLIENT_SECRET": "secondary-canary-secret-5678",
        "CANARY_UPLOAD_CONTENT_B64": base64.b64encode(canary_sample).decode("ascii"),
        "CANARY_NO_ANSWER_QUERY": "无线电频率许可",
        "CANARY_NO_ANSWER_EXPECTED_TEXT": "当前知识库没有找到足够相关的信息",
        "RAG_ENV": "base",
        "POSTGRES_AUTO_MIGRATE": "false",
        "RAG_API_KEY": "a" * 40,
        "RAG_METRICS_TOKEN": "m" * 40,
        "POSTGRES_PASSWORD": "admin-password-1234",
        "POSTGRES_ADMIN_USER": "postgres",
        "POSTGRES_RUNTIME_USER": "rag_runtime",
        "POSTGRES_APP_PASSWORD": "runtime-password-5678",
        "POSTGRES_HOST": "db.corp.test",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "rag",
        "POSTGRES_SSLMODE": "verify-full",
        "POSTGRES_SSLROOTCERT": "/etc/ssl/certs/ca-certificates.crt",
        "REDIS_PASSWORD": "redis-password-1234",
        "COMPOSE_REDIS_URL": "rediss://:redis-password-1234@redis.corp.test:6379/0",
        "REDIS_REQUIRE_TLS": "true",
        "QUEUE_PROVIDER": "celery",
        "COMPOSE_OIDC_ENABLED": "true",
        "OIDC_ENABLED": "true",
        "OIDC_ISSUER": "https://id.corp.test/realms/rag",
        "OIDC_AUDIENCE": "rag-api",
        "OIDC_JWKS_URL": "https://id.corp.test/realms/rag/certs",
        "VITE_OIDC_ISSUER": "https://id.corp.test/realms/rag",
        "VITE_OIDC_CLIENT_ID": "rag-frontend",
        "VITE_OIDC_AUDIENCE": "rag-api",
        "RAG_SECURE_MODE": "true",
        "RAG_SERVICE_TENANT_ID": "00000000-0000-0000-0000-000000000001",
        "RAG_SERVICE_ROLES": "viewer",
        "DEEPSEEK_API_KEY": "d" * 40,
        "DEEPSEEK_API_URL": "https://llm.corp.test/v1",
        "DEEPSEEK_MODEL": "deepseek-v4-flash-0731",
        "REDIS_URL": "rediss://:redis-password-1234@redis.corp.test:6379/0",
        "OBJECT_STORAGE_ENABLED": "true",
        "S3_ENDPOINT_URL": "https://s3.corp.test",
        "S3_BUCKET": "industrial-rag",
        "S3_REGION": "us-east-1",
        "S3_ACCESS_KEY_ID": "s3-access-key",
        "S3_SECRET_ACCESS_KEY": "s3-secret-key-1234",
        "S3_REQUIRE_TLS": "true",
        "S3_ADDRESSING_STYLE": "path",
        "S3_FAILED_PREFIX": "failed",
        "MODEL_BUNDLE_IMAGE": "registry.corp.test/rag/model-bundle@sha256:" + "a" * 64,
        "MODEL_BUNDLE_DIGEST": "sha256:" + "a" * 64,
        "MODEL_MANIFEST_SHA256": "sha256:" + "b" * 64,
        "API_IMAGE": "registry.corp.test/rag/api@sha256:" + "1" * 64,
        "WORKER_IMAGE": "registry.corp.test/rag/worker@sha256:" + "2" * 64,
        "FRONTEND_IMAGE": "registry.corp.test/rag/frontend@sha256:" + "3" * 64,
        "INGRESS_HOST": "rag.corp.test",
        "INGRESS_PROXY_CIDR": "10.244.0.0/16",
        "OIDC_CONNECT_SRC": "https://id.corp.test",
    }


def test_valid_production_environment_passes():
    assert validate_production_values(_valid_values()) == []


def test_repository_example_environment_passes_placeholder_validation():
    repository_root = Path(__file__).resolve().parents[2]
    values = resolve_validation_values(
        repository_root / ".env.example",
        process_values={},
    )

    assert validate_production_values(
        values,
        allow_placeholders=True,
        allow_local_http=True,
    ) == []


def test_production_environment_rejects_placeholder_and_insecure_transport():
    values = _valid_values()
    values["RAG_API_KEY"] = "change-me"
    values["COMPOSE_REDIS_URL"] = "redis://:redis-password-1234@redis:6379/0"
    values["OIDC_ISSUER"] = "http://localhost:18080/realms/rag"

    errors = validate_production_values(values)

    assert any("RAG_API_KEY must not" in error for error in errors)
    assert any("rediss://" in error for error in errors)
    assert any("OIDC_ISSUER must use https://" in error for error in errors)
    assert all("redis-password-1234" not in error for error in errors)


def test_local_acceptance_requires_explicit_transport_override():
    values = _valid_values()
    values["REDIS_REQUIRE_TLS"] = "false"
    values["COMPOSE_REDIS_URL"] = "redis://:redis-password-1234@redis:6379/0"
    values["OIDC_ISSUER"] = "http://localhost:18080/realms/rag"
    values["OIDC_JWKS_URL"] = "http://localhost:18080/realms/rag/certs"
    values["CANARY_OIDC_TOKEN_URL"] = (
        "http://localhost:18080/realms/rag/protocol/openid-connect/token"
    )
    values["VITE_OIDC_ISSUER"] = values["OIDC_ISSUER"]
    values["OIDC_CONNECT_SRC"] = "http://localhost:18080"
    values["RAG_SECURE_MODE"] = "false"

    assert validate_production_values(values, allow_local_http=True) == []


def test_production_environment_requires_secure_mode():
    values = _valid_values()
    values.pop("RAG_SECURE_MODE")

    errors = validate_production_values(values)

    assert "RAG_SECURE_MODE must be true in production" in errors


def test_database_passwords_must_be_distinct():
    values = _valid_values()
    values["POSTGRES_APP_PASSWORD"] = values["POSTGRES_PASSWORD"]

    errors = validate_production_values(values)

    assert "POSTGRES_PASSWORD and POSTGRES_APP_PASSWORD must be different" in errors


def test_production_database_transport_requires_hostname_verification():
    values = _valid_values()
    values["POSTGRES_SSLMODE"] = "require"

    errors = validate_production_values(values)

    assert "POSTGRES_SSLMODE must be verify-full in production" in errors


@pytest.mark.parametrize(
    "duplicate_line",
    [
        "API_IMAGE=second",
        " API_IMAGE =second",
        "export API_IMAGE=second",
        "export\tAPI_IMAGE=second",
        "'API_IMAGE'=second",
    ],
)
def test_release_env_rejects_normalized_duplicate_keys(
    tmp_path: Path, duplicate_line: str
):
    env_file = tmp_path / "release.env"
    env_file.write_text(
        f"API_IMAGE=first\n{duplicate_line}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate dotenv key 'API_IMAGE'"):
        load_release_env_file(env_file)


def test_release_snapshot_is_file_authoritative(tmp_path: Path):
    env_file = tmp_path / "release.env"
    env_file.write_text("API_IMAGE=file-image\n", encoding="utf-8")

    release_values = resolve_validation_values(
        env_file,
        release_snapshot=True,
        process_values={"API_IMAGE": "ambient-image"},
    )
    local_values = resolve_validation_values(
        env_file,
        process_values={"API_IMAGE": "ambient-image"},
    )

    assert release_values["API_IMAGE"] == "file-image"
    assert local_values["API_IMAGE"] == "ambient-image"


def test_dotenv_unique_export_whitespace_and_quotes_remain_compatible(tmp_path: Path):
    env_file = tmp_path / "release.env"
    env_file.write_text(
        " export API_IMAGE = 'registry.test/api@sha256:" + "a" * 64 + "'\n",
        encoding="utf-8",
    )

    expected = "registry.test/api@sha256:" + "a" * 64
    assert load_env_file(env_file)["API_IMAGE"] == expected
    assert load_release_env_file(env_file)["API_IMAGE"] == expected


@pytest.mark.parametrize(
    "line",
    [
        "API_IMAGE=value # ambiguous comment",
        "API_IMAGE='unbalanced",
        "API_IMAGE",
    ],
)
def test_release_env_rejects_ambiguous_value_syntax(tmp_path: Path, line: str):
    env_file = tmp_path / "release.env"
    env_file.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_release_env_file(env_file)


def test_release_snapshot_clears_ambient_overrides_and_maps_runtime_database(tmp_path: Path):
    env_file = tmp_path / "release.env"
    env_file.write_text(
        "POSTGRES_RUNTIME_USER=rag_runtime_v2\n"
        "POSTGRES_APP_PASSWORD=runtime-password\n"
        "RELEASE_ID=123e4567-e89b-42d3-a456-426614174002\n"
        "DEEPSEEK_MODEL=approved-model\n",
        encoding="utf-8",
    )
    environment = {
        "RAG_ENV_FILE": str(env_file),
        "RAG_RELEASE_SNAPSHOT": "1",
        "RAGAS_BASE_URL": "https://ambient.invalid/v1",
        "POSTGRES_USER": "ambient-owner",
        "POSTGRES_PASSWORD": "ambient-owner-password",
        "DEEPSEEK_MODEL": "ambient-model",
        "RELEASE_ID": "ambient-release-id",
        "CANARY_OIDC_SCOPE": "ambient-scope",
        "RUN_POSTGRES_INTEGRATION": "1",
    }

    apply_release_env_file(env_file, environ=environment)

    assert environment["POSTGRES_USER"] == "rag_runtime_v2"
    assert environment["POSTGRES_PASSWORD"] == "runtime-password"
    assert environment["DEEPSEEK_MODEL"] == "approved-model"
    assert environment["RELEASE_ID"] == "123e4567-e89b-42d3-a456-426614174002"
    assert "CANARY_OIDC_SCOPE" not in environment
    assert "RAGAS_BASE_URL" not in environment
    assert environment["RAG_ENV_FILE"] == str(env_file)
    assert environment["RUN_POSTGRES_INTEGRATION"] == "1"


def test_production_environment_rejects_shell_metacharacters_in_image_reference():
    values = _valid_values()
    values["API_IMAGE"] = "registry.test/$(id)/api@sha256:" + "a" * 64

    errors = validate_production_values(values)

    assert "API_IMAGE must be referenced by digest" in errors


def test_frontend_csp_origin_must_match_runtime_oidc_issuer():
    values = _valid_values()
    values["OIDC_CONNECT_SRC"] = "https://different-id.corp.test"

    errors = validate_production_values(values)

    assert "OIDC_CONNECT_SRC must match the VITE_OIDC_ISSUER origin" in errors


def test_functional_canary_contract_is_fail_closed_and_origin_bound():
    values = _valid_values()
    values.pop("CANARY_SECONDARY_CLIENT_SECRET")
    values["CANARY_OIDC_TOKEN_URL"] = "https://untrusted.corp.test/token"

    errors = validate_production_values(values)

    assert "CANARY_SECONDARY_CLIENT_SECRET is required" in errors
    assert "CANARY_OIDC_TOKEN_URL must share the OIDC_ISSUER origin" in errors


def test_functional_canary_clients_and_sample_must_be_distinct_and_real():
    values = _valid_values()
    values["CANARY_SECONDARY_CLIENT_ID"] = values["CANARY_PRIMARY_CLIENT_ID"]
    values["CANARY_SECONDARY_CLIENT_SECRET"] = values["CANARY_PRIMARY_CLIENT_SECRET"]
    values["CANARY_UPLOAD_CONTENT_B64"] = base64.b64encode(b"too short").decode("ascii")

    errors = validate_production_values(values)

    assert any("CLIENT_ID" in error and "different" in error for error in errors)
    assert any("CLIENT_SECRET" in error and "different" in error for error in errors)
    assert any("80 bytes" in error for error in errors)


def test_release_id_must_be_canonical_uuid_v4():
    values = _valid_values()
    values["RELEASE_ID"] = "123e4567-e89b-12d3-a456-426614174000"

    assert "RELEASE_ID must be a canonical lowercase UUIDv4" in (
        validate_production_values(values)
    )


def test_canary_no_answer_query_must_match_audited_scope_rule():
    values = _valid_values()
    values["CANARY_NO_ANSWER_QUERY"] = "普通且可能命中文档的问题"

    assert "CANARY_NO_ANSWER_QUERY must match an approved out-of-scope signal rule" in (
        validate_production_values(values)
    )
