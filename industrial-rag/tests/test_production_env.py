from scripts.validate_production_env import validate_production_values


def _valid_values() -> dict[str, str]:
    return {
        "RAG_API_KEY": "a" * 40,
        "RAG_METRICS_TOKEN": "m" * 40,
        "POSTGRES_PASSWORD": "admin-password-1234",
        "POSTGRES_APP_PASSWORD": "runtime-password-5678",
        "REDIS_PASSWORD": "redis-password-1234",
        "COMPOSE_REDIS_URL": "rediss://:redis-password-1234@redis.corp.test:6379/0",
        "REDIS_REQUIRE_TLS": "true",
        "QUEUE_PROVIDER": "celery",
        "COMPOSE_OIDC_ENABLED": "true",
        "OIDC_ISSUER": "https://id.corp.test/realms/rag",
        "OIDC_AUDIENCE": "rag-api",
        "OIDC_JWKS_URL": "https://id.corp.test/realms/rag/certs",
        "RAG_SERVICE_TENANT_ID": "00000000-0000-0000-0000-000000000001",
    }


def test_valid_production_environment_passes():
    assert validate_production_values(_valid_values()) == []


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

    assert validate_production_values(values, allow_local_http=True) == []


def test_database_passwords_must_be_distinct():
    values = _valid_values()
    values["POSTGRES_APP_PASSWORD"] = values["POSTGRES_PASSWORD"]

    errors = validate_production_values(values)

    assert "POSTGRES_PASSWORD and POSTGRES_APP_PASSWORD must be different" in errors
