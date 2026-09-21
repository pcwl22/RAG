import asyncio
import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse

from app.auth import OIDCValidator, current_principal
from app.main import app
from app.security import authentication_middleware, validate_security_config


def test_production_security_rejects_wildcard_credentials(monkeypatch):
    # validate_security_config checks that authentication exists before it
    # reaches the CORS rules, so a valid API key is required for this test to
    # exercise the wildcard check rather than tripping the earlier guard.
    monkeypatch.setenv("RAG_API_KEY", "w" * 40)
    monkeypatch.setenv("RAG_METRICS_TOKEN", "m" * 40)
    try:
        validate_security_config(
            {
                "app": {"debug": False},
                "security": {
                    "api_key": {"enabled": True},
                    "oidc": {"enabled": False},
                    "cors": {"allow_origins": ["*"], "allow_credentials": True},
                },
            }
        )
    except RuntimeError as exc:
        assert "wildcard" in str(exc)
    else:
        raise AssertionError("unsafe CORS configuration was accepted")


def test_production_security_requires_authentication(monkeypatch):
    monkeypatch.delenv("OIDC_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="requires API key or OIDC"):
        validate_security_config(
            {
                "app": {"debug": False},
                "security": {
                    "api_key": {"enabled": False},
                    "oidc": {"enabled": False},
                    "cors": {"allow_origins": []},
                },
            }
        )


def test_production_security_rejects_weak_api_key(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "short-key")
    with pytest.raises(RuntimeError, match="at least 32"):
        validate_security_config(
            {
                "app": {"debug": False},
                "security": {
                    "api_key": {"enabled": True},
                    "oidc": {"enabled": False},
                    "cors": {"allow_origins": []},
                },
            }
        )


def test_production_security_requires_independent_metrics_token(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "a" * 40)
    monkeypatch.delenv("RAG_METRICS_TOKEN", raising=False)
    config = {
        "app": {"debug": False},
        "security": {
            "api_key": {"enabled": True},
            "oidc": {"enabled": False},
            "cors": {"allow_origins": []},
        },
        "monitoring": {"prometheus": {"enabled": True}},
    }

    with pytest.raises(RuntimeError, match="RAG_METRICS_TOKEN"):
        validate_security_config(config)

    monkeypatch.setenv("RAG_METRICS_TOKEN", "a" * 40)
    with pytest.raises(RuntimeError, match="different"):
        validate_security_config(config)


def test_debug_bypass_is_confined_to_loopback():
    """Debug mode skips authentication, so it must not be network-reachable."""
    def debug_config(host):
        return {
            "app": {"debug": True, "host": host},
            # Deliberately unsafe: no authentication and wildcard CORS with
            # credentials. Debug mode is expected to tolerate this on loopback.
            "security": {
                "api_key": {"enabled": False},
                "oidc": {"enabled": False},
                "cors": {"allow_origins": ["*"], "allow_credentials": True},
            },
        }

    for host in ("127.0.0.1", "localhost", "::1", "[::1]", ""):
        validate_security_config(debug_config(host))

    for host in ("0.0.0.0", "::", "10.0.0.5", "api.internal.example.com"):
        with pytest.raises(RuntimeError, match="non-loopback host"):
            validate_security_config(debug_config(host))


def test_metrics_token_required_when_prometheus_default_applies(monkeypatch):
    """An omitted monitoring section must not exempt the metrics token.

    app.main instruments the app when monitoring.prometheus.enabled is absent,
    so validate_security_config has to treat the same omission as enabled.
    Otherwise a config without a monitoring section exposes the endpoint while
    skipping the token requirement entirely.
    """
    monkeypatch.setenv("RAG_API_KEY", "a" * 40)
    monkeypatch.delenv("RAG_METRICS_TOKEN", raising=False)

    # Both spellings of "not configured": no monitoring key at all, and a
    # prometheus block that omits `enabled`.
    for monitoring in ({}, {"prometheus": {}}):
        with pytest.raises(RuntimeError, match="RAG_METRICS_TOKEN"):
            validate_security_config(
                {
                    "app": {"debug": False},
                    "security": {
                        "api_key": {"enabled": True},
                        "oidc": {"enabled": False},
                        "cors": {"allow_origins": []},
                    },
                    "monitoring": monitoring,
                }
            )

    # An explicit opt-out still skips the requirement.
    validate_security_config(
        {
            "app": {"debug": False},
            "security": {
                "api_key": {"enabled": True},
                "oidc": {"enabled": False},
                "cors": {"allow_origins": []},
            },
            "monitoring": {"prometheus": {"enabled": False}},
        }
    )


def test_oidc_rejects_symmetric_signature_algorithms(monkeypatch):
    monkeypatch.delenv("OIDC_ENABLED", raising=False)
    validator = OIDCValidator(
        {
            "security": {
                "oidc": {
                    "enabled": True,
                    "issuer": "https://issuer.example/realms/rag",
                    "audience": "rag-api",
                    "jwks_url": "https://issuer.example/certs",
                    "algorithms": ["HS256"],
                }
            }
        }
    )
    with pytest.raises(RuntimeError, match="asymmetric"):
        validator.validate_configuration()


def test_secure_mode_requires_https_oidc_endpoints(monkeypatch):
    monkeypatch.setenv("RAG_SECURE_MODE", "true")
    monkeypatch.delenv("OIDC_ENABLED", raising=False)
    validator = OIDCValidator(
        {
            "security": {
                "oidc": {
                    "enabled": True,
                    "issuer": "http://id.internal/realms/rag",
                    "audience": "rag-api",
                    "jwks_url": "http://id.internal/certs",
                }
            }
        }
    )

    with pytest.raises(RuntimeError, match="OIDC_ISSUER must use https"):
        validator.validate_configuration()


def test_api_key_middleware_protects_api_routes(monkeypatch):
    async def run():
        monkeypatch.setenv("RAG_API_KEY", "test-key")
        middleware = authentication_middleware(
            {"security": {"api_key": {"enabled": True, "header_name": "X-API-Key"}}}
        )
        protected = await middleware(
            httpx.Request("GET", "http://test/api/v1/documents"),
            lambda request: asyncio.sleep(0, result=httpx.Response(200)),
        )
        assert protected.status_code == 401

    asyncio.run(run())


def test_api_key_middleware_also_protects_operational_endpoints(monkeypatch):
    async def run():
        monkeypatch.setenv("RAG_API_KEY", "test-key")
        middleware = authentication_middleware(
            {"security": {"api_key": {"enabled": True, "header_name": "X-API-Key"}}}
        )
        for path in ("/docs", "/openapi.json", "/metrics"):
            response = await middleware(
                httpx.Request("GET", f"http://test{path}"),
                lambda request: asyncio.sleep(0, result=httpx.Response(200)),
            )
            assert response.status_code == 401

    asyncio.run(run())


def test_metrics_endpoint_accepts_only_dedicated_token_when_configured(monkeypatch):
    async def run():
        monkeypatch.setenv("RAG_API_KEY", "api-key")
        monkeypatch.setenv("RAG_METRICS_TOKEN", "metrics-key")
        middleware = authentication_middleware(
            {
                "security": {"api_key": {"enabled": True, "header_name": "X-API-Key"}},
                "monitoring": {
                    "prometheus": {
                        "path": "/internal/metrics",
                        "header_name": "X-Metrics-Token",
                    }
                },
            }
        )

        def request(headers):
            return Request(
                {
                    "type": "http",
                    "http_version": "1.1",
                    "method": "GET",
                    "scheme": "http",
                    "path": "/internal/metrics",
                    "raw_path": b"/internal/metrics",
                    "query_string": b"",
                    "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()],
                    "client": ("test", 123),
                    "server": ("test", 80),
                }
            )

        async def endpoint(_request):
            return httpx.Response(200)

        assert (await middleware(request({"X-API-Key": "api-key"}), endpoint)).status_code == 401
        assert (
            await middleware(request({"X-Metrics-Token": "metrics-key"}), endpoint)
        ).status_code == 200

    asyncio.run(run())


def test_health_endpoints_are_public():
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            assert (await client.get("/health/live")).status_code == 200

    asyncio.run(run())


def test_rejected_requests_still_carry_cors_headers(monkeypatch):
    """A browser must be able to read the real status of an auth failure.

    CORS has to remain outside authentication; otherwise a 401 is returned
    without ``Access-Control-Allow-Origin`` and the browser reports an opaque
    CORS error instead of surfacing the authentication failure.
    """
    # The middleware resolves RAG_API_KEY from the environment first, and
    # importing the app loads a developer .env. Pin it so the assertion tests
    # the middleware order rather than the local environment.
    monkeypatch.setenv("RAG_API_KEY", "probe-key")

    async def run():
        origin = "https://ui.example.com"
        probe = FastAPI()

        @probe.get("/api/v1/documents")
        async def probe_route():
            return {"ok": True}

        # Same registration order as app/main.py.
        probe.middleware("http")(
            authentication_middleware(
                {
                    "security": {
                        "api_key": {
                            "enabled": True,
                            "value": "probe-key",
                            "header_name": "X-API-Key",
                        }
                    }
                }
            )
        )
        probe.add_middleware(
            CORSMiddleware,
            allow_origins=[origin],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        transport = httpx.ASGITransport(app=probe)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            rejected = await client.get("/api/v1/documents", headers={"Origin": origin})
            assert rejected.status_code == 401
            assert rejected.headers.get("access-control-allow-origin") == origin

            allowed = await client.get(
                    "/api/v1/documents",
                headers={"Origin": origin, "X-API-Key": "probe-key"},
            )
            assert allowed.status_code == 200
            assert allowed.headers.get("access-control-allow-origin") == origin

    asyncio.run(run())


def test_api_key_sets_service_tenant_and_enforces_roles(monkeypatch):
    async def run():
        monkeypatch.setenv("RAG_API_KEY", "test-key")
        monkeypatch.setenv("RAG_SERVICE_TENANT_ID", "00000000-0000-0000-0000-00000000000a")
        monkeypatch.setenv("RAG_SERVICE_ROLES", "viewer")
        middleware = authentication_middleware(
            {"security": {"api_key": {"enabled": True, "header_name": "X-API-Key"}}}
        )

        async def endpoint(request):
            assert request.state.principal == current_principal()
            assert current_principal().tenant_id.endswith("000a")
            return httpx.Response(200)

        def request(method, path):
            return Request(
                {
                    "type": "http",
                    "http_version": "1.1",
                    "method": method,
                    "scheme": "http",
                    "path": path,
                    "raw_path": path.encode(),
                    "query_string": b"",
                    "headers": [(b"x-api-key", b"test-key")],
                    "client": ("test", 123),
                    "server": ("test", 80),
                }
            )

        allowed = await middleware(
            request("GET", "/api/v1/documents"),
            endpoint,
        )
        denied = await middleware(
            request("DELETE", "/api/v1/documents/doc-1"),
            endpoint,
        )
        assert allowed.status_code == 200
        assert denied.status_code == 403

    asyncio.run(run())


def test_api_key_service_account_defaults_to_viewer(monkeypatch):
    async def run():
        monkeypatch.setenv("RAG_API_KEY", "test-key")
        monkeypatch.delenv("RAG_SERVICE_ROLES", raising=False)
        middleware = authentication_middleware(
            {"security": {"api_key": {"enabled": True, "header_name": "X-API-Key"}}}
        )
        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/v1/documents/ingest",
                "raw_path": b"/api/v1/documents/ingest",
                "query_string": b"",
                "headers": [(b"x-api-key", b"test-key")],
                "client": ("test", 123),
                "server": ("test", 80),
            }
        )

        response = await middleware(
            request,
            lambda request: asyncio.sleep(0, result=httpx.Response(200)),
        )
        assert response.status_code == 403

    asyncio.run(run())


def test_api_authorization_denies_unknown_write_routes(monkeypatch):
    async def run():
        monkeypatch.setenv("RAG_API_KEY", "test-key")
        monkeypatch.setenv("RAG_SERVICE_ROLES", "admin")
        middleware = authentication_middleware(
            {"security": {"api_key": {"enabled": True, "header_name": "X-API-Key"}}}
        )

        def request(path):
            return Request(
                {
                    "type": "http",
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": path,
                    "raw_path": path.encode(),
                    "query_string": b"",
                    "headers": [(b"x-api-key", b"test-key")],
                    "client": ("test", 123),
                    "server": ("test", 80),
                }
            )

        async def endpoint(_request):
            return httpx.Response(200)

        assert (await middleware(request("/api/v1/chat"), endpoint)).status_code == 200
        assert (await middleware(request("/api/v1/future-admin"), endpoint)).status_code == 403

    asyncio.run(run())


def test_api_authorization_denies_unknown_read_routes(monkeypatch):
    async def run():
        monkeypatch.setenv("RAG_API_KEY", "test-key")
        monkeypatch.setenv("RAG_SERVICE_ROLES", "admin")
        middleware = authentication_middleware(
            {"security": {"api_key": {"enabled": True, "header_name": "X-API-Key"}}}
        )

        def request(path):
            return Request(
                {
                    "type": "http",
                    "http_version": "1.1",
                    "method": "GET",
                    "scheme": "http",
                    "path": path,
                    "raw_path": path.encode(),
                    "query_string": b"",
                    "headers": [(b"x-api-key", b"test-key")],
                    "client": ("test", 123),
                    "server": ("test", 80),
                }
            )

        async def endpoint(_request):
            return httpx.Response(200)

        assert (await middleware(request("/api/v1/documents"), endpoint)).status_code == 200
        assert (await middleware(request("/api/v1/future-admin"), endpoint)).status_code == 403

    asyncio.run(run())


def test_disabled_authentication_uses_local_admin_identity(monkeypatch):
    async def run():
        monkeypatch.delenv("OIDC_ENABLED", raising=False)
        middleware = authentication_middleware(
            {
                "security": {
                    "api_key": {"enabled": False, "header_name": "X-API-Key"},
                    "oidc": {"enabled": False},
                }
            }
        )
        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/v1/documents/ingest",
                "raw_path": b"/api/v1/documents/ingest",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 123),
                "server": ("127.0.0.1", 8000),
            }
        )

        async def endpoint(endpoint_request):
            principal = endpoint_request.state.principal
            assert principal.auth_type == "local"
            assert principal.roles == frozenset({"viewer", "editor", "admin"})
            return httpx.Response(200)

        response = await middleware(request, endpoint)
        assert response.status_code == 200

    asyncio.run(run())


def test_authentication_context_survives_streaming_response(monkeypatch):
    async def run():
        monkeypatch.setenv("RAG_API_KEY", "test-key")
        monkeypatch.setenv("RAG_SERVICE_TENANT_ID", "00000000-0000-0000-0000-00000000000a")
        middleware = authentication_middleware(
            {"security": {"api_key": {"enabled": True, "header_name": "X-API-Key"}}}
        )
        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/v1/chat/stream",
                "raw_path": b"/api/v1/chat/stream",
                "query_string": b"",
                "headers": [(b"x-api-key", b"test-key")],
                "client": ("test", 123),
                "server": ("test", 80),
            }
        )

        async def endpoint(_request):
            async def body():
                assert current_principal().tenant_id.endswith("000a")
                yield b"data: ok\n\n"

            return StreamingResponse(body(), media_type="text/event-stream")

        response = await middleware(request, endpoint)
        assert current_principal().tenant_id.endswith("000a")
        assert b"".join([chunk async for chunk in response.body_iterator]) == b"data: ok\n\n"
        assert current_principal().auth_type == "local"

    asyncio.run(run())


def test_oidc_token_without_tenant_claim_is_rejected(monkeypatch):
    validator = OIDCValidator(
        {
            "security": {
                "oidc": {
                    "enabled": True,
                    "issuer": "https://issuer.example/realms/rag",
                    "audience": "rag-api",
                    "jwks_url": "https://issuer.example/certs",
                }
            }
        }
    )
    validator._jwks_client = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key="public-key")
    )
    fake_jwt = SimpleNamespace(
        decode=lambda *args, **kwargs: {
            "sub": "user-1",
            "iss": validator.issuer,
            "exp": 9999999999,
            "iat": 1,
            "realm_access": {"roles": ["viewer"]},
        }
    )
    monkeypatch.setitem(sys.modules, "jwt", fake_jwt)

    with pytest.raises(ValueError, match="tenant_id"):
        validator._decode("token")
