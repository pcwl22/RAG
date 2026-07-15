import asyncio

import httpx

from app.main import app
from app.security import authentication_middleware, validate_security_config


def test_production_security_rejects_wildcard_credentials():
    try:
        validate_security_config(
            {
                "app": {"debug": False},
                "security": {"cors": {"allow_origins": ["*"], "allow_credentials": True}},
            }
        )
    except RuntimeError as exc:
        assert "wildcard" in str(exc)
    else:
        raise AssertionError("unsafe CORS configuration was accepted")


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


def test_health_endpoints_are_public():
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            assert (await client.get("/health/live")).status_code == 200

    asyncio.run(run())
