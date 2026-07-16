"""API smoke tests."""
import asyncio

import httpx

from app.api.chat import ChatRequest, Message
from app.main import app


def test_chat_request_defaults():
    request = ChatRequest(messages=[Message(role="user", content="hello")])

    assert request.use_rag is True
    assert request.top_k is None
    assert request.similarity_threshold is None


def test_health_and_root_endpoints():
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            health = await client.get("/health")
            root = await client.get("/")

        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert root.status_code == 200
        assert root.json()["docs"] == "/docs"

    asyncio.run(run())


def test_readiness_uses_live_postgres_probe(monkeypatch):
    from app.vectorstore import storage_adapter

    async def unavailable():
        return False

    monkeypatch.setattr(storage_adapter, "check_vector_store_health", unavailable)

    async def run():
        previous_startup = app.state.startup_complete
        previous_dependencies = dict(app.state.dependencies)
        try:
            app.state.startup_complete = True
            app.state.dependencies = {"postgres": True, "embedding": True, "redis": True}
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.get("/health/ready")

            assert response.status_code == 503
            assert response.json()["dependencies"]["postgres"] is False
        finally:
            app.state.startup_complete = previous_startup
            app.state.dependencies = previous_dependencies

    asyncio.run(run())
