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
