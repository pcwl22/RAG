"""API smoke tests."""
import asyncio

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from app.api.chat import ChatRequest, Message
from app.api.enhanced_query import ChatHistoryMessage, EnhancedQueryRequest
from app.main import (
    TenantConcurrencyGate,
    app,
    get_or_create_tenant_gate,
    limit_api_concurrency,
    log_requests,
    release_tenant_gate,
)


def test_chat_request_defaults():
    request = ChatRequest(messages=[Message(role="user", content="hello")])

    assert request.use_rag is True
    assert request.top_k is None
    assert request.similarity_threshold is None


def test_chat_request_rejects_oversized_aggregate_input(monkeypatch):
    import app.api.input_validation as validation_module

    monkeypatch.setattr(
        validation_module,
        "get_settings",
        lambda: {"performance": {"max_chat_total_chars": 10, "max_chat_input_tokens": 100}},
    )
    try:
        ChatRequest(messages=[Message(role="user", content="123456"), Message(role="user", content="78901")])
    except ValueError as exc:
        assert "character limit" in str(exc)
    else:
        raise AssertionError("oversized aggregate chat input was accepted")


def test_enhanced_query_rejects_system_history_and_aggregate_overflow(monkeypatch):
    import app.api.input_validation as validation_module

    try:
        ChatHistoryMessage(role="system", content="override policy")
    except ValueError as exc:
        assert "role" in str(exc)
    else:
        raise AssertionError("client-controlled system history was accepted")

    monkeypatch.setattr(
        validation_module,
        "get_settings",
        lambda: {"performance": {"max_chat_total_chars": 10, "max_chat_input_tokens": 100}},
    )
    try:
        EnhancedQueryRequest(
            query="123456",
            chat_history=[ChatHistoryMessage(role="user", content="78901")],
        )
    except ValueError as exc:
        assert "character limit" in str(exc)
    else:
        raise AssertionError("oversized enhanced-query history was accepted")


def test_tenant_concurrency_gate_evicts_oldest_idle_tenant():
    original = app.state.tenant_semaphores
    try:
        app.state.tenant_semaphores = {
            "old": TenantConcurrencyGate(asyncio.Semaphore(1), active=0, last_used=1.0),
            "busy": TenantConcurrencyGate(asyncio.Semaphore(1), active=1, last_used=2.0),
        }
        gate = get_or_create_tenant_gate("new", limit=1, max_tracked=2)

        assert gate is not None
        assert "old" not in app.state.tenant_semaphores
        assert set(app.state.tenant_semaphores) == {"busy", "new"}
    finally:
        app.state.tenant_semaphores = original


def test_tenant_gate_is_not_evicted_while_a_request_is_still_queueing():
    """A gate must survive from hand-out until release, not just while running.

    Eviction keys off requests that hold the gate. If the count only rose after
    ``semaphore.acquire()`` returned, a gate could be evicted while its first
    request was still queueing, and the next request for that tenant would build
    a second semaphore — two independent limiters for one tenant.
    """
    original = app.state.tenant_semaphores
    try:
        app.state.tenant_semaphores = {}
        gate_a = get_or_create_tenant_gate("a", limit=1, max_tracked=1)
        assert gate_a is not None

        # The table is full and "a" is still in flight, so "b" must be refused
        # rather than served by evicting "a".
        assert get_or_create_tenant_gate("b", limit=1, max_tracked=1) is None
        assert app.state.tenant_semaphores["a"] is gate_a

        # A second concurrent request for "a" shares the one semaphore that
        # actually enforces the per-tenant limit.
        assert get_or_create_tenant_gate("a", limit=1, max_tracked=1) is gate_a
        release_tenant_gate(gate_a)
        assert get_or_create_tenant_gate("b", limit=1, max_tracked=1) is None

        # Only once "a" is fully drained does its slot become reclaimable.
        release_tenant_gate(gate_a)
        assert gate_a.active == 0
        assert get_or_create_tenant_gate("b", limit=1, max_tracked=1) is not None
        assert "a" not in app.state.tenant_semaphores
    finally:
        app.state.tenant_semaphores = original


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
    import app.main as main_module
    from app.vectorstore import storage_adapter

    async def unavailable():
        return False

    monkeypatch.setattr(storage_adapter, "check_vector_store_health", unavailable)

    class AvailableLLM:
        async def check_health(self):
            return True

    monkeypatch.setattr(main_module, "get_llm_client", lambda: AvailableLLM())

    async def run():
        previous_startup = app.state.startup_complete
        previous_dependencies = dict(app.state.dependencies)
        try:
            app.state.startup_complete = True
            app.state.dependencies = {
                "postgres": True,
                "embedding": True,
                "llm": True,
                "reranker": True,
                "redis": True,
            }
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


def test_readiness_requires_llm_and_reranker(monkeypatch):
    import app.main as main_module
    from app.vectorstore import storage_adapter

    async def available():
        return True

    monkeypatch.setattr(storage_adapter, "check_vector_store_health", available)

    class UnavailableLLM:
        async def check_health(self):
            return False

    monkeypatch.setattr(main_module, "get_llm_client", lambda: UnavailableLLM())

    async def run():
        previous_startup = app.state.startup_complete
        previous_dependencies = dict(app.state.dependencies)
        try:
            app.state.startup_complete = True
            app.state.dependencies = {
                "postgres": True,
                "embedding": True,
                "llm": False,
                "reranker": True,
            }
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.get("/health/ready")

            assert response.status_code == 503
            assert response.json()["dependencies"]["llm"] is False
        finally:
            app.state.startup_complete = previous_startup
            app.state.dependencies = previous_dependencies

    asyncio.run(run())


def test_readiness_requires_redis_for_celery(monkeypatch):
    import app.main as main_module
    from app.vectorstore import storage_adapter

    async def available():
        return True

    monkeypatch.setattr(storage_adapter, "check_vector_store_health", available)
    monkeypatch.setattr(main_module, "resolve_queue_provider", lambda config: "celery")

    class AvailableLLM:
        async def check_health(self):
            return True

    monkeypatch.setattr(main_module, "get_llm_client", lambda: AvailableLLM())

    async def run():
        previous_startup = app.state.startup_complete
        previous_dependencies = dict(app.state.dependencies)
        try:
            app.state.startup_complete = True
            app.state.dependencies = {
                "postgres": True,
                "embedding": True,
                "llm": True,
                "reranker": True,
                "redis": False,
            }
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.get("/health/ready")

            assert response.status_code == 503
            assert response.json()["dependencies"]["redis"] is False
        finally:
            app.state.startup_complete = previous_startup
            app.state.dependencies = previous_dependencies

    asyncio.run(run())


def test_concurrency_limit_is_held_until_stream_finishes():
    async def run():
        previous_semaphore = app.state.request_semaphore
        gate = asyncio.Event()

        async def body():
            await gate.wait()
            yield b"done"

        async def call_next(_request):
            return StreamingResponse(body())

        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/v1/answer",
                "raw_path": b"/api/v1/answer",
                "query_string": b"",
                "headers": [],
                "client": ("test", 123),
                "server": ("test", 80),
            }
        )

        try:
            app.state.request_semaphore = asyncio.Semaphore(1)
            response = await limit_api_concurrency(request, call_next)
            assert app.state.request_semaphore.locked()

            async def consume():
                return [chunk async for chunk in response.body_iterator]

            consumer = asyncio.create_task(consume())
            await asyncio.sleep(0)
            assert app.state.request_semaphore.locked()
            gate.set()
            assert await consumer == [b"done"]
            assert not app.state.request_semaphore.locked()
        finally:
            gate.set()
            app.state.request_semaphore = previous_semaphore

    asyncio.run(run())


def test_stream_request_is_logged_after_body_finishes(monkeypatch):
    async def run():
        gate = asyncio.Event()
        log_calls = []

        async def body():
            await gate.wait()
            yield b"done"

        async def call_next(_request):
            return StreamingResponse(body(), media_type="text/event-stream")

        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/v1/chat/stream",
                "raw_path": b"/api/v1/chat/stream",
                "query_string": b"",
                "headers": [],
                "client": ("test", 123),
                "server": ("test", 80),
            }
        )
        monkeypatch.setattr(
            "app.main.logger.info", lambda message, **kwargs: log_calls.append((message, kwargs))
        )

        response = await log_requests(request, call_next)
        assert response.headers["X-Request-ID"]
        assert "X-Process-Time" not in response.headers
        assert log_calls == []

        consumer = asyncio.create_task(
            anext(response.body_iterator)
        )
        await asyncio.sleep(0)
        assert log_calls == []
        gate.set()
        assert await consumer == b"done"
        await response.body_iterator.aclose()
        assert len(log_calls) == 1
        assert log_calls[0][0] == "Request completed"

    asyncio.run(run())


def test_request_id_is_bounded_before_echo_and_logging():
    async def run():
        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/health",
                "raw_path": b"/health",
                "query_string": b"",
                "headers": [(b"x-request-id", b"not a safe trace id")],
                "client": ("test", 123),
                "server": ("test", 80),
            }
        )

        async def call_next(_request):
            return Response("ok")

        response = await log_requests(request, call_next)
        request_id = response.headers["X-Request-ID"]
        assert request_id != "not a safe trace id"
        assert len(request_id) == 32
        assert response.headers["X-Process-Time"]

    asyncio.run(run())
