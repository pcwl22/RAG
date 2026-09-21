"""Regression tests for API availability, fairness, and public contracts."""

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

import app.main as main_module
from app.auth import OIDCAuthenticationError, OIDCProviderUnavailable
from app.main import app, limit_api_concurrency
from app.security import authentication_middleware


def _request(path: str, tenant_id: str = "tenant-a", *, bearer: bool = False) -> Request:
    headers = [(b"authorization", b"Bearer token")] if bearer else []
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET" if bearer else "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers,
            "client": ("test", 123),
            "server": ("test", 80),
        }
    )
    request.state.principal = SimpleNamespace(tenant_id=tenant_id)
    return request


def test_noisy_tenant_waiters_do_not_reserve_global_capacity(monkeypatch):
    from app.utils import cache

    async def allow_rate_limit(*_args, **_kwargs):
        return True, 99

    async def run():
        previous_semaphore = app.state.request_semaphore
        previous_gates = app.state.tenant_semaphores
        previous_performance = main_module.config.get("performance")
        first_body_gate = asyncio.Event()

        async def first_body():
            await first_body_gate.wait()
            yield b"done"

        try:
            monkeypatch.setattr(cache, "consume_tenant_rate_limit", allow_rate_limit)
            main_module.config["performance"] = {
                **(previous_performance or {}),
                "request_queue_timeout_seconds": 1,
                "tenant_max_concurrent_requests": 1,
                "max_tracked_tenants": 100,
            }
            app.state.request_semaphore = asyncio.Semaphore(2)
            app.state.tenant_semaphores = {}

            first = await limit_api_concurrency(
                _request("/api/v1/answer", "tenant-a"),
                lambda _request: asyncio.sleep(
                    0, result=StreamingResponse(first_body())
                ),
            )
            first_consumer = asyncio.create_task(anext(first.body_iterator))
            await asyncio.sleep(0)

            second_a = asyncio.create_task(
                limit_api_concurrency(
                    _request("/api/v1/answer", "tenant-a"),
                    lambda _request: asyncio.sleep(0, result=Response(status_code=200)),
                )
            )
            await asyncio.sleep(0.02)

            # Tenant A's waiter must still be outside the global semaphore.
            assert app.state.request_semaphore._value == 1
            tenant_b = await asyncio.wait_for(
                limit_api_concurrency(
                    _request("/api/v1/answer", "tenant-b"),
                    lambda _request: asyncio.sleep(0, result=Response(status_code=200)),
                ),
                timeout=0.2,
            )
            assert tenant_b.status_code == 200

            first_body_gate.set()
            assert await first_consumer == b"done"
            await first.body_iterator.aclose()
            assert (await asyncio.wait_for(second_a, timeout=0.5)).status_code == 200
        finally:
            first_body_gate.set()
            app.state.request_semaphore = previous_semaphore
            app.state.tenant_semaphores = previous_gates
            if previous_performance is None:
                main_module.config.pop("performance", None)
            else:
                main_module.config["performance"] = previous_performance

    asyncio.run(run())


@pytest.mark.parametrize(
    ("failure", "expected_status", "authenticate"),
    [
        (OIDCAuthenticationError("bad token"), 401, 'Bearer error="invalid_token"'),
        (OIDCProviderUnavailable("jwks down"), 503, None),
    ],
)
def test_oidc_token_failure_is_distinct_from_provider_outage(
    monkeypatch, failure, expected_status, authenticate
):
    monkeypatch.setenv("OIDC_ENABLED", "true")

    async def fail_validation(_self, _token):
        raise failure

    monkeypatch.setattr("app.security.OIDCValidator.validate", fail_validation)
    middleware = authentication_middleware(
        {
            "security": {
                "oidc": {
                    "enabled": True,
                    "issuer": "https://issuer.example",
                    "audience": "rag-api",
                    "jwks_url": "https://issuer.example/certs",
                },
                "api_key": {"enabled": False},
            },
            "monitoring": {"prometheus": {"enabled": False}},
        }
    )

    async def run():
        return await middleware(
            _request("/api/v1/documents", bearer=True),
            lambda _request: asyncio.sleep(0, result=Response(status_code=200)),
        )

    response = asyncio.run(run())
    assert response.status_code == expected_status
    assert response.headers.get("www-authenticate") == authenticate
    if expected_status == 503:
        assert response.headers["retry-after"] == "5"


def test_openapi_declares_authentication_and_sse_media_types():
    app.openapi_schema = None
    schema = app.openapi()

    schemes = schema["components"]["securitySchemes"]
    assert schemes["ApiKeyAuth"]["type"] == "apiKey"
    assert schemes["BearerAuth"]["scheme"] == "bearer"
    for path in ("/api/v1/answer", "/api/v1/query/enhanced", "/api/v1/chat/stream"):
        assert "text/event-stream" in schema["paths"][path]["post"]["responses"]["200"]["content"]


def test_failed_local_models_can_recover_without_process_restart(monkeypatch):
    previous_dependencies = dict(app.state.dependencies)
    previous_config = main_module.config
    calls = []
    try:
        main_module.config = {**previous_config, "reranker": {"enabled": True}}
        app.state.dependencies["embedding"] = False
        app.state.dependencies["reranker"] = False
        monkeypatch.setattr(main_module, "load_embedding_model", lambda: calls.append("embedding") or object())
        monkeypatch.setattr(main_module, "load_reranker", lambda: calls.append("reranker") or object())

        asyncio.run(main_module._recover_local_models(app))

        assert calls == ["embedding", "reranker"]
        assert app.state.dependencies["embedding"] is True
        assert app.state.dependencies["reranker"] is True
    finally:
        app.state.dependencies = previous_dependencies
        main_module.config = previous_config


def test_required_postgres_failure_aborts_startup(monkeypatch):
    monkeypatch.setattr(app.state, "dependencies", dict(app.state.dependencies))

    async def fail_postgres():
        raise ConnectionError("database unavailable")

    monkeypatch.setattr(main_module, "init_vector_store", fail_postgres)

    with pytest.raises(RuntimeError, match="required PostgreSQL"):
        asyncio.run(main_module._initialize_required_infrastructure(app))

    assert app.state.dependencies["postgres"] is False


def test_optional_redis_failure_remains_compatible_with_memory_mode(monkeypatch):
    import app.utils.cache as cache_module

    monkeypatch.setattr(app.state, "dependencies", dict(app.state.dependencies))

    async def init_postgres():
        return None

    async def unavailable_redis():
        return False

    monkeypatch.setattr(main_module, "init_vector_store", init_postgres)
    monkeypatch.setattr(main_module, "resolve_queue_provider", lambda _config: "memory")
    monkeypatch.setattr(cache_module, "init_redis", unavailable_redis)

    asyncio.run(main_module._initialize_required_infrastructure(app))

    assert app.state.dependencies["postgres"] is True
    assert app.state.dependencies["redis"] is False


def test_celery_mode_fails_closed_when_redis_is_unavailable(monkeypatch):
    import app.utils.cache as cache_module

    monkeypatch.setattr(app.state, "dependencies", dict(app.state.dependencies))

    async def init_postgres():
        return None

    async def unavailable_redis():
        return False

    monkeypatch.setattr(main_module, "init_vector_store", init_postgres)
    monkeypatch.setattr(main_module, "resolve_queue_provider", lambda _config: "celery")
    monkeypatch.setattr(cache_module, "init_redis", unavailable_redis)

    with pytest.raises(RuntimeError, match="required Redis"):
        asyncio.run(main_module._initialize_required_infrastructure(app))


def test_celery_mode_fails_closed_when_object_storage_is_unavailable(monkeypatch):
    import app.storage.object_store as object_store_module
    import app.utils.cache as cache_module

    monkeypatch.setattr(app.state, "dependencies", dict(app.state.dependencies))

    class UnhealthyObjectStore:
        def check_health(self):
            return False

    async def init_postgres():
        return None

    async def available_redis():
        return True

    monkeypatch.setattr(main_module, "init_vector_store", init_postgres)
    monkeypatch.setattr(main_module, "resolve_queue_provider", lambda _config: "celery")
    monkeypatch.setattr(cache_module, "init_redis", available_redis)
    monkeypatch.setattr(
        object_store_module,
        "get_object_store",
        lambda _config: UnhealthyObjectStore(),
    )

    with pytest.raises(RuntimeError, match="required object storage"):
        asyncio.run(main_module._initialize_required_infrastructure(app))

    assert app.state.dependencies["object_storage"] is False


def test_unicode_uploads_keep_display_names_and_use_server_owned_temp_paths(
    monkeypatch, tmp_path
):
    import app.api.upload as upload_api

    processed = []

    def fake_process(
        task_id, file_path, filename, partition, metadata, tenant_id=None, app_loop=None
    ):
        processed.append((task_id, Path(file_path).name, filename))
        Path(file_path).unlink(missing_ok=True)

    monkeypatch.setattr(
        upload_api,
        "config",
        {
            "document_processing": {
                "upload_dir": str(tmp_path),
                "supported_formats": ["txt"],
                "max_file_size": 1024,
            },
            "queue": {"provider": "memory"},
        },
    )
    monkeypatch.setattr(upload_api, "_process_document_memory", fake_process)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post(
                "/api/v1/documents/ingest",
                files={"file": ("劳动合同法.txt", "合法文本内容".encode(), "text/plain")},
            )
            second = await client.post(
                "/api/v1/documents/ingest",
                files={"file": ("民法典.txt", "另一份合法文本".encode(), "text/plain")},
            )
        return first, second

    first, second = asyncio.run(run())
    assert first.status_code == second.status_code == 200
    assert first.json()["filename"] == "劳动合同法.txt"
    assert second.json()["filename"] == "民法典.txt"
    assert {item[2] for item in processed} == {"劳动合同法.txt", "民法典.txt"}
    assert all(item[1].endswith(".txt") and "合同" not in item[1] for item in processed)
    assert processed[0][1] != processed[1][1]


def test_celery_status_reports_backend_outage_as_503(monkeypatch):
    import app.api.upload as upload_api
    import app.utils.cache as cache
    celery_module = importlib.import_module("app.workers.celery_app")

    upload_api._task_registry.clear()
    monkeypatch.setattr(upload_api, "config", {"queue": {"provider": "celery"}})
    monkeypatch.setattr(cache, "_redis_client", None)
    monkeypatch.setattr(
        celery_module,
        "celery_app",
        SimpleNamespace(
            AsyncResult=lambda _task_id: SimpleNamespace(state="PENDING", info=None)
        ),
    )

    with pytest.raises(HTTPException) as captured:
        asyncio.run(upload_api.get_document_status("00000000-0000-0000-0000-000000000099"))
    assert captured.value.status_code == 503
    assert captured.value.headers == {"Retry-After": "2"}


def test_celery_result_is_not_exposed_without_tenant_scoped_receipt(monkeypatch):
    import app.api.upload as upload_api

    celery_module = importlib.import_module("app.workers.celery_app")

    class UntrustedResult:
        @property
        def state(self):
            raise AssertionError("result backend must not be queried before tenant binding")

    async def missing_state(*_args, **_kwargs):
        return None

    monkeypatch.setattr(upload_api, "config", {"queue": {"provider": "celery"}})
    monkeypatch.setattr(upload_api, "_load_task_state", missing_state)
    monkeypatch.setattr(
        celery_module,
        "celery_app",
        SimpleNamespace(AsyncResult=lambda _task_id: UntrustedResult()),
    )

    with pytest.raises(HTTPException) as captured:
        asyncio.run(upload_api.get_document_status("guessed-cross-tenant-task"))
    assert captured.value.status_code == 404


def test_celery_success_exposes_tenant_scoped_document_id_for_cleanup(monkeypatch):
    import app.api.upload as upload_api

    celery_module = importlib.import_module("app.workers.celery_app")

    async def known_state(*_args, **_kwargs):
        return {"status": "processing", "progress": 50, "total_chunks": 0}

    monkeypatch.setattr(upload_api, "config", {"queue": {"provider": "celery"}})
    monkeypatch.setattr(upload_api, "_load_task_state", known_state)
    monkeypatch.setattr(
        celery_module,
        "celery_app",
        SimpleNamespace(
            AsyncResult=lambda _task_id: SimpleNamespace(
                state="SUCCESS",
                info={
                    "status": "completed",
                    "document_id": "tenant-document-1",
                    "total_chunks": 2,
                },
            )
        ),
    )

    status = asyncio.run(upload_api.get_document_status("tenant-task-1"))

    assert status.document_id == "tenant-document-1"
    assert status.status == "completed"
