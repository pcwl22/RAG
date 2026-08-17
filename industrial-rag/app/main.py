"""FastAPI application entrypoint for the rag-system layout."""
import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from app.api import chat, search, upload
from app.embedding.embedder import load_embedding_model
from app.llm.model import get_llm_client
from app.retrieval.reranker import load_reranker
from app.security import authentication_middleware, validate_security_config
from app.utils.config import get_settings, resolve_queue_provider, validate_runtime_config
from app.utils.inference import close_inference_executor
from app.utils.logger import get_logger
from app.vectorstore.storage_adapter import close_vector_store, init_vector_store

# Optional monitoring dependency: declared as Any so both branches can bind it.
Instrumentator: Any
try:
    from prometheus_fastapi_instrumentator import Instrumentator as _Instrumentator
except ImportError:  # pragma: no cover - exercised in installs without the extra
    Instrumentator = None
else:
    Instrumentator = _Instrumentator

logger = get_logger(__name__)
BASE_REQUIRED_DEPENDENCIES = ("postgres", "embedding", "llm", "reranker")


@dataclass
class TenantConcurrencyGate:
    semaphore: asyncio.Semaphore
    # Requests holding a reservation on this gate. Counted from the moment the
    # gate is handed out, not from semaphore acquisition, so that a request still
    # queueing for capacity keeps the gate alive. Zero means safe to evict.
    active: int
    last_used: float


def get_or_create_tenant_gate(
    tenant_id: str,
    *,
    limit: int,
    max_tracked: int,
) -> TenantConcurrencyGate | None:
    """Reserve a tenant gate, evicting the least-recently-used idle gate if full.

    The returned gate carries a reservation, so it cannot be evicted while the
    caller is still awaiting capacity. Without it, a concurrent request for a
    different tenant could evict this gate during that await; the next request
    for this tenant would then build a second semaphore, and the two would
    enforce the per-tenant limit independently — that is, not at all.

    Every successful call must be paired with :func:`release_tenant_gate`.
    """
    gates: dict[str, TenantConcurrencyGate] = app.state.tenant_semaphores
    gate = gates.get(tenant_id)
    if gate is None:
        if len(gates) >= max_tracked:
            idle = [
                (tracked_id, tracked_gate)
                for tracked_id, tracked_gate in gates.items()
                if tracked_gate.active == 0
            ]
            if not idle:
                return None
            evicted_id, _ = min(idle, key=lambda item: item[1].last_used)
            gates.pop(evicted_id, None)
        gate = TenantConcurrencyGate(asyncio.Semaphore(limit), 0, time.monotonic())
        gates[tenant_id] = gate

    gate.active += 1
    gate.last_used = time.monotonic()
    return gate


def release_tenant_gate(gate: TenantConcurrencyGate) -> None:
    """Drop a reservation taken by :func:`get_or_create_tenant_gate`."""
    gate.active = max(0, gate.active - 1)
    gate.last_used = time.monotonic()


def required_dependencies() -> tuple[str, ...]:
    """Return dependencies required for the configured runtime mode."""
    dependencies = list(BASE_REQUIRED_DEPENDENCIES)
    if resolve_queue_provider(config) == "celery":
        dependencies.append("redis")
    return tuple(dependencies)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown."""
    logger.info("Starting RAG System...")

    try:
        await init_vector_store()
        app.state.dependencies["postgres"] = True
        logger.info("Vector store initialized")
    except Exception as exc:
        logger.error("Vector store initialization failed: %s", exc, exc_info=True)

    try:
        from app.utils.cache import init_redis

        app.state.dependencies["redis"] = await init_redis()
    except Exception as exc:
        logger.warning(f"Redis initialization failed: {exc}. Caching will be disabled.")

    try:
        load_embedding_model()
        app.state.dependencies["embedding"] = True
    except Exception as exc:
        logger.error("Embedding model initialization failed: %s", exc, exc_info=True)

    try:
        llm_client = get_llm_client()
        timeout = float(config.get("llm", {}).get("text", {}).get("healthcheck_timeout_seconds", 5))
        app.state.dependencies["llm"] = await asyncio.wait_for(
            llm_client.check_health(force=True), timeout=timeout
        )
        if not app.state.dependencies["llm"]:
            raise RuntimeError("configured LLM upstream is unavailable")
    except Exception as exc:
        logger.error("LLM client initialization failed: %s", exc, exc_info=True)

    reranker_enabled = bool(config.get("reranker", {}).get("enabled", False))
    if not reranker_enabled:
        app.state.dependencies["reranker"] = True
    else:
        try:
            if load_reranker() is None:
                raise RuntimeError("configured reranker is unavailable")
            app.state.dependencies["reranker"] = True
        except Exception as exc:
            logger.error("Reranker initialization failed: %s", exc, exc_info=True)
    logger.info("System ready (dependencies=%s)", app.state.dependencies)
    app.state.startup_complete = True

    yield

    logger.info("Shutting down...")
    if app.state.dependencies.get("postgres"):
        await close_vector_store()

    try:
        from app.utils.cache import close_redis

        await close_redis()
    except Exception as exc:
        logger.warning(f"Redis shutdown failed: {exc}")
    close_inference_executor()


config = get_settings()
validate_runtime_config(config)
validate_security_config(config)

app = FastAPI(
    title="RAG System",
    description="本地知识库 RAG 问答系统",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.dependencies = {
    "postgres": False,
    "embedding": False,
    "llm": False,
    "reranker": False,
    "redis": False,
}
app.state.startup_complete = False
app.state.request_semaphore = asyncio.Semaphore(
    int(config.get("performance", {}).get("max_concurrent_requests", 100))
)
app.state.tenant_semaphores = {}


@app.middleware("http")
async def limit_api_concurrency(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """Bound API work for the complete response, including SSE body generation."""
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    timeout = float(
        config.get("performance", {}).get("request_queue_timeout_seconds", 5)
    )
    performance = config.get("performance", {})
    principal = getattr(request.state, "principal", None)
    tenant_id = getattr(principal, "tenant_id", "local")
    from app.utils.cache import consume_tenant_rate_limit

    try:
        allowed, remaining = await consume_tenant_rate_limit(
            tenant_id,
            limit=int(performance.get("tenant_requests_per_minute", 120)),
            window_seconds=60,
            fail_open=bool(performance.get("tenant_rate_limit_fail_open", False)),
        )
    except Exception:
        logger.warning("Tenant rate limiter is unavailable", exc_info=True)
        return JSONResponse(status_code=503, content={"detail": "Request limiter unavailable"})
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Tenant request rate limit exceeded"},
            headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
        )

    tenant_limit = max(1, int(performance.get("tenant_max_concurrent_requests", 10)))
    tenant_gate = get_or_create_tenant_gate(
        tenant_id,
        limit=tenant_limit,
        max_tracked=max(1, int(performance.get("max_tracked_tenants", 10000))),
    )
    if tenant_gate is None:
        return JSONResponse(status_code=503, content={"detail": "Tenant capacity unavailable"})
    try:
        await asyncio.wait_for(app.state.request_semaphore.acquire(), timeout=timeout)
    except TimeoutError:
        release_tenant_gate(tenant_gate)
        return JSONResponse(status_code=503, content={"detail": "Server is busy"})
    try:
        await asyncio.wait_for(tenant_gate.semaphore.acquire(), timeout=timeout)
    except TimeoutError:
        app.state.request_semaphore.release()
        release_tenant_gate(tenant_gate)
        return JSONResponse(
            status_code=429,
            content={"detail": "Tenant concurrency limit exceeded"},
            headers={"Retry-After": "1", "X-RateLimit-Remaining": str(remaining)},
        )

    def release_capacity() -> None:
        release_tenant_gate(tenant_gate)
        tenant_gate.semaphore.release()
        app.state.request_semaphore.release()

    try:
        response = await call_next(request)
    except BaseException:
        release_capacity()
        raise

    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:
        release_capacity()
        return response

    async def limited_body_iterator() -> AsyncGenerator[Any, None]:
        try:
            async for chunk in body_iterator:
                yield chunk
        finally:
            release_capacity()

    # Duck-typed on purpose: call_next returns Starlette's private
    # _StreamingResponse, which subclasses Response rather than StreamingResponse,
    # so an isinstance narrowing here would skip every streamed body.
    streaming: Any = response
    streaming.body_iterator = limited_body_iterator()
    return response

# Starlette wraps middleware so that the most recently registered runs
# outermost. Authentication must be registered BEFORE CORS so that CORS ends up
# on the outside and its headers are still attached to 401/403 responses;
# otherwise a browser sees an opaque CORS failure instead of the real status.
app.middleware("http")(authentication_middleware(config))

if config.get("security", {}).get("cors", {}).get("enabled", True):
    cors_config = config["security"]["cors"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("allow_origins", ["*"]),
        allow_credentials=bool(cors_config.get("allow_credentials", False)),
        allow_methods=cors_config.get("allow_methods", ["*"]),
        allow_headers=cors_config.get("allow_headers", ["*"]),
    )


@app.middleware("http")
async def log_requests(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Log each HTTP request after its response body has completed."""
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", f"{int(time.time() * 1000)}")

    response = await call_next(request)
    principal = getattr(request.state, "principal", None)
    response.headers["X-Request-ID"] = request_id

    def log_completed() -> float:
        duration = time.time() - start_time
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration": f"{duration:.3f}s",
                "subject": getattr(principal, "subject", None),
                "tenant_id": getattr(principal, "tenant_id", None),
                "auth_type": getattr(principal, "auth_type", None),
            },
        )
        return duration

    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:
        response.headers["X-Process-Time"] = f"{log_completed():.3f}"
        return response

    async def logged_body_iterator() -> AsyncGenerator[Any, None]:
        try:
            async for chunk in body_iterator:
                yield chunk
        finally:
            log_completed()

    streaming: Any = response
    streaming.body_iterator = logged_body_iterator()
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if config.get("app", {}).get("debug") else "An error occurred",
        },
    )


prometheus_config = config.get("monitoring", {}).get("prometheus", {})
if Instrumentator is not None and prometheus_config.get("enabled", True):
    Instrumentator().instrument(app).expose(
        app,
        endpoint=str(prometheus_config.get("path", "/internal/metrics")),
        include_in_schema=False,
    )


async def _probe_dependencies() -> dict[str, bool]:
    """Refresh critical dependency state with a bounded live probe."""
    dependencies = dict(getattr(app.state, "dependencies", {}))
    if not getattr(app.state, "startup_complete", False):
        return dependencies

    try:
        from app.vectorstore.storage_adapter import check_vector_store_health

        dependencies["postgres"] = await asyncio.wait_for(
            check_vector_store_health(), timeout=2.0
        )
    except Exception:
        logger.warning("PostgreSQL readiness probe failed", exc_info=True)
        dependencies["postgres"] = False

    try:
        llm_client = get_llm_client()
        timeout = float(config.get("llm", {}).get("text", {}).get("healthcheck_timeout_seconds", 5))
        dependencies["llm"] = await asyncio.wait_for(
            llm_client.check_health(), timeout=timeout
        )
    except Exception:
        logger.warning("LLM readiness probe failed", exc_info=True)
        dependencies["llm"] = False

    if "redis" in required_dependencies() or dependencies.get("redis"):
        try:
            from app.utils.cache import check_redis_health

            dependencies["redis"] = await asyncio.wait_for(
                check_redis_health(), timeout=1.0
            )
        except Exception:
            logger.warning("Redis readiness probe failed", exc_info=True)
            dependencies["redis"] = False

    app.state.dependencies.update(dependencies)
    return dependencies


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    dependencies = await _probe_dependencies()
    ready = getattr(app.state, "startup_complete", False) and all(
        dependencies.get(name, False) for name in required_dependencies()
    )
    return {
        "status": "healthy" if ready else "degraded",
        "version": "1.0.0",
        "timestamp": int(time.time()),
        "dependencies": dependencies,
    }


@app.get("/health/live")
async def liveness_check() -> dict[str, Any]:
    return {"status": "alive", "timestamp": int(time.time())}


@app.get("/health/ready")
async def readiness_check() -> JSONResponse:
    dependencies = await _probe_dependencies()
    ready = getattr(app.state, "startup_complete", False) and all(
        dependencies.get(name, False) for name in required_dependencies()
    )
    payload = {"status": "ready" if ready else "not_ready", "dependencies": dependencies}
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "RAG System API",
        "docs": "/docs",
        "health": "/health",
    }


app.include_router(upload.router, prefix="/api/v1", tags=["documents"])
app.include_router(search.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config["app"]["host"],
        port=config["app"]["port"],
        reload=config["app"]["debug"],
    )
