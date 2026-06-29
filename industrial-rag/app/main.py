"""FastAPI application entrypoint for the rag-system layout."""
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:
    Instrumentator = None

from app.api import chat, search, upload
from app.embedding.embedder import load_embedding_model
from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.vectorstore.qdrant_client import close_vector_store, init_vector_store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown."""
    logger.info("Starting RAG System...")

    await init_vector_store()
    logger.info("Vector store initialized")

    try:
        from app.utils.cache import init_redis

        await init_redis()
    except Exception as exc:
        logger.warning(f"Redis initialization failed: {exc}. Caching will be disabled.")

    load_embedding_model()
    logger.info("System ready")

    yield

    logger.info("Shutting down...")
    await close_vector_store()

    try:
        from app.utils.cache import close_redis

        await close_redis()
    except Exception as exc:
        logger.warning(f"Redis shutdown failed: {exc}")


config = get_settings()

app = FastAPI(
    title="RAG System",
    description="本地知识库 RAG 问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

if config.get("security", {}).get("cors", {}).get("enabled", True):
    cors_config = config["security"]["cors"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("allow_origins", ["*"]),
        allow_credentials=True,
        allow_methods=cors_config.get("allow_methods", ["*"]),
        allow_headers=cors_config.get("allow_headers", ["*"]),
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log each HTTP request."""
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", f"{int(time.time() * 1000)}")

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration": f"{duration:.3f}s",
        },
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration:.3f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if config.get("app", {}).get("debug") else "An error occurred",
        },
    )


if Instrumentator is not None and config.get("monitoring", {}).get("prometheus", {}).get("enabled", True):
    Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": int(time.time()),
    }


@app.get("/")
async def root():
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
