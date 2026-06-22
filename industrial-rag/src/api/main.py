"""
FastAPI主应用
"""
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

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("Starting Industrial RAG System...")
    config = get_settings()

    # 初始化向量数据库连接 (使用统一的存储适配器)
    from src.storage.storage_adapter import init_vector_store
    await init_vector_store()
    logger.info("Vector store initialized")

    # 初始化Redis连接（可选，失败不阻塞启动）
    try:
        from src.core.cache import init_redis

        await init_redis()
    except Exception as e:
        logger.warning(f"Redis initialization failed: {e}. Caching will be disabled.")

    # 加载嵌入模型
    from src.models.embedding import load_embedding_model

    load_embedding_model()

    logger.info("System ready")

    yield

    # 关闭时清理
    logger.info("Shutting down...")
    from src.storage.storage_adapter import close_vector_store
    await close_vector_store()

    # 关闭 Redis（如果已连接）
    try:
        from src.core.cache import close_redis

        await close_redis()
    except Exception as e:
        logger.warning(f"Redis shutdown failed: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="Industrial RAG System",
    description="高性能中文多模态RAG系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS配置
config = get_settings()
if config.get("security", {}).get("cors", {}).get("enabled", True):
    cors_config = config["security"]["cors"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("allow_origins", ["*"]),
        allow_credentials=True,
        allow_methods=cors_config.get("allow_methods", ["*"]),
        allow_headers=cors_config.get("allow_headers", ["*"]),
    )


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志"""
    start_time = time.time()

    # 添加请求ID
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


# 异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if config.get("app", {}).get("debug") else "An error occurred",
        },
    )


# Prometheus监控
if Instrumentator is not None and config.get("monitoring", {}).get("prometheus", {}).get("enabled", True):
    Instrumentator().instrument(app).expose(app)


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": int(time.time()),
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Industrial RAG System API",
        "docs": "/docs",
        "health": "/health",
    }


# 注册路由
from src.api.routes import chat, documents, query
from src.api.routes.query_simple import router_simple
from src.api.routes.enhanced_query import router as enhanced_query_router

app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(router_simple, prefix="/api/v1", tags=["query-simple"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(enhanced_query_router, prefix="/api/v1", tags=["enhanced"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=config["app"]["host"],
        port=config["app"]["port"],
        reload=config["app"]["debug"],
    )
