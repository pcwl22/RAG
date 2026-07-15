"""
缓存模块 - Redis + 语义缓存

缓存键绑定查询、实际检索上下文、模型/提示词配置与知识库版本。
当前实现是确定性的答案缓存，不声称执行向量相似度检索；如需真正的
语义近邻缓存，应单独接入 Redis Vector Search 并校准命中阈值。
"""
import hashlib
import json
import os
from typing import Any

try:
    import redis.asyncio as aioredis
except ImportError:  # Redis is optional; the app can run without semantic cache.
    aioredis = None

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 全局 Redis 客户端
_redis_client: Any | None = None


def _redis_config() -> dict:
    """获取 Redis 配置。"""
    return get_settings().get("redis", {})


async def init_redis() -> bool:
    """初始化 Redis 连接。"""
    global _redis_client

    if _redis_client is not None:
        return True

    if aioredis is None:
        logger.warning("redis package is not installed. Caching will be disabled.")
        return False

    cfg = _redis_config()
    url = cfg.get("url", "redis://localhost:6379/0")
    max_connections = cfg.get("max_connections", 20)

    try:
        _redis_client = aioredis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=max_connections,
        )
        # 测试连接
        await _redis_client.ping()
        logger.info(f"Redis connected: {url}")
        return True
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Caching will be disabled.")
        _redis_client = None
        return False


async def close_redis() -> None:
    """关闭 Redis 连接。"""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis connection closed")


def _get_redis() -> Any | None:
    """获取 Redis 客户端（可能为 None，表示缓存不可用）。"""
    return _redis_client


def _hash_payload(query: str, context: dict[str, Any]) -> str:
    payload = {
        "query": " ".join(query.split()),
        "context": context,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SemanticCache:
    """Versioned deterministic answer cache kept under the legacy class name."""

    def __init__(self):
        cfg = _redis_config().get("semantic_cache", {})
        self.enabled = cfg.get("enabled", True)
        self.ttl = cfg.get("ttl", 3600)  # 1小时
        # Bump for incompatible cache-schema or prompt/model migrations.
        version = os.getenv("RAG_CACHE_VERSION") or str(cfg.get("cache_version", "1"))
        self.prefix = f"rag:answer-cache:v{version}:"

    async def _corpus_version(self, client: Any) -> str:
        version = await client.get("rag:corpus-version")
        # Redis INCR creates a missing key at 1, so the pre-ingest namespace is 0.
        return str(version or "0")

    async def get(self, query: str, context: dict[str, Any]) -> str | None:
        """Return a cached raw model answer for the exact effective context."""
        if not self.enabled:
            return None

        client = _get_redis()
        if not client:
            return None

        try:
            corpus_version = await self._corpus_version(client)
            key = f"{self.prefix}c{corpus_version}:{_hash_payload(query, context)}"
            value = await client.get(key)
            if value:
                logger.info(f"Cache hit: {key}")
            return value
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None

    async def set(self, query: str, context: dict[str, Any], answer: str) -> None:
        """Cache a raw model answer for the exact effective context."""
        if not self.enabled:
            return

        client = _get_redis()
        if not client:
            return

        try:
            corpus_version = await self._corpus_version(client)
            key = f"{self.prefix}c{corpus_version}:{_hash_payload(query, context)}"
            await client.set(key, answer, ex=self.ttl)
            logger.info(f"Cache set: {key}")
        except Exception as e:
            logger.error(f"Cache set failed: {e}")


async def invalidate_semantic_cache() -> None:
    """Invalidate all answers in O(1) by advancing the corpus namespace."""
    client = _get_redis()
    if not client:
        return
    try:
        await client.incr("rag:corpus-version")
    except Exception as exc:
        logger.warning("Semantic cache invalidation failed: %s", exc)


async def set_task_state(task_id: str, state: dict[str, Any], ttl: int = 86400) -> None:
    """Persist upload task state when Redis is available."""
    client = _get_redis()
    if not client:
        return
    try:
        await client.set(f"rag:task:{task_id}", json.dumps(state, ensure_ascii=False), ex=ttl)
    except Exception as exc:
        logger.debug("Task state write failed: %s", exc)


async def get_task_state(task_id: str) -> dict[str, Any] | None:
    client = _get_redis()
    if not client:
        return None
    try:
        value = await client.get(f"rag:task:{task_id}")
        return json.loads(value) if value else None
    except Exception as exc:
        logger.debug("Task state read failed: %s", exc)
        return None


async def get_cache_stats() -> dict[str, Any]:
    """获取缓存统计信息。"""
    client = _get_redis()
    if not client:
        return {"status": "unavailable"}

    try:
        info = await client.info("stats")
        return {
            "status": "connected",
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {"status": "error", "error": str(e)}
