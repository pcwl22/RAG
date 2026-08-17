"""
缓存模块 - Redis + 语义缓存

缓存键绑定查询、实际检索上下文、模型/提示词配置与知识库版本。
当前实现是确定性的答案缓存，不声称执行向量相似度检索；如需真正的
语义近邻缓存，应单独接入 Redis Vector Search 并校准命中阈值。
"""
import hashlib
import json
import os
import time
from typing import Any
from urllib.parse import urlsplit

from app.auth import current_tenant_id, normalize_tenant_id
from app.utils.config import get_config_section
from app.utils.logger import get_logger
from app.utils.metrics import CACHE_LOOKUPS

aioredis: Any
try:
    import redis.asyncio as aioredis
except ImportError:  # Redis is optional; the app can run without semantic cache.
    aioredis = None

logger = get_logger(__name__)

# 全局 Redis 客户端
_redis_client: Any | None = None


def _redis_config() -> dict[str, Any]:
    """获取 Redis 配置。"""
    return get_config_section("redis")


async def init_redis() -> bool:
    """初始化 Redis 连接。"""
    global _redis_client

    if _redis_client is not None:
        return True

    if aioredis is None:
        logger.warning("redis package is not installed. Caching will be disabled.")
        return False

    cfg = _redis_config()
    url = os.getenv("REDIS_URL") or cfg.get("url", "redis://localhost:6379/0")
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
        parsed = urlsplit(str(url))
        endpoint = f"{parsed.scheme}://{parsed.hostname or 'unknown'}"
        if parsed.port:
            endpoint += f":{parsed.port}"
        logger.info("Redis connected: %s", endpoint)
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


async def check_redis_health() -> bool:
    """Return whether the configured Redis client can answer a live probe."""
    client = _get_redis()
    if client is None:
        return False
    try:
        return bool(await client.ping())
    except Exception:
        logger.warning("Redis readiness probe failed", exc_info=True)
        return False


async def consume_tenant_rate_limit(
    tenant_id: str,
    *,
    limit: int,
    window_seconds: int = 60,
    fail_open: bool = False,
) -> tuple[bool, int]:
    """Consume one request from a Redis-backed, fixed-window tenant quota."""
    if limit <= 0:
        return True, 0
    client = _get_redis()
    if client is None:
        if fail_open:
            # Laptop acceptance may intentionally run without Redis. Production
            # profiles keep the default fail-closed behavior.
            return True, limit
        raise ConnectionError("Redis tenant rate-limit backend is unavailable")

    resolved_tenant = _tenant_key(tenant_id)
    window = max(1, int(window_seconds))
    bucket = int(time.time()) // window
    key = f"rag:{resolved_tenant}:rate:api:{bucket}"
    script = """
    local count = redis.call('INCR', KEYS[1])
    if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
    return count
    """
    count = int(await client.eval(script, 1, key, window + 1))
    return count <= limit, max(0, limit - count)


def _hash_payload(query: str, context: dict[str, Any]) -> str:
    payload = {
        "query": " ".join(query.split()),
        "context": context,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tenant_key(tenant_id: str | None = None) -> str:
    return normalize_tenant_id(tenant_id or current_tenant_id())


class SemanticCache:
    """Versioned deterministic answer cache kept under the legacy class name."""

    def __init__(self) -> None:
        cfg = _redis_config().get("semantic_cache", {})
        self.enabled = cfg.get("enabled", True)
        self.ttl = cfg.get("ttl", 3600)  # 1小时
        # Bump for incompatible cache-schema or prompt/model migrations.
        version = os.getenv("RAG_CACHE_VERSION") or str(cfg.get("cache_version", "1"))
        self.prefix = f"rag:answer-cache:v{version}:"

    async def _corpus_version(self, client: Any, tenant_id: str) -> str:
        version = await client.get(f"rag:{tenant_id}:corpus-version")
        # Redis INCR creates a missing key at 1, so the pre-ingest namespace is 0.
        return str(version or "0")

    async def get(self, query: str, context: dict[str, Any]) -> str | None:
        """Return a cached raw model answer for the exact effective context."""
        if not self.enabled:
            CACHE_LOOKUPS.labels(outcome="disabled").inc()
            return None

        client = _get_redis()
        if not client:
            CACHE_LOOKUPS.labels(outcome="unavailable").inc()
            return None

        try:
            tenant_id = _tenant_key()
            corpus_version = await self._corpus_version(client, tenant_id)
            key = (
                f"rag:{tenant_id}:{self.prefix}"
                f"c{corpus_version}:{_hash_payload(query, context)}"
            )
            value: str | None = await client.get(key)
            if value:
                logger.info(f"Cache hit: {key}")
                CACHE_LOOKUPS.labels(outcome="hit").inc()
            else:
                CACHE_LOOKUPS.labels(outcome="miss").inc()
            return value
        except Exception as e:
            CACHE_LOOKUPS.labels(outcome="error").inc()
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
            tenant_id = _tenant_key()
            corpus_version = await self._corpus_version(client, tenant_id)
            key = (
                f"rag:{tenant_id}:{self.prefix}"
                f"c{corpus_version}:{_hash_payload(query, context)}"
            )
            await client.set(key, answer, ex=self.ttl)
            logger.info(f"Cache set: {key}")
        except Exception as e:
            logger.error(f"Cache set failed: {e}")


async def invalidate_semantic_cache(tenant_id: str | None = None) -> None:
    """Invalidate one tenant's answers in O(1) by advancing its corpus namespace."""
    client = _get_redis()
    if not client:
        return
    try:
        resolved_tenant = _tenant_key(tenant_id)
        await client.incr(f"rag:{resolved_tenant}:corpus-version")
    except Exception as exc:
        logger.warning("Semantic cache invalidation failed: %s", exc)


async def set_task_state(
    task_id: str,
    state: dict[str, Any],
    ttl: int = 86400,
    tenant_id: str | None = None,
) -> None:
    """Persist upload task state when Redis is available."""
    client = _get_redis()
    if not client:
        return
    try:
        resolved_tenant = _tenant_key(tenant_id)
        await client.set(
            f"rag:{resolved_tenant}:task:{task_id}",
            json.dumps(state, ensure_ascii=False),
            ex=ttl,
        )
    except Exception as exc:
        logger.debug("Task state write failed: %s", exc)


async def get_task_state(task_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
    client = _get_redis()
    if not client:
        return None
    try:
        resolved_tenant = _tenant_key(tenant_id)
        value = await client.get(f"rag:{resolved_tenant}:task:{task_id}")
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
