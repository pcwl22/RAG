"""
缓存模块 - Redis + 语义缓存

语义缓存工作原理：
1. 查询向量经过哈希（降维 → 量化 → 字符串）
2. 用哈希值作 key 在 Redis 查询，命中则直接返回缓存的答案
3. 未命中则调用下游（检索+生成），并将结果存入缓存

注意：语义缓存的相似度判断在向量空间进行（需要先存向量，再计算余弦相似度），
这里简化为哈希匹配（精确命中），适合完全相同或极相似的查询。
生产环境可以用 Redis Vector Search 或单独的向量缓存层。
"""
import hashlib
import json
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


async def init_redis() -> None:
    """初始化 Redis 连接。"""
    global _redis_client

    if _redis_client is not None:
        return

    if aioredis is None:
        logger.warning("redis package is not installed. Caching will be disabled.")
        return

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
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Caching will be disabled.")
        _redis_client = None


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


def _hash_embedding(embedding: list[float]) -> str:
    """将向量哈希为字符串（简化版，精确匹配）。

    生产环境建议用 LSH (Locality-Sensitive Hashing) 或直接存向量做相似度查询。
    """
    # 量化到 2 位小数，降低精度提高命中率
    quantized = [round(x, 2) for x in embedding]
    vec_str = json.dumps(quantized)
    return hashlib.sha256(vec_str.encode()).hexdigest()[:16]


class SemanticCache:
    """语义缓存（基于查询向量哈希）。"""

    def __init__(self):
        cfg = _redis_config().get("semantic_cache", {})
        self.enabled = cfg.get("enabled", True)
        self.ttl = cfg.get("ttl", 3600)  # 1小时
        self.prefix = "semantic_cache:"

    async def get(self, query_embedding: list[float]) -> str | None:
        """根据查询向量查缓存。

        Args:
            query_embedding: 查询向量

        Returns:
            缓存的答案（若命中），否则 None
        """
        if not self.enabled:
            return None

        client = _get_redis()
        if not client:
            return None

        key = self.prefix + _hash_embedding(query_embedding)
        try:
            value = await client.get(key)
            if value:
                logger.info(f"Cache hit: {key}")
            return value
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None

    async def set(self, query_embedding: list[float], answer: str) -> None:
        """将答案存入缓存。

        Args:
            query_embedding: 查询向量
            answer: 生成的答案
        """
        if not self.enabled:
            return

        client = _get_redis()
        if not client:
            return

        key = self.prefix + _hash_embedding(query_embedding)
        try:
            await client.set(key, answer, ex=self.ttl)
            logger.info(f"Cache set: {key}")
        except Exception as e:
            logger.error(f"Cache set failed: {e}")


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
