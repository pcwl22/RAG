"""Answer-cache correctness tests."""
import asyncio

import app.utils.cache as cache_module


class FakeRedis:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value

    async def incr(self, key):
        value = int(self.values.get(key, 0)) + 1
        self.values[key] = str(value)
        return value


def test_answer_cache_binds_context_and_corpus_version(monkeypatch):
    async def run():
        redis = FakeRedis()
        monkeypatch.setattr(cache_module, "_redis_client", redis)
        cache = cache_module.SemanticCache()

        context_a = {"documents": [{"id": "a", "content_sha256": "aaa"}]}
        context_b = {"documents": [{"id": "b", "content_sha256": "bbb"}]}

        await cache.set("同一个问题", context_a, "answer-a")
        assert await cache.get("同一个问题", context_a) == "answer-a"
        assert await cache.get("同一个问题", context_b) is None

        await cache_module.invalidate_semantic_cache()
        assert await cache.get("同一个问题", context_a) is None

    asyncio.run(run())


def test_answer_cache_normalizes_query_whitespace(monkeypatch):
    async def run():
        redis = FakeRedis()
        monkeypatch.setattr(cache_module, "_redis_client", redis)
        cache = cache_module.SemanticCache()
        context = {"documents": []}

        await cache.set("劳动合同  解除", context, "answer")
        assert await cache.get("劳动合同 解除", context) == "answer"

    asyncio.run(run())
