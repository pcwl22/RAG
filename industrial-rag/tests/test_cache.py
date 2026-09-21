"""Answer-cache correctness tests."""
import asyncio

import pytest

import app.utils.cache as cache_module
from app.auth import Principal, reset_current_principal, set_current_principal


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

    async def ping(self):
        return True

    async def eval(self, _script, _key_count, key, _ttl):
        return await self.incr(key)


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


def test_answer_and_task_cache_are_tenant_scoped(monkeypatch):
    async def run():
        redis = FakeRedis()
        monkeypatch.setattr(cache_module, "_redis_client", redis)
        cache = cache_module.SemanticCache()
        context = {"documents": [{"id": "shared-id"}]}
        tenant_a = "00000000-0000-0000-0000-00000000000a"
        tenant_b = "00000000-0000-0000-0000-00000000000b"

        token = set_current_principal(
            Principal("user-a", tenant_a, frozenset({"viewer"}), "test")
        )
        try:
            await cache.set("同一个问题", context, "tenant-a-answer")
            await cache_module.set_task_state("shared-task", {"status": "completed"})
        finally:
            reset_current_principal(token)

        token = set_current_principal(
            Principal("user-b", tenant_b, frozenset({"viewer"}), "test")
        )
        try:
            assert await cache.get("同一个问题", context) is None
            assert await cache_module.get_task_state("shared-task") is None
            await cache.set("同一个问题", context, "tenant-b-answer")
            assert await cache.get("同一个问题", context) == "tenant-b-answer"
        finally:
            reset_current_principal(token)

    asyncio.run(run())


def test_redis_health_probe_tracks_live_connection(monkeypatch):
    class UnavailableRedis:
        async def ping(self):
            raise ConnectionError("redis unavailable")

    async def run():
        monkeypatch.setattr(cache_module, "_redis_client", None)
        assert await cache_module.check_redis_health() is False

        monkeypatch.setattr(cache_module, "_redis_client", FakeRedis())
        assert await cache_module.check_redis_health() is True

        monkeypatch.setattr(cache_module, "_redis_client", UnavailableRedis())
        assert await cache_module.check_redis_health() is False

    asyncio.run(run())


def test_tenant_rate_limit_is_scoped_and_enforced(monkeypatch):
    async def run():
        monkeypatch.setattr(cache_module, "_redis_client", FakeRedis())
        tenant = "00000000-0000-0000-0000-00000000000a"
        assert await cache_module.consume_tenant_rate_limit(tenant, limit=2) == (True, 1)
        assert await cache_module.consume_tenant_rate_limit(tenant, limit=2) == (True, 0)
        assert await cache_module.consume_tenant_rate_limit(tenant, limit=2) == (False, 0)

    asyncio.run(run())


def test_tenant_rate_limit_requires_explicit_fail_open(monkeypatch):
    async def run():
        monkeypatch.setattr(cache_module, "_redis_client", None)
        tenant = "00000000-0000-0000-0000-00000000000a"
        try:
            await cache_module.consume_tenant_rate_limit(tenant, limit=2)
        except ConnectionError as exc:
            assert "unavailable" in str(exc)
        else:
            raise AssertionError("missing Redis backend was allowed in fail-closed mode")

        assert await cache_module.consume_tenant_rate_limit(
            tenant,
            limit=2,
            fail_open=True,
        ) == (True, 2)

    asyncio.run(run())


def test_tenant_rate_limit_has_an_application_deadline(monkeypatch):
    class HangingRedis:
        async def eval(self, *_args):
            await asyncio.Event().wait()

    async def run():
        monkeypatch.setattr(cache_module, "_redis_client", HangingRedis())
        monkeypatch.setattr(
            cache_module,
            "_redis_config",
            lambda: {"operation_timeout_seconds": 0.01},
        )
        with pytest.raises(TimeoutError):
            await cache_module.consume_tenant_rate_limit(
                "00000000-0000-0000-0000-00000000000a",
                limit=2,
            )

    asyncio.run(run())


def test_redis_client_receives_transport_timeouts(monkeypatch):
    captured = {}

    class ClosableRedis(FakeRedis):
        async def aclose(self):
            captured["closed"] = True

    class FakeRedisModule:
        @staticmethod
        def from_url(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return ClosableRedis()

    async def run():
        monkeypatch.setattr(cache_module, "_redis_client", None)
        monkeypatch.setattr(cache_module, "aioredis", FakeRedisModule())
        monkeypatch.setattr(
            cache_module,
            "_redis_config",
            lambda: {
                "url": "redis://:secret@redis.internal:6379/0",
                "max_connections": 13,
                "socket_connect_timeout_seconds": 1.5,
                "socket_timeout_seconds": 2.5,
                "operation_timeout_seconds": 3.5,
                "health_check_interval_seconds": 17,
            },
        )

        assert await cache_module.init_redis() is True
        assert captured["url"] == "redis://:secret@redis.internal:6379/0"
        assert captured["kwargs"] == {
            "encoding": "utf-8",
            "decode_responses": True,
            "max_connections": 13,
            "socket_connect_timeout": 1.5,
            "socket_timeout": 2.5,
            "health_check_interval": 17,
            "retry_on_timeout": True,
        }
        await cache_module.close_redis()
        assert captured["closed"] is True

    asyncio.run(run())


def test_invalid_operation_timeout_does_not_start_redis_operation(monkeypatch):
    calls = 0

    class TrackingRedis:
        async def ping(self):
            nonlocal calls
            calls += 1
            return True

    async def run():
        monkeypatch.setattr(cache_module, "_redis_client", TrackingRedis())
        monkeypatch.setattr(
            cache_module,
            "_redis_config",
            lambda: {"operation_timeout_seconds": 0},
        )
        assert await cache_module.check_redis_health() is False
        assert calls == 0

    asyncio.run(run())


def test_task_dispatch_lease_is_tenant_scoped_and_atomic(monkeypatch):
    captured = {}

    class LeaseRedis:
        async def set(self, key, value, **kwargs):
            captured.update({"key": key, "value": value, **kwargs})
            return True

    async def run():
        monkeypatch.setattr(cache_module, "_redis_client", LeaseRedis())
        monkeypatch.setattr(
            cache_module,
            "_redis_config",
            lambda: {"operation_timeout_seconds": 1},
        )
        acquired = await cache_module.acquire_task_dispatch_lease(
            "task-123",
            tenant_id="00000000-0000-0000-0000-00000000000a",
            ttl_seconds=45,
        )
        assert acquired is True

    asyncio.run(run())
    assert captured == {
        "key": "rag:00000000-0000-0000-0000-00000000000a:task-dispatch:task-123",
        "value": "1",
        "ex": 45,
        "nx": True,
    }
