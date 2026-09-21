import asyncio
import time

from app.llm.model import LLMClient


class _Delta:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _EmptyChunk:
    choices = []


class _Completions:
    async def create(self, **kwargs):
        assert kwargs["stream"] is True

        async def _chunks():
            yield _EmptyChunk()
            yield _Chunk("第一段")
            await asyncio.sleep(0.25)
            yield _Chunk("第二段")

        return _chunks()


class _Chat:
    completions = _Completions()


class _OpenAICompatibleClient:
    chat = _Chat()


def test_openai_compatible_stream_yields_first_chunk_without_buffering_entire_response():
    client = object.__new__(LLMClient)
    client.config = {"model_name": "glm-4-plus"}
    client._client = _OpenAICompatibleClient()

    async def collect():
        stream = client._generate_openai_stream("prompt", None, 0.1, 128)
        start = time.perf_counter()
        first = await anext(stream)
        first_elapsed = time.perf_counter() - start
        rest = [chunk async for chunk in stream]
        return first, first_elapsed, rest

    first, first_elapsed, rest = asyncio.run(collect())

    assert first == "第一段"
    assert first_elapsed < 0.15
    assert rest == ["第二段"]


def test_llm_health_probe_is_cached():
    class Models:
        def __init__(self):
            self.calls = 0

        async def list(self):
            self.calls += 1
            return []

    client = object.__new__(LLMClient)
    client.provider = "openai_compatible"
    client.config = {"healthcheck_ttl_seconds": 30}
    client._client = type("Client", (), {"models": Models()})()
    client._health_lock = asyncio.Lock()
    client._health_checked_at = 0.0
    client._health_available = False

    async def probe():
        assert await client.check_health() is True
        assert await client.check_health() is True

    asyncio.run(probe())
    assert client._client.models.calls == 1


def test_openai_sdk_async_paginator_is_consumed_by_health_probe():
    class AsyncPaginator:
        def __aiter__(self):
            async def items():
                yield {"id": "model"}

            return items()

    class Models:
        def __init__(self):
            self.calls = 0

        def list(self):
            self.calls += 1
            return AsyncPaginator()

    client = object.__new__(LLMClient)
    client.provider = "openai_compatible"
    client.config = {"healthcheck_ttl_seconds": 30}
    client._client = type("Client", (), {"models": Models()})()
    client._health_lock = asyncio.Lock()
    client._health_checked_at = 0.0
    client._health_available = False

    assert asyncio.run(client.check_health()) is True
    assert client._client.models.calls == 1


def test_generation_failure_requires_probe_instead_of_poisoning_readiness():
    class Models:
        def __init__(self):
            self.calls = 0

        async def list(self):
            self.calls += 1
            return []

    class Completions:
        async def create(self, **_kwargs):
            raise ValueError("request context is invalid")

    client = object.__new__(LLMClient)
    client.provider = "openai_compatible"
    client.config = {"healthcheck_ttl_seconds": 30}
    client._client = type(
        "Client",
        (),
        {"models": Models(), "chat": type("Chat", (), {"completions": Completions()})()},
    )()
    client._health_lock = asyncio.Lock()
    client._health_checked_at = time.monotonic()
    client._health_available = True

    async def run():
        try:
            await client.generate("bad request")
        except ValueError:
            pass
        else:
            raise AssertionError("generation failure was swallowed")
        assert client._health_checked_at == 0.0
        assert await client.check_health() is True

    asyncio.run(run())
    assert client._client.models.calls == 1
