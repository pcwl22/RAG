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


class _Completions:
    def create(self, **kwargs):
        assert kwargs["stream"] is True

        def _chunks():
            yield _Chunk("第一段")
            time.sleep(0.25)
            yield _Chunk("第二段")

        return _chunks()


class _Chat:
    completions = _Completions()


class _ZhipuClient:
    chat = _Chat()


def test_zhipu_stream_yields_first_chunk_without_buffering_entire_response():
    client = object.__new__(LLMClient)
    client.config = {"model_name": "glm-4-plus"}
    client._client = _ZhipuClient()

    async def collect():
        stream = client._generate_zhipu_stream("prompt", None, 0.1, 128)
        start = time.perf_counter()
        first = await anext(stream)
        first_elapsed = time.perf_counter() - start
        rest = [chunk async for chunk in stream]
        return first, first_elapsed, rest

    first, first_elapsed, rest = asyncio.run(collect())

    assert first == "第一段"
    assert first_elapsed < 0.15
    assert rest == ["第二段"]
