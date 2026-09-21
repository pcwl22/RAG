"""Mock integration tests for vectorstore adapters and LLM clients."""
import asyncio
from types import SimpleNamespace


def test_storage_adapter_uses_postgres_backend(monkeypatch):
    import app.vectorstore.postgres_store as postgres_store
    import app.vectorstore.storage_adapter as storage_adapter

    async def fake_add(ids, embeddings, documents, metadatas=None, partition="general"):
        return len(ids)

    async def fake_vector_search(query_embedding, top_k=5, partition=None):
        return [{"id": "pg-vector", "score": 0.9, "content": "vector", "metadata": {}}]

    async def fake_hybrid_search(**kwargs):
        fake_hybrid_search.kwargs = kwargs
        return [{"id": "pg-hybrid", "score": 0.8, "content": "hybrid", "metadata": {}}]

    async def fake_delete(document_id):
        fake_delete.document_id = document_id

    async def fake_list(skip=0, limit=100):
        return [{"document_id": "doc-1"}]

    async def fake_list_chunks(document_id, skip=0, limit=2000):
        fake_list_chunks.kwargs = {"document_id": document_id, "skip": skip, "limit": limit}
        return {"total": 1, "chunks": [{"id": "chunk-1", "content": "stored"}]}

    monkeypatch.setattr(postgres_store, "add_documents", fake_add)
    monkeypatch.setattr(postgres_store, "vector_search", fake_vector_search)
    monkeypatch.setattr(postgres_store, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(postgres_store, "delete_document", fake_delete)
    monkeypatch.setattr(postgres_store, "list_documents", fake_list)
    monkeypatch.setattr(postgres_store, "list_document_chunks", fake_list_chunks)

    async def run():
        added = await storage_adapter.add_documents(["id1"], [[0.1]], ["doc"])
        vector = await storage_adapter.search([0.1], top_k=1)
        hybrid = await storage_adapter.hybrid_search("query", [0.1], top_k=2)
        await storage_adapter.delete_document("doc-1")
        listed = await storage_adapter.list_documents()
        chunks = await storage_adapter.list_document_chunks("doc-1", limit=50)

        assert added == 1
        assert vector[0]["id"] == "pg-vector"
        assert hybrid[0]["id"] == "pg-hybrid"
        assert fake_hybrid_search.kwargs["enable_rrf"] is True
        assert fake_delete.document_id == "doc-1"
        assert listed == [{"document_id": "doc-1"}]
        assert chunks["chunks"][0]["id"] == "chunk-1"
        assert fake_list_chunks.kwargs == {"document_id": "doc-1", "skip": 0, "limit": 50}

    asyncio.run(run())


def test_openai_compatible_llm_client_with_fake_client():
    from app.llm.model import LLMClient

    class AsyncStream:
        def __init__(self, items):
            self.items = iter(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("stream"):
                return AsyncStream(
                    [
                        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))]),
                        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" world"))]),
                    ]
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="complete answer"))]
            )

    client = LLMClient.__new__(LLMClient)
    client.provider = "openai_compatible"
    client.config = {"model_name": "fake-model", "temperature": 0.1, "max_tokens": 64}
    completions = FakeCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    async def run():
        answer = await client.generate("prompt", system_prompt="system")
        stream = "".join([chunk async for chunk in client.generate_stream("prompt")])
        deterministic = await client.generate("prompt", temperature=0, max_tokens=0)

        assert answer == "complete answer"
        assert stream == "hello world"
        assert deterministic == "complete answer"
        assert completions.calls[-1]["temperature"] == 0
        assert completions.calls[-1]["max_tokens"] == 0

    asyncio.run(run())


def test_structured_output_options_are_scoped_to_official_deepseek():
    from app.llm.model import LLMClient

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
            )

    client = LLMClient.__new__(LLMClient)
    client.provider = "openai_compatible"
    client.config = {
        "model_name": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "temperature": 0,
        "max_tokens": 64,
        "max_retries": 0,
    }
    completions = FakeCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    async def run():
        await client.generate("prompt", structured_output=True)
        await client.generate("prompt", structured_output=False)
        client.config["base_url"] = "https://proxy.example/v1"
        await client.generate("prompt", structured_output=True)

    asyncio.run(run())

    official, ordinary, proxy = completions.calls
    assert official["reasoning_effort"] == "none"
    assert official["extra_body"] == {"thinking": {"type": "disabled"}}
    assert official["response_format"] == {"type": "json_object"}
    for call in (ordinary, proxy):
        assert "reasoning_effort" not in call
        assert "extra_body" not in call
        assert "response_format" not in call


def test_structured_output_policy_rejects_self_hashed_semantic_tampering():
    import hashlib
    import json

    from app.llm.request_policy import (
        build_structured_output_policy,
        normalize_structured_output_policy,
    )

    policy = build_structured_output_policy(
        provider="openai_compatible",
        model_name="deepseek-flash",
        base_url="https://api.deepseek.com/v1",
    )
    assert normalize_structured_output_policy(policy) == policy

    snapshot = dict(policy["snapshot"])
    snapshot["thinking"] = "provider_default"
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tampered = {"sha256": hashlib.sha256(encoded).hexdigest(), "snapshot": snapshot}

    assert normalize_structured_output_policy(tampered) is None


def test_official_policy_requires_exact_secure_deepseek_endpoint():
    from app.llm.request_policy import build_structured_output_policy

    for base_url in (
        "http://api.deepseek.com/v1",
        "https://api.deepseek.com.evil.example/v1",
        "https://user@api.deepseek.com/v1",
        "https://api.deepseek.com:8443/v1",
    ):
        policy = build_structured_output_policy(
            provider="openai_compatible",
            model_name="deepseek-flash",
            base_url=base_url,
        )
        assert policy["snapshot"]["mode"] == "prompt_only"


def test_openai_compatible_sdk_retry_is_disabled(monkeypatch):
    import openai

    from app.llm.model import LLMClient

    captured = {}

    def fake_async_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(openai, "AsyncOpenAI", fake_async_openai)
    client = LLMClient.__new__(LLMClient)
    client.provider = "openai_compatible"
    client.config = {
        "api_key": "test-api-key",
        "base_url": "https://provider.example/v1",
        "timeout": 45,
    }

    assert client._init_openai_compatible() is not None
    assert captured["max_retries"] == 0
    assert captured["timeout"] == 45


def test_openai_compatible_retries_transient_gateway_function_reference_error():
    from app.llm.model import LLMClient

    class FlakyCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("Function id 'gateway-function' is not found")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="recovered"))]
            )

    client = LLMClient.__new__(LLMClient)
    client.provider = "openai_compatible"
    client.config = {
        "model_name": "fake-model",
        "max_retries": 1,
        "retry_backoff_seconds": 0,
    }
    completions = FlakyCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    async def run():
        return await client.generate("prompt")

    assert asyncio.run(run()) == "recovered"
    assert completions.calls == 2


def test_openai_compatible_retries_transport_error():
    from app.llm.model import LLMClient

    class APIConnectionError(Exception):
        pass

    class FlakyCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise APIConnectionError("Connection error.")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="recovered"))]
            )

    client = LLMClient.__new__(LLMClient)
    client.provider = "openai_compatible"
    client.config = {
        "model_name": "fake-model",
        "max_retries": 1,
        "retry_backoff_seconds": 0,
    }
    completions = FlakyCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    async def run():
        return await client.generate("prompt")

    assert asyncio.run(run()) == "recovered"
    assert completions.calls == 2
