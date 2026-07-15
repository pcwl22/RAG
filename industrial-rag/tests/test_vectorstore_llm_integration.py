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
        async def create(self, **kwargs):
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
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    async def run():
        answer = await client.generate("prompt", system_prompt="system")
        stream = "".join([chunk async for chunk in client.generate_stream("prompt")])

        assert answer == "complete answer"
        assert stream == "hello world"

    asyncio.run(run())
