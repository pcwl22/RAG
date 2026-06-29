"""Mock integration tests for vectorstore adapters and LLM clients."""
import asyncio
from types import SimpleNamespace


def test_qdrant_adapter_routes_to_postgres_backend(monkeypatch):
    import app.vectorstore.postgres_store as postgres_store
    import app.vectorstore.qdrant_client as qdrant_client

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

    monkeypatch.setattr(qdrant_client, "_backend", "postgres")
    monkeypatch.setattr(postgres_store, "add_documents", fake_add)
    monkeypatch.setattr(postgres_store, "vector_search", fake_vector_search)
    monkeypatch.setattr(postgres_store, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(postgres_store, "delete_document", fake_delete)
    monkeypatch.setattr(postgres_store, "list_documents", fake_list)

    async def run():
        added = await qdrant_client.add_documents(["id1"], [[0.1]], ["doc"])
        vector = await qdrant_client.search([0.1], top_k=1)
        hybrid = await qdrant_client.hybrid_search("query", [0.1], top_k=2)
        await qdrant_client.delete_document("doc-1")
        listed = await qdrant_client.list_documents()

        assert added == 1
        assert vector[0]["id"] == "pg-vector"
        assert hybrid[0]["id"] == "pg-hybrid"
        assert fake_hybrid_search.kwargs["enable_rrf"] is True
        assert fake_delete.document_id == "doc-1"
        assert listed == [{"document_id": "doc-1"}]

    asyncio.run(run())


def test_qdrant_hybrid_search_falls_back_to_chroma_search(monkeypatch):
    import app.vectorstore.chroma_store as chroma_store
    import app.vectorstore.qdrant_client as qdrant_client

    async def fake_search(query_embedding, top_k=5, partition=None, where=None):
        fake_search.kwargs = {
            "query_embedding": query_embedding,
            "top_k": top_k,
            "partition": partition,
            "where": where,
        }
        return [{"id": "chroma", "score": 0.7, "content": "fallback", "metadata": {}}]

    monkeypatch.setattr(qdrant_client, "_backend", "chroma")
    monkeypatch.setattr(chroma_store, "search", fake_search)

    async def run():
        result = await qdrant_client.hybrid_search("query", [0.2], top_k=3, partition="text")

        assert result[0]["id"] == "chroma"
        assert fake_search.kwargs["top_k"] == 3
        assert fake_search.kwargs["partition"] == "text"

    asyncio.run(run())


def test_chroma_store_with_fake_collection(monkeypatch):
    import app.vectorstore.chroma_store as chroma_store

    class FakeCollection:
        def __init__(self):
            self.added = None
            self.deleted = None

        def add(self, **kwargs):
            self.added = kwargs

        def query(self, **kwargs):
            self.query_kwargs = kwargs
            return {
                "ids": [["id1"]],
                "documents": [["stored document"]],
                "metadatas": [[{"document_id": "doc-1", "filename": "demo.txt", "chunk_index": 2}]],
                "distances": [[0.25]],
            }

        def get(self, include=None):
            return {
                "metadatas": [
                    {"document_id": "doc-1", "filename": "demo.txt", "partition": "text"},
                    {"document_id": "doc-1", "filename": "demo.txt", "partition": "text"},
                ]
            }

        def delete(self, where):
            self.deleted = where

    fake_collection = FakeCollection()
    monkeypatch.setattr(chroma_store, "_collection", fake_collection)

    async def run():
        added = await chroma_store.add_documents(
            ids=["id1"],
            embeddings=[[0.1, 0.2]],
            documents=["stored document"],
            metadatas=[{"document_id": "doc-1", "filename": "demo.txt"}],
            partition="text",
        )
        results = await chroma_store.search([0.1, 0.2], top_k=1, partition="text")
        listed = await chroma_store.list_documents()
        await chroma_store.delete_document("doc-1")

        assert added == 1
        assert fake_collection.added["metadatas"][0]["partition"] == "text"
        assert fake_collection.query_kwargs["where"] == {"partition": "text"}
        assert results[0]["score"] == 0.75
        assert results[0]["chunk_index"] == 2
        assert listed == [
            {
                "document_id": "doc-1",
                "filename": "demo.txt",
                "partition": "text",
                "chunk_count": 2,
            }
        ]
        assert fake_collection.deleted == {"document_id": "doc-1"}

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
