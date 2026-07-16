"""Mocked API route tests for retrieval, generation, and document routes."""
import asyncio

import httpx

from app.main import app

DOC = {
    "id": "doc-1",
    "content": "matched context",
    "score": 0.91,
    "metadata": {"filename": "demo.txt", "chunk_index": 0},
}


async def _post_json(path: str, payload: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post(path, json=payload)


def test_query_and_answer_routes_with_mocked_engine_and_generator(monkeypatch):
    import app.api.query as query_api

    retrieve_calls = []

    class FakeEngine:
        async def retrieve(self, **kwargs):
            retrieve_calls.append(kwargs)
            return [DOC]

    class FakeGenerator:
        async def generate(self, query, context_docs, use_cache=True):
            assert query == "what happened"
            assert context_docs == [DOC]
            return "mock answer"

    monkeypatch.setattr(query_api, "_retrieval_engine", lambda: FakeEngine())
    monkeypatch.setattr(query_api, "Generator", FakeGenerator)

    async def run():
        query_response = await _post_json(
            "/api/v1/query",
            {"query": "what happened", "top_k": 3, "similarity_threshold": 0.1},
        )
        answer_response = await _post_json(
            "/api/v1/answer",
            {"query": "what happened", "top_k": 3},
        )

        assert query_response.status_code == 200
        assert query_response.json()["total"] == 1
        assert query_response.json()["documents"][0]["id"] == "doc-1"
        assert answer_response.status_code == 200
        assert answer_response.json()["answer"] == "mock answer"
        assert answer_response.json()["sources"][0]["metadata"]["filename"] == "demo.txt"
        assert retrieve_calls[0]["similarity_threshold"] == 0.1
        assert retrieve_calls[1]["similarity_threshold"] == 0.05

    asyncio.run(run())


def test_query_response_score_uses_rerank_or_score_before_rrf():
    import app.api.query as query_api

    doc = {
        "id": "ranked",
        "content": "matched context",
        "score": 0.62,
        "rrf_score": 9.0,
        "metadata": {"rerank_prob": 0.84, "chunk_index": 0},
    }

    retrieved = query_api._to_retrieved_document(doc)

    assert retrieved.score == 0.84


def test_standard_query_and_answer_refuse_known_out_of_scope_before_retrieval(monkeypatch):
    import app.api.query as query_api

    def fail_engine():
        raise AssertionError("Known out-of-scope queries must not start retrieval")

    monkeypatch.setattr(query_api, "_retrieval_engine", fail_engine)

    async def run():
        query_response = await _post_json(
            "/api/v1/query",
            {"query": "增值税专用发票抵扣期限如何规定？"},
        )
        answer_response = await _post_json(
            "/api/v1/answer",
            {"query": "增值税专用发票抵扣期限如何规定？"},
        )
        stream_response = await _post_json(
            "/api/v1/answer",
            {"query": "增值税专用发票抵扣期限如何规定？", "stream": True},
        )

        assert query_response.status_code == 200
        assert query_response.json()["documents"] == []
        assert answer_response.status_code == 200
        assert answer_response.json()["sources"] == []
        assert "没有找到足够相关的信息" in answer_response.json()["answer"]
        assert stream_response.status_code == 200
        assert '"type": "sources", "data": []' in stream_response.text
        assert '"type": "done"' in stream_response.text

    asyncio.run(run())


def test_simple_query_and_answer_routes_with_mocks(monkeypatch):
    import app.api.query_simple as query_simple_api
    import app.service.chat_service as chat_service

    retrieve_calls = []

    class FakeEngine:
        async def retrieve(self, **kwargs):
            retrieve_calls.append(kwargs)
            return [DOC]

    class FakeGenerator:
        async def generate(self, query, context_docs, use_cache=True):
            return f"answer for {query}"

    monkeypatch.setattr(query_simple_api, "RetrievalEngine", FakeEngine)
    monkeypatch.setattr(chat_service, "Generator", FakeGenerator)

    async def run():
        query_response = await _post_json("/api/v1/query_simple", {"query": "simple"})
        answer_response = await _post_json("/api/v1/answer_simple", {"query": "simple"})

        assert query_response.status_code == 200
        assert query_response.json()["total"] == 1
        assert answer_response.status_code == 200
        assert answer_response.json()["answer"] == "answer for simple"
        assert answer_response.json()["sources"][0]["id"] == "doc-1"
        assert [call["similarity_threshold"] for call in retrieve_calls] == [0.05, 0.05]

    asyncio.run(run())


def test_chat_route_uses_config_threshold_by_default(monkeypatch):
    from types import SimpleNamespace

    import app.api.chat as chat_api

    retrieve_calls = []

    class FakeService:
        async def run(self, options):
            retrieve_calls.append(options)
            return SimpleNamespace(results=[DOC], answer="chat answer")

    monkeypatch.setattr(chat_api, "EnhancedQueryService", FakeService)

    async def run():
        response = await _post_json(
            "/api/v1/chat",
            {"messages": [{"role": "user", "content": "chat question"}], "top_k": 3},
        )

        assert response.status_code == 200
        assert response.json()["message"]["content"] == "chat answer"
        assert retrieve_calls[0].similarity_threshold == 0.05
        assert retrieve_calls[0].top_k == 3

    asyncio.run(run())


def test_chat_without_user_message_preserves_client_error():
    async def run():
        response = await _post_json(
            "/api/v1/chat",
            {"messages": [{"role": "assistant", "content": "hello"}]},
        )
        assert response.status_code == 400

    asyncio.run(run())


def test_enhanced_query_route_with_mocked_understanding_retrieval_and_generation(monkeypatch):
    import app.service.enhanced_query_service as enhanced_service

    understanding = {
        "original_query": "enhanced",
        "resolved_query": "enhanced resolved",
        "rewritten_query": "enhanced rewritten",
        "retrieval_signals": {},
        "subqueries": ["enhanced rewritten"],
        "retrieval_queries": ["enhanced", "enhanced rewritten"],
        "is_decomposed": False,
    }
    retrieve_calls = []

    class FakeUnderstanding:
        async def understand_query(self, **kwargs):
            return understanding

    class FakeEngine:
        async def retrieve(
            self,
            query,
            top_k=5,
            similarity_threshold=None,
            enable_rerank=True,
            partition=None,
            enable_exact_citations=True,
        ):
            retrieve_calls.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "enable_rerank": enable_rerank,
                    "partition": partition,
                }
            )
            return [{**DOC, "id": query, "score": 0.9}]

    class FakeGenerator:
        async def generate(self, query, context_docs, use_cache=True):
            assert query == "enhanced resolved"
            assert len(context_docs) == 2
            return "enhanced answer"

    monkeypatch.setattr(enhanced_service, "QueryUnderstanding", FakeUnderstanding)
    monkeypatch.setattr(enhanced_service, "HybridRetrievalEngine", FakeEngine)
    monkeypatch.setattr(enhanced_service, "Generator", FakeGenerator)

    async def run():
        response = await _post_json("/api/v1/query/enhanced", {"query": "enhanced", "top_k": 5})

        assert response.status_code == 200
        body = response.json()
        assert body["understanding"]["resolved_query"] == "enhanced resolved"
        assert body["answer"] == "enhanced answer"
        assert len(body["results"]) == 2
        assert [call["similarity_threshold"] for call in retrieve_calls] == [0.05, 0.05]

    asyncio.run(run())


def test_enhanced_query_refuses_missing_requested_law_before_decomposed_generation(monkeypatch):
    import app.service.enhanced_query_service as enhanced_service

    understanding = {
        "original_query": "（公司法） 公司为股东提供担保是否需要股东会决议？",
        "resolved_query": "（公司法） 公司为股东提供担保是否需要股东会决议？",
        "rewritten_query": "公司为股东提供担保 决议机关",
        "subqueries": ["公司为股东提供担保", "决议机关"],
        "retrieval_queries": ["公司为股东提供担保", "决议机关"],
        "is_decomposed": True,
    }

    class FakeUnderstanding:
        async def understand_query(self, **kwargs):
            return understanding

    class FakeEngine:
        async def retrieve(
            self,
            query,
            top_k=5,
            similarity_threshold=None,
            enable_rerank=True,
            partition=None,
            enable_exact_citations=True,
        ):
            return [
                {
                    **DOC,
                    "id": query,
                    "metadata": {"filename": "中华人民共和国民法典_20200528.docx"},
                }
            ]

    class FakeGenerator:
        async def generate(self, **kwargs):
            raise AssertionError("Generator should not run when requested law is missing")

    monkeypatch.setattr(enhanced_service, "QueryUnderstanding", FakeUnderstanding)
    monkeypatch.setattr(enhanced_service, "HybridRetrievalEngine", FakeEngine)
    monkeypatch.setattr(enhanced_service, "Generator", FakeGenerator)

    async def run():
        response = await _post_json(
            "/api/v1/query/enhanced",
            {"query": understanding["original_query"], "top_k": 5},
        )

        assert response.status_code == 200
        body = response.json()
        assert "当前知识库未检索到《公司法》相关文档" in body["answer"]
        assert body["sub_answers"] == []

    asyncio.run(run())


def test_upload_status_list_and_delete_routes_with_mocks(monkeypatch, tmp_path):
    import app.api.upload as upload_api
    import app.vectorstore.storage_adapter as storage_adapter

    upload_api._task_registry.clear()
    monkeypatch.setattr(
        upload_api,
        "config",
        {
            "document_processing": {
                "upload_dir": str(tmp_path),
                "supported_formats": ["txt"],
                "max_file_size": 1024,
            },
            "queue": {"provider": "memory"},
        },
    )

    def fake_process_document_memory(
        task_id, file_path, filename, partition, metadata, app_loop=None
    ):
        upload_api._task_registry[task_id] = {
            "status": "completed",
            "progress": 100,
            "total_chunks": 2,
            "error": None,
        }

    async def fake_delete_document(document_id):
        fake_delete_document.deleted = document_id
        return True

    async def fake_list_documents(skip=0, limit=100):
        return [{"document_id": "doc-1", "filename": "demo.txt"}]

    async def fake_count_documents():
        return 23

    async def fake_list_document_chunks(document_id, skip=0, limit=2000):
        fake_list_document_chunks.document_id = document_id
        return {
            "total": 1,
            "chunks": [
                {
                    "id": "chunk-1",
                    "content": "stored chunk",
                    "content_length": 12,
                    "metadata": {"document_id": document_id, "chunk_index": 0},
                    "partition": "text",
                    "chunk_index": 0,
                    "chunk_strategy": "recursive",
                }
            ],
        }

    monkeypatch.setattr(upload_api, "_process_document_memory", fake_process_document_memory)
    monkeypatch.setattr(storage_adapter, "delete_document", fake_delete_document)
    monkeypatch.setattr(storage_adapter, "list_documents", fake_list_documents)
    monkeypatch.setattr(storage_adapter, "count_documents", fake_count_documents)
    monkeypatch.setattr(storage_adapter, "list_document_chunks", fake_list_document_chunks)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            invalid_json = await client.post(
                "/api/v1/documents/ingest",
                files={"file": ("sample.txt", b"sample content", "text/plain")},
                data={"metadata": "{invalid"},
            )
            non_object_json = await client.post(
                "/api/v1/documents/ingest",
                files={"file": ("sample.txt", b"sample content", "text/plain")},
                data={"metadata": "[]"},
            )
            unsupported_file = await client.post(
                "/api/v1/documents/ingest",
                files={"file": ("sample.pptx", b"not a presentation", "application/octet-stream")},
            )
            ingest_response = await client.post(
                "/api/v1/documents/ingest",
                files={"file": ("sample.txt", b"sample content", "text/plain")},
                data={"partition": "text", "metadata": '{"source":"test"}'},
            )
            assert ingest_response.status_code == 200
            task_id = ingest_response.json()["task_id"]

            status_response = await client.get(f"/api/v1/documents/status/{task_id}")
            list_response = await client.get("/api/v1/documents")
            chunks_response = await client.get("/api/v1/documents/doc-1/chunks")
            delete_response = await client.delete("/api/v1/documents/doc-1")

        assert invalid_json.status_code == 400
        assert non_object_json.status_code == 400
        assert unsupported_file.status_code == 400
        assert list_response.json()["total"] == 23
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "completed"
        assert list_response.status_code == 200
        assert len(list_response.json()["documents"]) == 1
        assert chunks_response.status_code == 200
        assert chunks_response.json()["chunks"][0]["id"] == "chunk-1"
        assert fake_list_document_chunks.document_id == "doc-1"
        assert delete_response.status_code == 200
        assert fake_delete_document.deleted == "doc-1"

    asyncio.run(run())


def test_memory_upload_keeps_redis_work_on_application_loop(monkeypatch, tmp_path):
    import app.api.upload as upload_api
    import app.service.ingest_service as ingest_service

    source = tmp_path / "loop-safe.txt"
    source.write_text("loop safe upload content", encoding="utf-8")
    operations = []
    process_calls = []

    class FakeAppLoop:
        def is_closed(self):
            return False

    def fake_run_on_app_loop(app_loop, coroutine, operation):
        assert isinstance(app_loop, FakeAppLoop)
        operations.append(operation)
        coroutine.close()
        return True

    async def fake_process_document(*args, **kwargs):
        process_calls.append(kwargs)
        return {"status": "completed", "total_chunks": 1}

    monkeypatch.setattr(upload_api, "_run_on_app_loop", fake_run_on_app_loop)
    monkeypatch.setattr(ingest_service, "process_document", fake_process_document)

    upload_api._process_document_memory(
        task_id="loop-safe-task",
        file_path=str(source),
        filename=source.name,
        partition="general",
        metadata={},
        app_loop=FakeAppLoop(),
    )

    assert process_calls == [{"invalidate_cache": False}]
    assert any("invalidate semantic cache" in operation for operation in operations)
    assert upload_api._task_registry["loop-safe-task"]["status"] == "completed"
    assert not source.exists()


def test_failed_memory_upload_is_quarantined(monkeypatch, tmp_path):
    import app.api.upload as upload_api

    monkeypatch.setattr(
        upload_api,
        "config",
        {"document_processing": {"upload_dir": str(tmp_path)}},
    )
    source = tmp_path / "failed.txt"
    source.write_text("retry me", encoding="utf-8")

    upload_api._quarantine_failed_upload(str(source), "task-1")

    retained = tmp_path / "failed" / "failed.txt"
    assert retained.read_text(encoding="utf-8") == "retry me"
    assert not source.exists()


def test_queue_provider_environment_override(monkeypatch):
    import app.api.upload as upload_api

    monkeypatch.setenv("QUEUE_PROVIDER", "celery")
    assert upload_api._queue_provider() == "celery"
