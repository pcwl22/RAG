"""Mocked API route tests for retrieval, generation, and document routes."""
import asyncio
import os

import httpx
import pytest

from app.main import app


@pytest.fixture(autouse=True)
def disable_oidc_for_route_mocks(monkeypatch):
    """Route mocks exercise handlers, not an external identity provider."""
    monkeypatch.setenv("OIDC_ENABLED", "false")

DOC = {
    "id": "doc-1",
    "content": "matched context",
    "score": 0.91,
    "metadata": {"filename": "demo.txt", "chunk_index": 0},
}


async def _post_json(path: str, payload: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    headers = {}
    if api_key := os.getenv("RAG_API_KEY"):
        headers["X-API-Key"] = api_key
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post(path, json=payload, headers=headers)


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


def test_answer_stream_emits_terminal_event_after_failure(monkeypatch):
    import app.api.query as query_api

    class FakeEngine:
        async def retrieve(self, **kwargs):
            return [DOC]

    class FailingGenerator:
        async def generate_stream(self, **kwargs):
            if False:
                yield ""
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(query_api, "_retrieval_engine", lambda: FakeEngine())
    monkeypatch.setattr(query_api, "Generator", FailingGenerator)

    async def run():
        response = await _post_json(
            "/api/v1/answer",
            {"query": "what happened", "stream": True},
        )

        assert response.status_code == 200
        assert '"type": "error"' in response.text
        assert '"type": "done", "error": true' in response.text

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

    monkeypatch.setattr(query_simple_api, "build_retrieval_engine", FakeEngine)
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
    import app.service.chat_service as chat_service

    retrieve_calls = []

    class UnexpectedGenerator:
        def __init__(self):
            raise AssertionError("RAG chat must not initialize the direct-chat generator")

    class FakeService:
        async def run(self, options):
            retrieve_calls.append(options)
            return SimpleNamespace(results=[DOC], answer="chat answer")

    monkeypatch.setattr(chat_service, "Generator", UnexpectedGenerator)
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


def test_chat_rag_uses_history_before_last_user_turn(monkeypatch):
    from types import SimpleNamespace

    import app.api.chat as chat_api
    import app.service.chat_service as chat_service

    captured = []

    class UnexpectedGenerator:
        def __init__(self):
            raise AssertionError("RAG chat must not initialize the direct-chat generator")

    class FakeService:
        async def run(self, options):
            captured.append(options)
            return SimpleNamespace(results=[], answer="chat answer")

    monkeypatch.setattr(chat_service, "Generator", UnexpectedGenerator)
    monkeypatch.setattr(chat_api, "EnhancedQueryService", FakeService)

    async def run():
        response = await _post_json(
            "/api/v1/chat",
            {
                "messages": [
                    {"role": "user", "content": "previous question"},
                    {"role": "assistant", "content": "previous answer"},
                    {"role": "user", "content": "current question"},
                    {"role": "assistant", "content": "stale trailing answer"},
                ]
            },
        )

        assert response.status_code == 200
        assert captured[0].query == "current question"
        assert captured[0].chat_history == [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

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
            rerank_top_k=None,
            rerank_apply_threshold=None,
            similarity_threshold=None,
            enable_rerank=True,
            partition=None,
            enable_exact_citations=True,
        ):
            retrieve_calls.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "rerank_top_k": rerank_top_k,
                    "rerank_apply_threshold": rerank_apply_threshold,
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
    monkeypatch.setattr(enhanced_service, "build_retrieval_engine", FakeEngine)
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
            rerank_top_k=None,
            rerank_apply_threshold=None,
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
    monkeypatch.setattr(enhanced_service, "build_retrieval_engine", FakeEngine)
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

    monkeypatch.setenv("RAG_SERVICE_ROLES", "viewer,editor,admin")
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
        task_id, file_path, filename, partition, metadata, tenant_id=None, app_loop=None
    ):
        upload_api._remember_task(task_id, {
            "status": "completed",
            "progress": 100,
            "total_chunks": 2,
            "error": None,
        }, tenant_id)

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
        headers = {}
        if api_key := os.getenv("RAG_API_KEY"):
            headers["X-API-Key"] = api_key
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            invalid_json = await client.post(
                "/api/v1/documents/ingest",
                files={"file": ("sample.txt", b"sample content", "text/plain")},
                data={"metadata": "{invalid"},
                headers=headers,
            )
            non_object_json = await client.post(
                "/api/v1/documents/ingest",
                files={"file": ("sample.txt", b"sample content", "text/plain")},
                data={"metadata": "[]"},
                headers=headers,
            )
            unsupported_file = await client.post(
                "/api/v1/documents/ingest",
                files={"file": ("sample.pptx", b"not a presentation", "application/octet-stream")},
                headers=headers,
            )
            ingest_response = await client.post(
                "/api/v1/documents/ingest",
                files={"file": ("sample.txt", b"sample content", "text/plain")},
                data={"partition": "text", "metadata": '{"source":"test"}'},
                headers=headers,
            )
            assert ingest_response.status_code == 200
            task_id = ingest_response.json()["task_id"]

            status_response = await client.get(f"/api/v1/documents/status/{task_id}", headers=headers)
            list_response = await client.get("/api/v1/documents", headers=headers)
            chunks_response = await client.get("/api/v1/documents/doc-1/chunks", headers=headers)
            delete_response = await client.delete("/api/v1/documents/doc-1", headers=headers)

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


def test_celery_upload_passes_only_tenant_and_object_key(monkeypatch, tmp_path):
    import app.api.upload as upload_api
    import app.workers.tasks as worker_tasks

    calls = []

    class FakeStore:
        def upload_file(self, local_path, object_key):
            calls.append(("upload", object_key))

        def put_json(self, object_key, payload):
            calls.append(("manifest", object_key, payload))

        def delete_object(self, object_key):
            calls.append(("delete", object_key))

    class FakeTask:
        def apply_async(self, *, kwargs, task_id, queue):
            calls.append(("apply_async", kwargs, task_id, queue))
            return self

    monkeypatch.setattr(
        upload_api,
        "config",
        {
            "document_processing": {
                "upload_dir": str(tmp_path),
                "supported_formats": ["txt"],
                "max_file_size": 1024,
            },
            "queue": {"provider": "celery"},
        },
    )
    monkeypatch.setattr(upload_api, "_object_storage_for_queue", lambda _provider: FakeStore())
    monkeypatch.setattr(
        upload_api,
        "_persist_task_state",
        lambda *args, **kwargs: _completed_async(True),
    )
    monkeypatch.setattr(worker_tasks, "process_document_task", FakeTask())
    monkeypatch.setenv("CELERY_TASK_DEFAULT_QUEUE", "release-canary-test-pod")

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/v1/documents/ingest",
                files={"file": ("folder\\sample.txt", b"sample content", "text/plain")},
                data={"partition": "text", "metadata": '{"source":"test"}'},
            )
        return response

    async def _run_and_assert():
        response = await run()
        assert response.status_code == 200
        delay_calls = [call for call in calls if call[0] == "apply_async"]
        assert len(delay_calls) == 1
        assert set(delay_calls[0][1]) == {"tenant_id", "object_key"}
        assert delay_calls[0][1]["object_key"].endswith("/payload.txt")
        assert delay_calls[0][2]
        assert delay_calls[0][3] == "release-canary-test-pod"
        manifest = next(call for call in calls if call[0] == "manifest")
        assert manifest[2]["partition"] == "text"
        assert manifest[2]["metadata"]["source"] == "test"
        assert manifest[2]["metadata"]["source_id"].startswith("upload-name-sha256:")

    async def _completed_async(value=None):
        return value

    asyncio.run(_run_and_assert())


def test_simple_query_and_answer_refuse_known_out_of_scope(monkeypatch):
    import app.api.query_simple as query_simple_api

    class ScopeAwareEngine:
        async def retrieve(self, *, query, **kwargs):
            raise AssertionError("out-of-scope queries must not initialize retrieval")

    monkeypatch.setattr(query_simple_api, "build_retrieval_engine", ScopeAwareEngine)

    async def run():
        query_response = await _post_json(
            "/api/v1/query_simple",
            {"query": "增值税专用发票抵扣期限如何规定？"},
        )
        answer_response = await _post_json(
            "/api/v1/answer_simple",
            {"query": "增值税专用发票抵扣期限如何规定？"},
        )

        assert query_response.status_code == 200
        assert query_response.json()["documents"] == []
        assert answer_response.status_code == 200
        assert answer_response.json()["sources"] == []

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

    assert process_calls == [
        {
            "invalidate_cache": False,
            "tenant_id": "00000000-0000-0000-0000-000000000001",
        }
    ]
    assert any("invalidate semantic cache" in operation for operation in operations)
    assert upload_api._task_registry[
        ("00000000-0000-0000-0000-000000000001", "loop-safe-task")
    ]["status"] == "completed"
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


def test_public_task_state_redacts_internal_failure_details():
    import app.api.upload as upload_api

    state = upload_api._public_task_state(
        {
            "status": "failed",
            "progress": 0,
            "total_chunks": 0,
            "error": "database password appeared in an internal exception",
        }
    )

    assert state["error"] == "Document ingestion failed"
    assert state["error_code"] == "DOCUMENT_INGESTION_FAILED"
    assert "password" not in state["error"]


def test_queue_provider_environment_override(monkeypatch):
    import app.api.upload as upload_api

    monkeypatch.setenv("QUEUE_PROVIDER", "celery")
    assert upload_api._queue_provider() == "celery"


def test_queue_provider_typo_does_not_fall_back_to_memory(monkeypatch):
    import app.api.upload as upload_api

    monkeypatch.setenv("QUEUE_PROVIDER", "celrey")
    with pytest.raises(ValueError, match="Unsupported queue provider"):
        upload_api._queue_provider()


def test_task_registry_evicts_least_recently_used_not_first_inserted(monkeypatch):
    """Both writing and polling a task must protect it from eviction.

    A plain dict keeps its original insertion order when a key is reassigned, so
    an in-flight task would be dropped ahead of untouched completed entries and
    its status poll would return 404.
    """
    import app.api.upload as upload_api

    tenant = "00000000-0000-0000-0000-000000000001"
    upload_api._task_registry.clear()
    monkeypatch.setattr(upload_api, "_MAX_IN_MEMORY_TASKS", 3)

    def remember(task_id):
        upload_api._remember_task(task_id, {"status": "processing"}, tenant)

    def tracked():
        return [task_id for _, task_id in upload_api._task_registry]

    for task_id in ("task-0", "task-1", "task-2"):
        remember(task_id)
    assert tracked() == ["task-0", "task-1", "task-2"]

    # Writing task-0 again makes it the newest, so task-1 is the eviction target.
    remember("task-0")
    remember("task-3")
    assert tracked() == ["task-2", "task-0", "task-3"]

    # task-2 is now the eviction target; reading it must reprieve it, leaving
    # task-0 (untouched the longest) as the one that goes.
    assert asyncio.run(upload_api._load_task_state("task-2", tenant)) is not None
    remember("task-4")
    assert tracked() == ["task-3", "task-2", "task-4"]

    upload_api._task_registry.clear()
