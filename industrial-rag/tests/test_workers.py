"""Worker task tests."""

import asyncio

import pytest

from app.workers import tasks


@pytest.fixture
def worker_loop(monkeypatch):
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(tasks, "_ensure_worker_runtime", lambda: loop)
    yield loop
    loop.close()


def test_process_document_task_runner_uses_ingest_service(monkeypatch, worker_loop):
    async def fake_process_document(
        file_path, filename, partition="general", metadata=None, *, tenant_id=None
    ):
        return {
            "document_id": "doc-1",
            "total_chunks": 2,
            "status": "completed",
            "metadata": metadata,
            "tenant_id": tenant_id,
        }

    import app.service.ingest_service as ingest_service

    monkeypatch.setattr(ingest_service, "process_document", fake_process_document)

    result = tasks._run_process_document(
        file_path="sample.txt",
        filename="sample.txt",
        partition="text",
        metadata={"source": "test"},
    )

    assert result["status"] == "completed"
    assert result["total_chunks"] == 2
    assert result["metadata"] == {"source": "test"}
    assert result["tenant_id"] == "00000000-0000-0000-0000-000000000001"


def test_process_document_failure_raises_and_quarantines_source(monkeypatch, tmp_path, worker_loop):
    source = tmp_path / "retry.txt"
    source.write_text("data", encoding="utf-8")

    async def fake_process_document(*args, **kwargs):
        return {"status": "failed", "error": "embedding unavailable"}

    import app.service.ingest_service as ingest_service

    monkeypatch.setattr(ingest_service, "process_document", fake_process_document)

    with pytest.raises(RuntimeError, match="Document ingestion failed"):
        tasks._run_process_document(str(source), source.name)

    assert not source.exists()
    assert (tmp_path / "failed" / "retry.txt").read_text(encoding="utf-8") == "data"


def test_quarantine_failure_does_not_mask_ingestion_error(monkeypatch, tmp_path, worker_loop):
    source = tmp_path / "retry.txt"
    source.write_text("data", encoding="utf-8")

    async def fake_process_document(*args, **kwargs):
        raise RuntimeError("embedding unavailable")

    def fail_quarantine(*args, **kwargs):
        raise OSError("disk full")

    import app.service.ingest_service as ingest_service

    monkeypatch.setattr(ingest_service, "process_document", fake_process_document)
    monkeypatch.setattr(tasks, "quarantine_failed_upload", fail_quarantine)

    with pytest.raises(RuntimeError, match="Document ingestion failed"):
        tasks._run_process_document(str(source), source.name)


def test_worker_runtime_initializes_storage_and_redis(monkeypatch):
    calls = []

    async def init_vector_store():
        calls.append("postgres:init")

    async def init_redis():
        calls.append("redis:init")
        return True

    async def close_vector_store():
        calls.append("postgres:close")

    async def close_redis():
        calls.append("redis:close")

    import app.utils.cache as cache
    import app.vectorstore.storage_adapter as storage_adapter

    monkeypatch.setattr(storage_adapter, "init_vector_store", init_vector_store)
    monkeypatch.setattr(storage_adapter, "close_vector_store", close_vector_store)
    monkeypatch.setattr(cache, "init_redis", init_redis)
    monkeypatch.setattr(cache, "close_redis", close_redis)
    monkeypatch.setattr(tasks, "_worker_loop", None)
    monkeypatch.setattr(tasks, "_worker_runtime_ready", False)

    tasks._ensure_worker_runtime()
    tasks._shutdown_worker_runtime()

    assert calls == ["postgres:init", "redis:init", "redis:close", "postgres:close"]
