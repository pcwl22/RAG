"""Worker task tests."""

import pytest

from app.workers import tasks


def test_process_document_task_runner_uses_ingest_service(monkeypatch):
    async def fake_process_document(file_path, filename, partition="general", metadata=None):
        return {
            "document_id": "doc-1",
            "total_chunks": 2,
            "status": "completed",
            "metadata": metadata,
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


def test_process_document_failure_raises_and_quarantines_source(monkeypatch, tmp_path):
    source = tmp_path / "retry.txt"
    source.write_text("data", encoding="utf-8")

    async def fake_process_document(*args, **kwargs):
        return {"status": "failed", "error": "embedding unavailable"}

    import app.service.ingest_service as ingest_service

    monkeypatch.setattr(ingest_service, "process_document", fake_process_document)

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        tasks._run_process_document(str(source), source.name)

    assert not source.exists()
    assert (tmp_path / "failed" / "retry.txt").read_text(encoding="utf-8") == "data"


def test_quarantine_failure_does_not_mask_ingestion_error(monkeypatch, tmp_path):
    source = tmp_path / "retry.txt"
    source.write_text("data", encoding="utf-8")

    async def fake_process_document(*args, **kwargs):
        raise RuntimeError("embedding unavailable")

    def fail_quarantine(*args, **kwargs):
        raise OSError("disk full")

    import app.service.ingest_service as ingest_service

    monkeypatch.setattr(ingest_service, "process_document", fake_process_document)
    monkeypatch.setattr(tasks, "quarantine_failed_upload", fail_quarantine)

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        tasks._run_process_document(str(source), source.name)
