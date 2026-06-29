"""Worker task tests."""

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
