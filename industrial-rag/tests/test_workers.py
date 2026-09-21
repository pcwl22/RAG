"""Worker task tests."""

import asyncio
import contextlib
import importlib

import pytest

from app.workers import tasks
from app.workers.celery_app import create_celery_app, task_default_queue


@pytest.fixture
def worker_loop(monkeypatch):
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(tasks, "_ensure_worker_runtime", lambda: loop)
    yield loop
    loop.close()


def test_process_document_task_runner_uses_ingest_service(monkeypatch, worker_loop):
    async def fake_process_document(
        file_path,
        filename,
        partition="general",
        metadata=None,
        *,
        tenant_id=None,
        internal_metadata=None,
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


def test_celery_default_queue_can_be_isolated_for_release_canary(monkeypatch):
    celery_module = importlib.import_module("app.workers.celery_app")

    class FakeConfig:
        def update(self, **values):
            self.__dict__.update(values)

    class FakeCelery:
        def __init__(self, *_args, **_kwargs):
            self.conf = FakeConfig()

        def autodiscover_tasks(self, _packages):
            return None

    monkeypatch.setattr(celery_module, "Celery", FakeCelery)
    monkeypatch.setenv("CELERY_TASK_DEFAULT_QUEUE", "release-canary-rag-canary-abc12")

    app = create_celery_app()

    assert app is not None
    assert app.conf.task_default_queue == "release-canary-rag-canary-abc12"
    assert app.conf.broker_connection_timeout == 5
    assert app.conf.task_publish_retry is True
    assert app.conf.task_publish_retry_policy["max_retries"] == 3
    assert app.conf.broker_transport_options["socket_timeout"] == 5


def test_celery_transport_timeout_rejects_unbounded_values(monkeypatch):
    celery_module = importlib.import_module("app.workers.celery_app")
    monkeypatch.setenv("CELERY_BROKER_TIMEOUT_SECONDS", "0")

    with pytest.raises(RuntimeError, match="between 1 and 60"):
        celery_module._positive_timeout("CELERY_BROKER_TIMEOUT_SECONDS", 5)


@pytest.mark.parametrize("queue", ["", "contains whitespace", "slash/queue", "x" * 129])
def test_celery_default_queue_rejects_unsafe_names(monkeypatch, queue):
    monkeypatch.setenv("CELERY_TASK_DEFAULT_QUEUE", queue)

    with pytest.raises(RuntimeError, match="CELERY_TASK_DEFAULT_QUEUE"):
        task_default_queue()


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


def test_object_task_downloads_and_deletes_successful_object(monkeypatch, tmp_path, worker_loop):
    calls = []

    class FakeStore:
        def get_json(self, object_key):
            calls.append(("manifest", object_key))
            return {
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "filename": "file.txt",
                "partition": "text",
                "metadata": {},
            }

        def download_file(self, object_key, local_path):
            calls.append(("download", object_key))
            local_path.write_text("document", encoding="utf-8")

        def delete_object(self, object_key):
            calls.append(("delete", object_key))

        def put_json(self, object_key, payload):
            calls.append(("put", object_key))
            assert payload["completion_receipt"]["result"]["status"] == "completed"

        def move_to_failed(self, object_key):
            calls.append(("failed", object_key))
            return "tenants/t/failed/task/file.txt"

    monkeypatch.setattr(tasks, "get_object_store", lambda: FakeStore())
    monkeypatch.setattr(
        tasks,
        "_run_process_document",
        lambda file_path, filename, partition, metadata, tenant_id, **_kwargs: {
            "document_id": "doc-1",
            "source_key": "source-1",
            "status": "completed",
            "total_chunks": 3,
        },
    )

    result = tasks._run_process_document_object(
        "tenants/00000000-0000-0000-0000-000000000001/uploads/task/file.txt",
        "00000000-0000-0000-0000-000000000001",
    )

    assert result["status"] == "completed"
    assert calls == [
        (
            "manifest",
            "tenants/00000000-0000-0000-0000-000000000001/uploads/task/.upload-manifest.json",
        ),
        (
            "download",
            "tenants/00000000-0000-0000-0000-000000000001/uploads/task/file.txt",
        ),
        (
            "put",
            "tenants/00000000-0000-0000-0000-000000000001/uploads/task/.upload-manifest.json",
        ),
        (
            "delete",
            "tenants/00000000-0000-0000-0000-000000000001/uploads/task/file.txt",
        ),
    ]


def test_object_task_replays_completion_receipt_without_reprocessing(monkeypatch, worker_loop):
    object_key = "tenants/00000000-0000-0000-0000-000000000001/uploads/task/payload.txt"
    calls = []
    commit_checks = []

    class FakeStore:
        def get_json(self, manifest_key):
            from app.storage.upload_receipt import build_completion_receipt

            manifest = {
                "version": 1,
                "task_id": "task",
                "object_key": object_key,
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "filename": "劳动合同法.txt",
                "partition": "text",
                "metadata": {},
                "created_at_unix": 100,
            }
            manifest["completion_receipt"] = build_completion_receipt(
                manifest,
                task_id="task",
                tenant_id="00000000-0000-0000-0000-000000000001",
                object_key=object_key,
                result={
                    "document_id": "doc-1",
                    "source_key": "source-1",
                    "total_chunks": 3,
                    "status": "completed",
                },
            )
            return manifest

        def download_file(self, *args, **kwargs):
            raise AssertionError("a completed receipt must skip processing")

        def delete_object(self, key):
            calls.append(key)

        def move_to_failed(self, key):
            raise AssertionError("a valid completed receipt must not be quarantined")

    monkeypatch.setattr(tasks, "get_object_store", lambda: FakeStore())
    import app.vectorstore.postgres_store as postgres_store

    def commit_matches(**kwargs):
        commit_checks.append(kwargs)
        return True

    monkeypatch.setattr(postgres_store, "document_commit_matches", commit_matches)

    result = tasks._run_process_document_object(
        object_key,
        "00000000-0000-0000-0000-000000000001",
    )

    assert result == {
        "document_id": "doc-1",
        "source_key": "source-1",
        "total_chunks": 3,
        "status": "completed",
    }
    assert calls == [object_key]
    assert commit_checks == [
        {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "document_id": "doc-1",
            "source_key": "source-1",
            "upload_task_id": "task",
            "upload_object_key": object_key,
            "total_chunks": 3,
        }
    ]


def test_object_task_rejects_valid_receipt_without_database_commit(
    monkeypatch, worker_loop
):
    object_key = "tenants/00000000-0000-0000-0000-000000000001/uploads/task/payload.txt"
    manifest_key = (
        "tenants/00000000-0000-0000-0000-000000000001/"
        "uploads/task/.upload-manifest.json"
    )
    deleted = []
    quarantined = []

    class FakeStore:
        def get_json(self, _manifest_key):
            from app.storage.upload_receipt import build_completion_receipt

            manifest = {
                "version": 1,
                "task_id": "task",
                "object_key": object_key,
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "filename": "劳动合同法.txt",
                "partition": "text",
                "metadata": {},
                "created_at_unix": 100,
            }
            manifest["completion_receipt"] = build_completion_receipt(
                manifest,
                task_id="task",
                tenant_id="00000000-0000-0000-0000-000000000001",
                object_key=object_key,
                result={
                    "document_id": "forged-doc",
                    "source_key": "forged-source",
                    "total_chunks": 3,
                    "status": "completed",
                },
            )
            return manifest

        def delete_object(self, key):
            deleted.append(key)

        def move_to_failed(self, key):
            quarantined.append(key)
            return f"failed/{key}"

    monkeypatch.setattr(tasks, "get_object_store", lambda: FakeStore())
    import app.vectorstore.postgres_store as postgres_store

    monkeypatch.setattr(
        postgres_store,
        "document_commit_matches",
        lambda **_kwargs: False,
    )

    with pytest.raises(
        tasks.ObjectStorageManifestError,
        match="no matching database commit",
    ):
        tasks._run_process_document_object(
            object_key,
            "00000000-0000-0000-0000-000000000001",
        )

    assert deleted == []
    assert quarantined == [object_key, manifest_key]


def test_object_task_uses_opaque_local_path_and_unicode_manifest_filename(
    monkeypatch, worker_loop
):
    object_key = "tenants/00000000-0000-0000-0000-000000000001/uploads/task/payload.pdf"
    captured = {}

    class FakeStore:
        def get_json(self, manifest_key):
            return {
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "filename": "劳动合同法.pdf",
                "partition": "legal",
                "metadata": {},
            }

        def download_file(self, key, local_path):
            captured["local_name"] = local_path.name
            local_path.write_bytes(b"pdf")

        def put_json(self, key, payload):
            pass

        def delete_object(self, key):
            pass

        def move_to_failed(self, key):
            raise AssertionError("successful upload must not be quarantined")

    def fake_process(file_path, filename, partition, metadata, tenant_id, **_kwargs):
        captured["filename"] = filename
        return {
            "document_id": "doc-1",
            "source_key": "source-1",
            "total_chunks": 1,
            "status": "completed",
        }

    monkeypatch.setattr(tasks, "get_object_store", lambda: FakeStore())
    monkeypatch.setattr(tasks, "_run_process_document", fake_process)

    tasks._run_process_document_object(
        object_key,
        "00000000-0000-0000-0000-000000000001",
    )

    assert captured == {"local_name": "payload.pdf", "filename": "劳动合同法.pdf"}


def test_transient_object_ingestion_failure_keeps_upload_for_retry(
    monkeypatch, worker_loop
):
    calls = []

    class FakeStore:
        def get_json(self, manifest_key):
            return {
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "filename": "file.txt",
                "partition": "text",
                "metadata": {},
            }

        def download_file(self, key, local_path):
            local_path.write_text("document", encoding="utf-8")

        def move_to_failed(self, key):
            calls.append(("failed", key))

    monkeypatch.setattr(tasks, "get_object_store", lambda: FakeStore())
    monkeypatch.setattr(
        tasks,
        "_run_process_document",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            tasks.TransientIngestionError("database unavailable")
        ),
    )

    with pytest.raises(tasks.TransientIngestionError):
        tasks._run_process_document_object(
            "tenants/00000000-0000-0000-0000-000000000001/uploads/task/file.txt",
            "00000000-0000-0000-0000-000000000001",
        )

    assert calls == []


def test_object_task_moves_processing_failure_to_failed_prefix(monkeypatch, worker_loop):
    calls = []

    class FakeStore:
        def get_json(self, object_key):
            return {
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "filename": "file.txt",
                "partition": "text",
                "metadata": {},
            }

        def download_file(self, object_key, local_path):
            local_path.write_text("document", encoding="utf-8")

        def delete_object(self, object_key):
            calls.append("delete")

        def move_to_failed(self, object_key):
            calls.append(("failed", object_key))
            return "tenants/t/failed/task/file.txt"

    monkeypatch.setattr(tasks, "get_object_store", lambda: FakeStore())

    def fail_process(*args, **kwargs):
        raise RuntimeError("processing failed")

    monkeypatch.setattr(tasks, "_run_process_document", fail_process)

    with pytest.raises(RuntimeError, match="processing failed"):
        tasks._run_process_document_object(
            "tenants/00000000-0000-0000-0000-000000000001/uploads/task/file.txt",
            tenant_id="00000000-0000-0000-0000-000000000001",
        )

    assert calls[0][0] == "failed"


def test_object_task_download_failure_leaves_source_for_retry(monkeypatch, worker_loop):
    calls = []

    class FakeStore:
        def get_json(self, object_key):
            calls.append(("manifest", object_key))
            return {
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "filename": "file.txt",
                "partition": "text",
                "metadata": {},
            }

        def download_file(self, object_key, local_path):
            calls.append(("download", object_key))
            raise TimeoutError("object storage unavailable")

        def delete_object(self, object_key):
            calls.append(("delete", object_key))

        def move_to_failed(self, object_key):
            calls.append(("failed", object_key))
            return "unused"

    monkeypatch.setattr(tasks, "get_object_store", lambda: FakeStore())

    with pytest.raises(tasks.ObjectStorageDownloadError):
        tasks._run_process_document_object(
            "tenants/00000000-0000-0000-0000-000000000001/uploads/task/file.txt",
            tenant_id="00000000-0000-0000-0000-000000000001",
        )

    assert calls == [
        (
            "manifest",
            "tenants/00000000-0000-0000-0000-000000000001/uploads/task/.upload-manifest.json",
        ),
        (
            "download",
            "tenants/00000000-0000-0000-0000-000000000001/uploads/task/file.txt",
        ),
    ]


def test_serialized_object_task_holds_postgres_lease(monkeypatch, worker_loop):
    import app.vectorstore.postgres_store as postgres_store

    calls = []

    @contextlib.contextmanager
    def lease(name):
        calls.append(("lease", name))
        yield

    monkeypatch.setattr(postgres_store, "postgres_advisory_lease", lease)
    monkeypatch.setattr(
        tasks,
        "_run_process_document_object",
        lambda object_key, tenant_id: calls.append(("process", object_key, tenant_id))
        or {"status": "completed"},
    )

    object_key = (
        "tenants/00000000-0000-0000-0000-000000000001/"
        "uploads/4dff36c5-2ff9-4d4b-8913-666247967020/payload.txt"
    )
    result = tasks._run_process_document_object_serialized(
        object_key,
        "00000000-0000-0000-0000-000000000001",
    )

    assert result == {"status": "completed"}
    assert calls[0][0] == "lease"
    assert calls[1] == (
        "process",
        object_key,
        "00000000-0000-0000-0000-000000000001",
    )


def test_serialized_object_task_retries_when_lease_is_held(monkeypatch, worker_loop):
    import app.vectorstore.postgres_store as postgres_store

    @contextlib.contextmanager
    def unavailable_lease(_name):
        raise postgres_store.PostgresAdvisoryLeaseUnavailable("held")
        yield

    monkeypatch.setattr(postgres_store, "postgres_advisory_lease", unavailable_lease)

    with pytest.raises(tasks.DuplicateIngestionInProgress):
        tasks._run_process_document_object_serialized(
            "tenants/00000000-0000-0000-0000-000000000001/"
            "uploads/4dff36c5-2ff9-4d4b-8913-666247967020/payload.txt",
            "00000000-0000-0000-0000-000000000001",
        )
