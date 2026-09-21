"""Durable upload outbox reconciliation tests."""

import asyncio
import importlib

import app.service.upload_reconciler as reconciler
from app.storage.upload_receipt import build_completion_receipt

TENANT_ID = "00000000-0000-0000-0000-000000000001"
TASK_ID = "4dff36c5-2ff9-4d4b-8913-666247967020"
OBJECT_KEY = f"tenants/{TENANT_ID}/uploads/{TASK_ID}/payload.txt"
MANIFEST_KEY = f"tenants/{TENANT_ID}/uploads/{TASK_ID}/.upload-manifest.json"


def _config(**overrides):
    reconciliation = {
        "enabled": True,
        "interval_seconds": 30,
        "redispatch_after_seconds": 60,
        "batch_size": 10,
        "dispatch_lease_seconds": 45,
        "max_dispatch_attempts": 5,
        "completion_receipt_retention_seconds": 300,
        "producer_timeout_seconds": 2,
        **overrides,
    }
    return {"queue": {"provider": "celery", "reconciliation": reconciliation}}


def _manifest(**overrides):
    return {
        "version": 1,
        "task_id": TASK_ID,
        "object_key": OBJECT_KEY,
        "tenant_id": TENANT_ID,
        "filename": "file.txt",
        "partition": "text",
        "metadata": {},
        "created_at_unix": 100,
        "dispatch_attempts": 1,
        "last_dispatched_at_unix": 100,
        **overrides,
    }


def _completed_manifest(**overrides):
    manifest = _manifest(completed_at_unix=100, **overrides)
    manifest["completion_receipt"] = build_completion_receipt(
        manifest,
        task_id=TASK_ID,
        tenant_id=TENANT_ID,
        object_key=OBJECT_KEY,
        result={
            "document_id": "doc-1",
            "source_key": "source-1",
            "total_chunks": 2,
            "status": "completed",
        },
    )
    return manifest


def test_stale_pending_manifest_is_redispatched_with_original_identity(monkeypatch):
    calls = []

    class Store:
        def list_upload_manifest_keys(self, limit, token):
            assert limit == 10
            assert token is None
            return [MANIFEST_KEY], "next-page"

        def get_json(self, key):
            assert key == MANIFEST_KEY
            return _manifest()

        def put_json(self, key, payload):
            calls.append(("put", key, payload))

    class CeleryApp:
        @staticmethod
        def AsyncResult(task_id):
            assert task_id == TASK_ID
            return type("Result", (), {"state": "PENDING"})()

    class Task:
        @staticmethod
        def apply_async(**kwargs):
            calls.append(("dispatch", kwargs))

    celery_module = importlib.import_module("app.workers.celery_app")
    task_module = importlib.import_module("app.workers.tasks")

    async def acquire(*_args, **kwargs):
        assert kwargs == {"tenant_id": TENANT_ID, "ttl_seconds": 45}
        return True

    async def missing_state(*_args, **_kwargs):
        return None

    async def persist(*_args, **_kwargs):
        return True

    monkeypatch.setattr(reconciler, "get_object_store", lambda _config: Store())
    monkeypatch.setattr(reconciler, "_manifest_scan_token", None)
    monkeypatch.setattr(reconciler, "acquire_task_dispatch_lease", acquire)
    monkeypatch.setattr(reconciler, "get_task_state", missing_state)
    monkeypatch.setattr(reconciler, "set_task_state", persist)
    monkeypatch.setattr(celery_module, "celery_app", CeleryApp())
    monkeypatch.setattr(celery_module, "task_default_queue", lambda: "documents")
    monkeypatch.setattr(task_module, "process_document_task", Task())

    stats = asyncio.run(
        reconciler.reconcile_pending_uploads(_config(), now_unix=200)
    )

    assert stats == {
        "scanned": 1,
        "dispatched": 1,
        "skipped": 0,
        "errors": 0,
        "cleaned": 0,
    }
    dispatch = calls[0][1]
    assert dispatch["task_id"] == TASK_ID
    assert dispatch["kwargs"] == {"object_key": OBJECT_KEY, "tenant_id": TENANT_ID}
    assert dispatch["queue"] == "documents"
    assert calls[1][2]["dispatch_attempts"] == 2
    assert calls[1][2]["last_dispatched_at_unix"] == 200


def test_active_celery_task_is_not_redispatched(monkeypatch):
    class Store:
        def list_upload_manifest_keys(self, _limit, _token):
            return [MANIFEST_KEY], None

        def get_json(self, _key):
            return _manifest()

    class CeleryApp:
        @staticmethod
        def AsyncResult(_task_id):
            return type("Result", (), {"state": "STARTED"})()

    celery_module = importlib.import_module("app.workers.celery_app")

    monkeypatch.setattr(reconciler, "get_object_store", lambda _config: Store())
    monkeypatch.setattr(reconciler, "_manifest_scan_token", None)
    monkeypatch.setattr(celery_module, "celery_app", CeleryApp())

    stats = asyncio.run(
        reconciler.reconcile_pending_uploads(_config(), now_unix=200)
    )

    assert stats["skipped"] == 1
    assert stats["dispatched"] == 0


def test_expired_completion_receipt_is_cleaned_without_dispatch(monkeypatch):
    deleted = []

    class Store:
        def list_upload_manifest_keys(self, _limit, _token):
            return [MANIFEST_KEY], None

        def get_json(self, _key):
            return _completed_manifest()

        def delete_object(self, key):
            deleted.append(key)

    class CeleryApp:
        pass

    celery_module = importlib.import_module("app.workers.celery_app")

    monkeypatch.setattr(reconciler, "get_object_store", lambda _config: Store())
    monkeypatch.setattr(reconciler, "_manifest_scan_token", None)
    monkeypatch.setattr(celery_module, "celery_app", CeleryApp())
    import app.vectorstore.postgres_store as postgres_store

    monkeypatch.setattr(postgres_store, "document_commit_matches", lambda **_kwargs: True)

    stats = asyncio.run(
        reconciler.reconcile_pending_uploads(_config(), now_unix=500)
    )

    assert deleted == [MANIFEST_KEY]
    assert stats["cleaned"] == 1
    assert stats["dispatched"] == 0


def test_valid_completion_receipt_without_database_commit_is_not_cleaned(monkeypatch):
    deleted = []

    class Store:
        def list_upload_manifest_keys(self, _limit, _token):
            return [MANIFEST_KEY], None

        def get_json(self, _key):
            return _completed_manifest()

        def delete_object(self, key):
            deleted.append(key)

    celery_module = importlib.import_module("app.workers.celery_app")
    monkeypatch.setattr(reconciler, "get_object_store", lambda _config: Store())
    monkeypatch.setattr(reconciler, "_manifest_scan_token", None)
    monkeypatch.setattr(celery_module, "celery_app", object())
    import app.vectorstore.postgres_store as postgres_store

    monkeypatch.setattr(
        postgres_store,
        "document_commit_matches",
        lambda **_kwargs: False,
    )

    stats = asyncio.run(
        reconciler.reconcile_pending_uploads(_config(), now_unix=500)
    )

    assert deleted == []
    assert stats["errors"] == 1
    assert stats["cleaned"] == 0
    assert stats["dispatched"] == 0


def test_invalid_completion_receipt_is_not_deleted(monkeypatch):
    deleted = []

    class Store:
        def list_upload_manifest_keys(self, _limit, _token):
            return [MANIFEST_KEY], None

        def get_json(self, _key):
            return _manifest(
                completion_receipt={"version": 1},
                completed_at_unix=100,
            )

        def delete_object(self, key):
            deleted.append(key)

    celery_module = importlib.import_module("app.workers.celery_app")
    monkeypatch.setattr(reconciler, "get_object_store", lambda _config: Store())
    monkeypatch.setattr(reconciler, "_manifest_scan_token", None)
    monkeypatch.setattr(celery_module, "celery_app", object())

    stats = asyncio.run(
        reconciler.reconcile_pending_uploads(_config(), now_unix=500)
    )

    assert deleted == []
    assert stats["errors"] == 1
    assert stats["cleaned"] == 0
