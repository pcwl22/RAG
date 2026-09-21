"""Durable Celery upload-intent reconciliation.

S3 manifests are the outbox. Redis supplies a short dispatch cooldown across
API replicas, while Celery keeps the same deterministic task id on every
delivery. The worker's PostgreSQL advisory lease serializes duplicate delivery.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any

from app.auth import normalize_tenant_id
from app.storage.object_store import (
    build_upload_manifest_key,
    get_object_store,
    validate_upload_object_key,
)
from app.storage.upload_receipt import parse_completion_receipt
from app.utils.cache import (
    acquire_task_dispatch_lease,
    get_task_state,
    set_task_state,
)
from app.utils.config import TRUE_VALUES, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

UPLOAD_MANIFEST_VERSION = 1
_ACTIVE_CELERY_STATES = frozenset({"STARTED", "PROGRESS", "RETRY", "SUCCESS", "REVOKED"})
_manifest_scan_token: str | None = None


def _positive_int(
    values: dict[str, Any],
    key: str,
    env_name: str,
    default: int,
    *,
    maximum: int,
) -> int:
    raw = os.getenv(env_name) or values.get(key, default)
    try:
        result = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be a positive integer") from exc
    if result < 1 or result > maximum:
        raise ValueError(f"{env_name} must be between 1 and {maximum}")
    return result


def reconciliation_settings(config: dict[str, Any]) -> dict[str, int | bool]:
    queue = config.get("queue", {})
    values = queue.get("reconciliation", {}) if isinstance(queue, dict) else {}
    if not isinstance(values, dict):
        raise ValueError("queue.reconciliation must be a mapping")
    enabled_raw = os.getenv("UPLOAD_RECONCILIATION_ENABLED")
    enabled = (
        str(enabled_raw).strip().lower() in TRUE_VALUES
        if enabled_raw is not None
        else bool(values.get("enabled", True))
    )
    return {
        "enabled": enabled,
        "interval_seconds": _positive_int(
            values,
            "interval_seconds",
            "UPLOAD_RECONCILIATION_INTERVAL_SECONDS",
            30,
            maximum=3600,
        ),
        "redispatch_after_seconds": _positive_int(
            values,
            "redispatch_after_seconds",
            "UPLOAD_REDISPATCH_AFTER_SECONDS",
            90,
            maximum=86400,
        ),
        "batch_size": _positive_int(
            values,
            "batch_size",
            "UPLOAD_RECONCILIATION_BATCH_SIZE",
            100,
            maximum=1000,
        ),
        "dispatch_lease_seconds": _positive_int(
            values,
            "dispatch_lease_seconds",
            "UPLOAD_DISPATCH_LEASE_SECONDS",
            60,
            maximum=3600,
        ),
        "max_dispatch_attempts": _positive_int(
            values,
            "max_dispatch_attempts",
            "UPLOAD_MAX_DISPATCH_ATTEMPTS",
            10,
            maximum=100,
        ),
        "completion_receipt_retention_seconds": _positive_int(
            values,
            "completion_receipt_retention_seconds",
            "UPLOAD_COMPLETION_RECEIPT_RETENTION_SECONDS",
            86400,
            maximum=2592000,
        ),
        "producer_timeout_seconds": _positive_int(
            values,
            "producer_timeout_seconds",
            "UPLOAD_PRODUCER_TIMEOUT_SECONDS",
            10,
            maximum=60,
        ),
    }


def _pending_manifest_identity(
    manifest_key: str,
    manifest: dict[str, Any],
) -> tuple[str, str, str]:
    if manifest.get("version") != UPLOAD_MANIFEST_VERSION:
        raise ValueError("upload manifest version is unsupported")
    object_key = manifest.get("object_key")
    task_id = manifest.get("task_id")
    tenant_id = normalize_tenant_id(manifest.get("tenant_id"))
    if not isinstance(object_key, str):
        raise ValueError("upload manifest object_key is invalid")
    validate_upload_object_key(object_key)
    if build_upload_manifest_key(object_key) != manifest_key:
        raise ValueError("upload manifest key does not match object_key")
    key_parts = object_key.strip("/").split("/")
    if not isinstance(task_id, str) or task_id != key_parts[3]:
        raise ValueError("upload manifest task_id is invalid")
    try:
        uuid.UUID(task_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("upload manifest task_id is invalid") from exc
    if tenant_id != key_parts[1]:
        raise ValueError("upload manifest tenant does not match object_key")
    return task_id, tenant_id, object_key


def _manifest_timestamp(manifest: dict[str, Any]) -> int:
    raw = manifest.get("last_dispatched_at_unix") or manifest.get("created_at_unix")
    if raw is None or isinstance(raw, bool):
        raise ValueError("upload manifest timestamp is invalid")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("upload manifest timestamp is invalid") from exc
    if value < 1:
        raise ValueError("upload manifest timestamp is invalid")
    return value


async def _celery_state(celery_app: Any, task_id: str, timeout: int) -> str:
    def read_state() -> str:
        return str(celery_app.AsyncResult(task_id).state).upper()

    return await asyncio.wait_for(asyncio.to_thread(read_state), timeout=timeout)


async def reconcile_pending_uploads(
    config: dict[str, Any] | None = None,
    *,
    now_unix: int | None = None,
) -> dict[str, int]:
    """Replay stale S3 upload intents with bounded, cross-replica dispatch."""
    runtime_config = config or get_settings()
    settings = reconciliation_settings(runtime_config)
    stats = {"scanned": 0, "dispatched": 0, "skipped": 0, "errors": 0, "cleaned": 0}
    if not settings["enabled"]:
        return stats

    from app.workers.celery_app import celery_app, task_default_queue
    from app.workers.tasks import process_document_task

    if celery_app is None:
        raise RuntimeError("Celery is required for upload reconciliation")

    global _manifest_scan_token
    store = get_object_store(runtime_config)
    manifest_keys, _manifest_scan_token = await asyncio.to_thread(
        store.list_upload_manifest_keys,
        int(settings["batch_size"]),
        _manifest_scan_token,
    )
    now = int(now_unix if now_unix is not None else time.time())

    for manifest_key in manifest_keys:
        stats["scanned"] += 1
        try:
            manifest = await asyncio.to_thread(store.get_json, manifest_key)
            task_id, tenant_id, object_key = _pending_manifest_identity(
                manifest_key,
                manifest,
            )

            receipt_result = parse_completion_receipt(
                manifest,
                expected_task_id=task_id,
                expected_tenant_id=tenant_id,
                expected_object_key=object_key,
            )
            if receipt_result is not None:
                from app.vectorstore.postgres_store import document_commit_matches

                committed = await asyncio.to_thread(
                    document_commit_matches,
                    tenant_id=tenant_id,
                    document_id=receipt_result["document_id"],
                    source_key=receipt_result["source_key"],
                    upload_task_id=task_id,
                    upload_object_key=object_key,
                    total_chunks=receipt_result["total_chunks"],
                )
                if not committed:
                    raise ValueError(
                        "completion receipt has no matching database commit"
                    )
                completed_at = manifest.get("completed_at_unix")
                if (
                    isinstance(completed_at, int)
                    and not isinstance(completed_at, bool)
                    and now - completed_at
                    >= int(settings["completion_receipt_retention_seconds"])
                ):
                    await asyncio.to_thread(store.delete_object, manifest_key)
                    stats["cleaned"] += 1
                else:
                    stats["skipped"] += 1
                continue

            attempts = int(manifest.get("dispatch_attempts", 0))
            if attempts < 0 or attempts >= int(settings["max_dispatch_attempts"]):
                stats["skipped"] += 1
                continue
            if now - _manifest_timestamp(manifest) < int(
                settings["redispatch_after_seconds"]
            ):
                stats["skipped"] += 1
                continue

            state = await _celery_state(
                celery_app,
                task_id,
                int(settings["producer_timeout_seconds"]),
            )
            if state in _ACTIVE_CELERY_STATES:
                stats["skipped"] += 1
                continue

            acquired = await acquire_task_dispatch_lease(
                task_id,
                tenant_id=tenant_id,
                ttl_seconds=int(settings["dispatch_lease_seconds"]),
            )
            if not acquired:
                stats["skipped"] += 1
                continue

            known_state = await get_task_state(
                task_id,
                tenant_id=tenant_id,
                require_backend=True,
            )
            if known_state is None:
                persisted = await set_task_state(
                    task_id,
                    {
                        "status": "pending",
                        "progress": 0,
                        "total_chunks": 0,
                        "error": None,
                        "error_code": None,
                    },
                    tenant_id=tenant_id,
                )
                if not persisted:
                    raise RuntimeError("task state could not be persisted")

            await asyncio.wait_for(
                asyncio.to_thread(
                    process_document_task.apply_async,
                    kwargs={"object_key": object_key, "tenant_id": tenant_id},
                    task_id=task_id,
                    queue=task_default_queue(),
                ),
                timeout=int(settings["producer_timeout_seconds"]),
            )
            updated_manifest = dict(manifest)
            updated_manifest["dispatch_attempts"] = attempts + 1
            updated_manifest["last_dispatched_at_unix"] = now
            await asyncio.to_thread(store.put_json, manifest_key, updated_manifest)
            stats["dispatched"] += 1
        except Exception:
            stats["errors"] += 1
            logger.warning(
                "Upload intent reconciliation failed",
                extra={"manifest_key": manifest_key},
                exc_info=True,
            )
    return stats


async def upload_reconciliation_loop(config: dict[str, Any] | None = None) -> None:
    """Continuously reconcile upload intents; individual failures are retryable."""
    runtime_config = config or get_settings()
    settings = reconciliation_settings(runtime_config)
    if not settings["enabled"]:
        return
    interval = int(settings["interval_seconds"])
    while True:
        try:
            stats = await reconcile_pending_uploads(runtime_config)
            if stats["dispatched"] or stats["errors"] or stats["cleaned"]:
                logger.info("Upload reconciliation completed", extra=stats)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Upload reconciliation cycle failed", exc_info=True)
        await asyncio.sleep(interval)


__all__ = [
    "UPLOAD_MANIFEST_VERSION",
    "reconcile_pending_uploads",
    "reconciliation_settings",
    "upload_reconciliation_loop",
]
