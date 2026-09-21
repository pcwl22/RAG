"""Celery tasks for app service workflows."""
import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from app.auth import normalize_tenant_id
from app.storage.object_store import (
    build_upload_manifest_key,
    get_object_store,
    sanitize_upload_filename,
    upload_filename_extension,
    validate_upload_object_key,
)
from app.storage.upload_receipt import (
    build_completion_receipt,
    parse_completion_receipt,
)
from app.utils.config import get_settings, validate_runtime_config
from app.utils.logger import get_logger
from app.utils.upload_files import quarantine_failed_upload
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_runtime_ready = False
MAX_OBJECT_DOWNLOAD_RETRIES = 3
MAX_TRANSIENT_INGEST_RETRIES = 3


class ObjectStorageDownloadError(RuntimeError):
    """A transient object download failure that is safe for Celery to retry."""


class ObjectStorageManifestError(ValueError):
    """A permanent upload-envelope error that must be retained for inspection."""


class ObjectStorageFinalizeError(RuntimeError):
    """A transient receipt/finalization failure safe for Celery retry."""


class TransientIngestionError(RuntimeError):
    """A transient database, cache or inference failure safe for bounded retry."""


class DuplicateIngestionInProgress(TransientIngestionError):
    """Another worker owns the same durable upload intent."""


def _ensure_worker_runtime() -> asyncio.AbstractEventLoop:
    """Initialize async resources once per Celery worker process.

    Celery tasks are synchronous functions, while the ingest pipeline and both
    storage clients are async. A persistent loop is required so the Redis
    connection is not created on a loop that is immediately closed by
    ``asyncio.run``.
    """
    global _worker_loop, _worker_runtime_ready
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
        _worker_runtime_ready = False

    if not _worker_runtime_ready:
        from app.utils.cache import init_redis
        from app.vectorstore.storage_adapter import init_vector_store

        runtime_config = get_settings()
        validate_runtime_config(runtime_config)

        async def initialize() -> None:
            await init_vector_store()
            if not await init_redis():
                raise RuntimeError("Redis is required for Celery worker task state and cache invalidation")

        try:
            _worker_loop.run_until_complete(initialize())
        except Exception as exc:
            logger.error("Celery worker runtime initialization failed", exc_info=True)
            raise TransientIngestionError(
                "Celery worker runtime dependency is unavailable"
            ) from exc
        _worker_runtime_ready = True
    return _worker_loop


def _shutdown_worker_runtime() -> None:
    """Close worker-owned async resources before the process exits."""
    global _worker_loop, _worker_runtime_ready
    if _worker_loop is None or _worker_loop.is_closed():
        return

    from app.utils.cache import close_redis
    from app.vectorstore.storage_adapter import close_vector_store

    async def close() -> None:
        await close_redis()
        await close_vector_store()

    try:
        _worker_loop.run_until_complete(close())
    except Exception:
        logger.warning("Celery worker runtime shutdown failed", exc_info=True)
    finally:
        _worker_loop.close()
        _worker_loop = None
        _worker_runtime_ready = False


def _run_process_document(
    file_path: str,
    filename: str,
    partition: str = "general",
    metadata: dict | None = None,
    tenant_id: str | None = None,
    *,
    upload_task_id: str | None = None,
    upload_object_key: str | None = None,
) -> dict[str, Any]:
    from app.service.ingest_service import is_transient_ingest_error, process_document

    try:
        loop = _ensure_worker_runtime()
        result = cast(
            dict[str, Any],
            loop.run_until_complete(
                process_document(
                    file_path,
                    filename,
                    partition,
                    metadata,
                    tenant_id=normalize_tenant_id(tenant_id),
                    internal_metadata=(
                        {
                            "upload_task_id": upload_task_id,
                            "upload_object_key": upload_object_key,
                        }
                        if upload_task_id and upload_object_key
                        else None
                    ),
                )
            ),
        )
    except Exception as exc:
        logger.error("Document ingestion raised an exception", exc_info=True)
        if isinstance(exc, TransientIngestionError):
            raise
        if is_transient_ingest_error(exc):
            raise TransientIngestionError("Transient document ingestion failure") from exc
        _retain_failed_upload(file_path)
        raise RuntimeError("Document ingestion failed") from exc
    if result.get("status") != "completed":
        if result.get("retryable") is True:
            raise TransientIngestionError("Transient document ingestion failure")
        # Celery only records FAILURE when the task raises. Preserve the source
        # file so an operator or an automatic retry can process it again.
        _retain_failed_upload(file_path)
        logger.error("Document ingestion failed: %s", result.get("error") or "unknown error")
        raise RuntimeError("Document ingestion failed")

    try:
        return result
    finally:
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass


def _completed_result_from_manifest(
    manifest: dict[str, Any],
    object_key: str,
    tenant_id: str,
    task_id: str,
) -> dict[str, Any] | None:
    """Validate and return an existing durable completion receipt."""
    try:
        return cast(
            dict[str, Any] | None,
            parse_completion_receipt(
                manifest,
                expected_task_id=task_id,
                expected_tenant_id=tenant_id,
                expected_object_key=object_key,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ObjectStorageManifestError("completion receipt is invalid") from exc


def _persist_completion_receipt(
    object_store: Any,
    manifest_key: str,
    manifest: dict[str, Any],
    object_key: str,
    result: dict[str, Any],
) -> None:
    """Persist the commit result before deleting the upload envelope."""
    payload = dict(manifest)
    key_parts = object_key.strip("/").split("/")
    task_id = key_parts[3]
    tenant_id = normalize_tenant_id(payload.get("tenant_id"))
    # Upgrade a legacy envelope before hashing it so every new receipt has the
    # same task/tenant/object binding as current manifests.
    payload["version"] = 1
    payload["task_id"] = task_id
    payload["tenant_id"] = tenant_id
    payload["object_key"] = object_key
    payload["completion_receipt"] = build_completion_receipt(
        payload,
        task_id=task_id,
        tenant_id=tenant_id,
        object_key=object_key,
        result=result,
    )
    payload["completed_at_unix"] = int(time.time())
    try:
        object_store.put_json(manifest_key, payload)
    except Exception as exc:
        raise ObjectStorageFinalizeError("completion receipt persistence failed") from exc


def _cleanup_completed_upload(object_store: Any, object_key: str) -> None:
    """Delete the payload while retaining its bounded idempotency receipt."""
    try:
        object_store.delete_object(object_key)
    except Exception as exc:
        raise ObjectStorageFinalizeError("completed upload cleanup failed") from exc


def _run_process_document_object(
    object_key: str,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Download one tenant-scoped object, process it, then finalize it.

    A download/transport failure leaves the source object in the upload prefix
    so a Celery retry can consume it. Once processing starts and fails, the
    object is moved to the tenant's failure prefix for investigation/retry.
    """
    validate_upload_object_key(object_key)
    _ensure_worker_runtime()
    object_store = get_object_store()
    manifest_key = build_upload_manifest_key(object_key)
    resolved_tenant = normalize_tenant_id(tenant_id)
    manifest_loaded = False
    downloaded = False
    processed = False

    try:
        manifest = object_store.get_json(manifest_key)
    except Exception as exc:
        raise ObjectStorageDownloadError("upload manifest download failed") from exc
    manifest_loaded = True
    try:
        manifest_tenant = normalize_tenant_id(manifest.get("tenant_id"))
        key_tenant = object_key.strip("/").split("/")[1]
        key_task_id = object_key.strip("/").split("/")[3]
        manifest_version = manifest.get("version")
        if manifest_version is not None:
            if manifest_version != 1:
                raise ValueError("upload manifest version is invalid")
            if manifest.get("task_id") != key_task_id:
                raise ValueError("upload manifest task id does not match the task")
            if manifest.get("object_key") != object_key:
                raise ValueError("upload manifest object key does not match the task")
        filename = manifest.get("filename")
        partition = manifest.get("partition")
        metadata = manifest.get("metadata")
        basename = sanitize_upload_filename(filename if isinstance(filename, str) else None)
        storage_basename = object_key.rsplit("/", 1)[-1]
        if manifest_tenant != resolved_tenant or manifest_tenant != key_tenant:
            raise ValueError("upload manifest tenant does not match the task")
        if not isinstance(filename, str) or filename != basename:
            raise ValueError("upload manifest filename is not service-generated")
        if upload_filename_extension(storage_basename) != upload_filename_extension(basename):
            raise ValueError("upload object and display filename extensions do not match")
        if not isinstance(partition, str) or not partition.strip() or len(partition) > 100:
            raise ValueError("upload manifest partition is invalid")
        if not isinstance(metadata, dict):
            raise ValueError("upload manifest metadata must be an object")
    except (TypeError, ValueError, KeyError) as exc:
        try:
            object_store.move_to_failed(object_key)
        except Exception:
            logger.warning("Invalid upload could not be retained", exc_info=True)
        try:
            object_store.move_to_failed(manifest_key)
        except Exception:
            logger.warning("Invalid upload manifest could not be retained", exc_info=True)
        raise ObjectStorageManifestError("upload manifest is invalid") from exc

    try:
        completed_result = _completed_result_from_manifest(
            manifest,
            object_key,
            resolved_tenant,
            key_task_id,
        )
        if completed_result is not None:
            from app.vectorstore.postgres_store import document_commit_matches

            try:
                committed = document_commit_matches(
                    tenant_id=resolved_tenant,
                    document_id=completed_result["document_id"],
                    source_key=completed_result["source_key"],
                    upload_task_id=key_task_id,
                    upload_object_key=object_key,
                    total_chunks=completed_result["total_chunks"],
                )
            except Exception as exc:
                raise ObjectStorageFinalizeError(
                    "completion receipt database verification failed"
                ) from exc
            if not committed:
                raise ObjectStorageManifestError(
                    "completion receipt has no matching database commit"
                )
    except ObjectStorageManifestError:
        for source_key in (object_key, manifest_key):
            try:
                object_store.move_to_failed(source_key)
            except Exception:
                logger.warning("Invalid completion receipt could not be retained", exc_info=True)
        raise
    if completed_result is not None:
        _cleanup_completed_upload(object_store, object_key)
        return completed_result

    with tempfile.TemporaryDirectory(prefix="rag-upload-") as temporary_dir:
        # The local storage path remains opaque/ASCII while the parser and
        # source metadata receive the Unicode display name from the manifest.
        local_path = Path(temporary_dir) / storage_basename
        try:
            try:
                object_store.download_file(object_key, local_path)
            except Exception as exc:
                # Keep this phase distinguishable from parsing/embedding
                # failures: the source object remains in the upload prefix and
                # Celery may safely retry the download.
                raise ObjectStorageDownloadError("object download failed") from exc
            downloaded = True
            result = _run_process_document(
                str(local_path),
                basename,
                partition,
                metadata,
                tenant_id,
                upload_task_id=key_task_id,
                upload_object_key=object_key,
            )
            processed = True
            _persist_completion_receipt(
                object_store,
                manifest_key,
                manifest,
                object_key,
                result,
            )
            _cleanup_completed_upload(object_store, object_key)
            return result
        except (ObjectStorageFinalizeError, TransientIngestionError):
            # A committed replacement is idempotent by source_key.  Keep the
            # source/manifest in place so either the receipt cleanup or the
            # bounded ingest retry can be replayed without data duplication.
            raise
        except Exception:
            if downloaded and not processed and manifest_loaded:
                for source_key in (object_key, manifest_key):
                    try:
                        failed_key = object_store.move_to_failed(source_key)
                        logger.warning("Failed upload moved to retention prefix: %s", failed_key)
                    except Exception:
                        logger.warning(
                            "Failed upload object could not be retained",
                            exc_info=True,
                        )
            raise


def _run_process_document_object_serialized(
    object_key: str,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Serialize at-least-once delivery with a crash-safe PostgreSQL lease."""
    resolved_tenant = normalize_tenant_id(tenant_id)
    _ensure_worker_runtime()
    from app.vectorstore.postgres_store import (
        PostgresAdvisoryLeaseUnavailable,
        postgres_advisory_lease,
    )

    lease_name = f"document-ingest:{resolved_tenant}:{object_key}"
    try:
        with postgres_advisory_lease(lease_name):
            return _run_process_document_object(object_key, resolved_tenant)
    except PostgresAdvisoryLeaseUnavailable as exc:
        raise DuplicateIngestionInProgress(
            "Another worker is processing this upload"
        ) from exc


def _retain_failed_upload(file_path: str) -> None:
    processing = get_settings().get("document_processing", {})
    try:
        quarantine_failed_upload(
            file_path,
            retention_seconds=int(processing.get("failed_upload_retention_seconds", 604800)),
            max_files=int(processing.get("failed_upload_max_files", 100)),
        )
    except OSError:
        logger.warning("Failed upload could not be quarantined: %s", file_path, exc_info=True)


def _publish_celery_progress(task: Any, progress: int, total_chunks: int = 0) -> None:
    """Publish bounded, non-sensitive progress metadata without masking work errors."""
    try:
        task.update_state(
            state="PROGRESS",
            meta={"progress": max(0, min(int(progress), 100)), "total_chunks": int(total_chunks)},
        )
    except Exception:
        # A result-backend outage must not turn a successfully ingested document
        # into a failed task.
        logger.warning("Celery task progress update failed", exc_info=True)


if celery_app is not None:

    from celery.signals import worker_process_shutdown, worker_shutdown

    @worker_process_shutdown.connect
    def _on_worker_process_shutdown(**_kwargs: Any) -> None:
        _shutdown_worker_runtime()

    @worker_shutdown.connect
    def _on_worker_shutdown(**_kwargs: Any) -> None:
        _shutdown_worker_runtime()

    @celery_app.task(bind=True, name="app.workers.tasks.process_document_task")
    def process_document_task(
        self: Any,
        object_key: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        _publish_celery_progress(self, 5)
        try:
            result = _run_process_document_object_serialized(
                object_key,
                tenant_id,
            )
        except (
            ObjectStorageDownloadError,
            ObjectStorageFinalizeError,
            TransientIngestionError,
        ) as exc:
            retries = int(getattr(getattr(self, "request", None), "retries", 0))
            if isinstance(exc, DuplicateIngestionInProgress):
                retry_limit = 20
                countdown = 30
            else:
                retry_limit = (
                    MAX_TRANSIENT_INGEST_RETRIES
                    if isinstance(exc, TransientIngestionError)
                    else MAX_OBJECT_DOWNLOAD_RETRIES
                )
                countdown = 2**retries
            if retries < retry_limit:
                raise self.retry(exc=exc, countdown=countdown) from exc
            raise
        _publish_celery_progress(self, 95, result.get("total_chunks", 0))
        return result

else:

    class _MissingCeleryTask:
        def delay(self, *args: Any, **kwargs: Any) -> Any:
            raise ImportError("celery is required to use queue.provider=celery")

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return _run_process_document(*args, **kwargs)

    process_document_task = _MissingCeleryTask()


__all__ = ["process_document_task"]
