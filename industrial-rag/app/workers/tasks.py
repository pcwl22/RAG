"""Celery tasks for app service workflows."""
import asyncio
import os
from typing import Any

from app.auth import normalize_tenant_id
from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.utils.upload_files import quarantine_failed_upload
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_runtime_ready = False


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

        async def initialize() -> None:
            await init_vector_store()
            if not await init_redis():
                raise RuntimeError("Redis is required for Celery worker task state and cache invalidation")

        try:
            _worker_loop.run_until_complete(initialize())
        except Exception:
            logger.error("Celery worker runtime initialization failed", exc_info=True)
            raise
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
) -> dict[str, Any]:
    from app.service.ingest_service import process_document

    try:
        loop = _ensure_worker_runtime()
        result = loop.run_until_complete(
            process_document(
                file_path,
                filename,
                partition,
                metadata,
                tenant_id=normalize_tenant_id(tenant_id),
            )
        )
    except Exception as exc:
        logger.error("Document ingestion raised an exception", exc_info=True)
        _retain_failed_upload(file_path)
        raise RuntimeError("Document ingestion failed") from exc
    if result.get("status") != "completed":
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


if celery_app is not None:

    from celery.signals import worker_process_shutdown, worker_shutdown

    @worker_process_shutdown.connect
    def _on_worker_process_shutdown(**_kwargs: Any) -> None:
        _shutdown_worker_runtime()

    @worker_shutdown.connect
    def _on_worker_shutdown(**_kwargs: Any) -> None:
        _shutdown_worker_runtime()

    @celery_app.task(name="app.workers.tasks.process_document_task")
    def process_document_task(
        file_path: str,
        filename: str,
        partition: str = "general",
        metadata: dict | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        return _run_process_document(file_path, filename, partition, metadata, tenant_id)

else:

    class _MissingCeleryTask:
        def delay(self, *args: Any, **kwargs: Any) -> Any:
            raise ImportError("celery is required to use queue.provider=celery")

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return _run_process_document(*args, **kwargs)

    process_document_task = _MissingCeleryTask()


__all__ = ["process_document_task"]
