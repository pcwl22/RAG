"""Celery tasks for app service workflows."""
import asyncio
from typing import Any

from app.workers.celery_app import celery_app


def _run_process_document(
    file_path: str,
    filename: str,
    partition: str = "general",
    metadata: dict | None = None,
) -> dict[str, Any]:
    from app.service.ingest_service import process_document

    return asyncio.run(process_document(file_path, filename, partition, metadata))


if celery_app is not None:

    @celery_app.task(name="app.workers.tasks.process_document_task")
    def process_document_task(
        file_path: str,
        filename: str,
        partition: str = "general",
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        return _run_process_document(file_path, filename, partition, metadata)

else:

    class _MissingCeleryTask:
        def delay(self, *args: Any, **kwargs: Any) -> Any:
            raise ImportError("celery is required to use queue.provider=celery")

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return _run_process_document(*args, **kwargs)

    process_document_task = _MissingCeleryTask()


__all__ = ["process_document_task"]
