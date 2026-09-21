"""Celery application for background document processing."""

import os
import re
from typing import Any

try:
    from celery import Celery
except ImportError:
    Celery = None

_QUEUE_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,127})")


def _positive_timeout(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1 or value > 60:
        raise RuntimeError(f"{name} must be between 1 and 60 seconds")
    return value


def task_default_queue() -> str:
    """Return a broker-safe queue name, including isolated release-canary queues."""
    queue = os.getenv("CELERY_TASK_DEFAULT_QUEUE", "document_processing").strip()
    if not _QUEUE_NAME.fullmatch(queue):
        raise RuntimeError(
            "CELERY_TASK_DEFAULT_QUEUE must be 1-128 safe alphanumeric/._- characters"
        )
    return queue


def create_celery_app() -> Any | None:
    """Create the Celery app when celery is installed."""
    if Celery is None:
        return None

    broker_url = (
        os.getenv("CELERY_BROKER_URL")
        or os.getenv("RABBITMQ_URL")
        or os.getenv("REDIS_URL")
        or "redis://localhost:6379/0"
    )
    result_backend = os.getenv("CELERY_RESULT_BACKEND") or os.getenv("REDIS_URL") or broker_url
    default_queue = task_default_queue()
    broker_timeout = _positive_timeout("CELERY_BROKER_TIMEOUT_SECONDS", 5)

    app = Celery("industrial_rag", broker=broker_url, backend=result_backend)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Asia/Shanghai",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_soft_time_limit=1800,
        task_time_limit=1860,
        result_expires=86400,
        broker_connection_retry_on_startup=True,
        broker_connection_timeout=broker_timeout,
        broker_transport_options={
            "socket_connect_timeout": broker_timeout,
            "socket_timeout": broker_timeout,
            "visibility_timeout": 1900,
        },
        result_backend_transport_options={
            "socket_connect_timeout": broker_timeout,
            "socket_timeout": broker_timeout,
        },
        task_publish_retry=True,
        task_publish_retry_policy={
            "max_retries": 3,
            "interval_start": 0,
            "interval_step": 0.5,
            "interval_max": 2,
        },
        task_default_queue=default_queue,
    )
    app.autodiscover_tasks(["app.workers"])
    return app


celery_app = create_celery_app()

__all__ = ["celery_app", "create_celery_app", "task_default_queue"]
