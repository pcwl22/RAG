"""Celery application for background document processing."""
import os
from typing import Any

try:
    from celery import Celery
except ImportError:
    Celery = None


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

    app = Celery("industrial_rag", broker=broker_url, backend=result_backend)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Asia/Shanghai",
        enable_utc=True,
        task_track_started=True,
    )
    app.autodiscover_tasks(["app.workers"])
    return app


celery_app = create_celery_app()

__all__ = ["celery_app", "create_celery_app"]
