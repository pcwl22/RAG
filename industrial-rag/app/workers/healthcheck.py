"""Executable health checks for the Celery worker container."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Any

from app.storage.object_store import get_object_store
from app.utils.config import get_settings, resolve_queue_provider, validate_runtime_config


def _timeout_seconds() -> int:
    raw = os.getenv("WORKER_HEALTHCHECK_TIMEOUT_SECONDS", "5")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("WORKER_HEALTHCHECK_TIMEOUT_SECONDS must be an integer") from exc
    if value < 1 or value > 30:
        raise ValueError("WORKER_HEALTHCHECK_TIMEOUT_SECONDS must be between 1 and 30")
    return value


def _worker_node_name() -> str:
    return os.getenv("CELERY_WORKER_NODENAME", f"celery@{socket.gethostname()}").strip()


def check_worker_process() -> bool:
    """Verify that PID 1 is the worker process without depending on the broker."""
    if os.name != "posix":
        return check_worker_ping()
    try:
        command = Path("/proc/1/cmdline").read_bytes().replace(b"\0", b" ").lower()
    except OSError:
        return False
    return b"celery" in command and b"worker" in command


def check_worker_ping() -> bool:
    """Ask the exact worker node to respond through the configured broker."""
    from app.workers.celery_app import celery_app

    if celery_app is None:
        return False
    node_name = _worker_node_name()
    replies = celery_app.control.ping(
        destination=[node_name],
        timeout=float(_timeout_seconds()),
    )
    if not isinstance(replies, list):
        return False
    return any(
        isinstance(reply, dict)
        and isinstance(reply.get(node_name), dict)
        and reply[node_name].get("ok") == "pong"
        for reply in replies
    )


def check_postgres(config: dict[str, Any]) -> bool:
    """Open a bounded connection and verify the migrated schema exists."""
    import psycopg2

    postgres = config.get("postgres", {})
    kwargs: dict[str, Any] = {
        "host": postgres.get("host", "localhost"),
        "port": int(postgres.get("port", 5432)),
        "database": postgres.get("database", "rag_db"),
        "user": postgres.get("user", "postgres"),
        "password": postgres.get("password", ""),
        "connect_timeout": min(
            _timeout_seconds(),
            int(postgres.get("connect_timeout", 5)),
        ),
    }
    sslmode = str(postgres.get("sslmode", "")).strip()
    sslrootcert = str(postgres.get("sslrootcert", "")).strip()
    if sslmode and not sslmode.startswith("${"):
        kwargs["sslmode"] = sslmode
    if sslrootcert and not sslrootcert.startswith("${"):
        kwargs["sslrootcert"] = sslrootcert
    connection = psycopg2.connect(**kwargs)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('public.documents') IS NOT NULL "
                "AND to_regclass('public.rag_schema_metadata') IS NOT NULL"
            )
            row = cursor.fetchone()
            return bool(row and row[0])
    finally:
        connection.close()


def check_redis(config: dict[str, Any]) -> bool:
    """Verify the worker's broker/task-state Redis transport."""
    import redis

    redis_config = config.get("redis", {})
    url = os.getenv("REDIS_URL") or redis_config.get("url")
    timeout = min(
        float(_timeout_seconds()),
        float(redis_config.get("socket_timeout_seconds", 2)),
    )
    client = redis.Redis.from_url(
        str(url),
        socket_connect_timeout=timeout,
        socket_timeout=timeout,
    )
    try:
        return bool(client.ping())
    finally:
        client.close()


def check_readiness() -> bool:
    """Verify the worker itself plus every dependency required to ingest."""
    config = get_settings()
    validate_runtime_config(config)
    if resolve_queue_provider(config) != "celery":
        return False
    if not check_worker_ping():
        return False
    if not check_postgres(config) or not check_redis(config):
        return False
    return bool(get_object_store(config).check_health())


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    mode = args[0] if args else "readiness"
    try:
        if mode == "liveness":
            healthy = check_worker_process()
        elif mode == "readiness":
            healthy = check_readiness()
        else:
            raise ValueError("healthcheck mode must be readiness or liveness")
    except Exception:
        healthy = False
    return 0 if healthy else 1


if __name__ == "__main__":  # pragma: no cover - exercised by container probes
    raise SystemExit(main())
