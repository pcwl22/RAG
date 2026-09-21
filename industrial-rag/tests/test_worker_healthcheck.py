"""Celery container probe tests."""

import importlib

import app.workers.healthcheck as healthcheck


def test_worker_ping_targets_the_current_node(monkeypatch):
    calls = []

    class Control:
        @staticmethod
        def ping(**kwargs):
            calls.append(kwargs)
            return [{"celery@worker-1": {"ok": "pong"}}]

    celery_module = importlib.import_module("app.workers.celery_app")
    monkeypatch.setattr(
        celery_module,
        "celery_app",
        type("App", (), {"control": Control()})(),
    )
    monkeypatch.setenv("CELERY_WORKER_NODENAME", "celery@worker-1")

    assert healthcheck.check_worker_ping() is True
    assert calls == [{"destination": ["celery@worker-1"], "timeout": 5.0}]


def test_worker_readiness_checks_worker_and_required_dependencies(monkeypatch):
    calls = []
    config = {"queue": {"provider": "celery"}}

    class Store:
        def check_health(self):
            calls.append("s3")
            return True

    monkeypatch.setattr(healthcheck, "get_settings", lambda: config)
    monkeypatch.setattr(
        healthcheck,
        "validate_runtime_config",
        lambda value: calls.append(("validate", value)),
    )
    monkeypatch.setattr(healthcheck, "resolve_queue_provider", lambda _value: "celery")
    monkeypatch.setattr(
        healthcheck,
        "check_worker_ping",
        lambda: calls.append("worker") or True,
    )
    monkeypatch.setattr(
        healthcheck,
        "check_postgres",
        lambda value: calls.append(("postgres", value)) or True,
    )
    monkeypatch.setattr(
        healthcheck,
        "check_redis",
        lambda value: calls.append(("redis", value)) or True,
    )
    monkeypatch.setattr(healthcheck, "get_object_store", lambda _value: Store())

    assert healthcheck.check_readiness() is True
    assert calls == [
        ("validate", config),
        "worker",
        ("postgres", config),
        ("redis", config),
        "s3",
    ]


def test_healthcheck_main_fails_closed_on_probe_exception(monkeypatch):
    def fail():
        raise TimeoutError("broker unavailable")

    monkeypatch.setattr(healthcheck, "check_readiness", fail)

    assert healthcheck.main(["readiness"]) == 1
