"""Logging helpers."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any

structlog: Any
try:
    import structlog
except ImportError:
    structlog = None


_configured = False
_configure_lock = Lock()


def _configure_logging() -> None:
    global _configured
    if _configured:
        return
    with _configure_lock:
        if _configured:
            return
        try:
            from app.utils.config import get_settings

            cfg = get_settings().get("logging", {})
        except Exception:
            cfg = {}

        level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)
        log_format = str(cfg.get("format", "json")).lower()
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        for item in cfg.get("handlers", []):
            if item.get("type") != "file" or not item.get("filename"):
                continue
            path = Path(str(item["filename"]))
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                handlers.append(
                    RotatingFileHandler(
                        path,
                        maxBytes=int(item.get("max_bytes", 10 * 1024 * 1024)),
                        backupCount=int(item.get("backup_count", 5)),
                        encoding="utf-8",
                    )
                )
            except OSError:
                continue

        logging.basicConfig(
            format=(
                "%(message)s"
                if log_format == "json"
                else "%(asctime)s %(levelname)s [%(name)s] %(message)s"
            ),
            level=level,
            handlers=handlers,
        )

        if structlog is not None:
            renderer = (
                structlog.processors.JSONRenderer()
                if log_format == "json"
                else structlog.dev.ConsoleRenderer(colors=False)
            )
            structlog.configure(
                processors=[
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.add_logger_name,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    renderer,
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
        _configured = True


def get_logger(name: str) -> Any:
    """Return a logger configured from the active runtime profile."""
    _configure_logging()

    if structlog is None:
        return logging.getLogger(name)

    return structlog.get_logger(name)
