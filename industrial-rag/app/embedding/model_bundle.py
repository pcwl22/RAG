"""Runtime verification for the immutable local model bundle."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def configured_manifest_digest(*, required: bool | None = None) -> str | None:
    """Return the normalized configured manifest digest.

    Production (``RAG_ENV=base``) deliberately fails closed when no digest is
    configured.  Local/test profiles may omit it, but an explicitly supplied
    value is always validated.
    """
    value = os.getenv("MODEL_MANIFEST_SHA256", "").strip().lower()
    if required is None:
        required = os.getenv("RAG_ENV", "laptop").strip().lower() == "base"
    if not value:
        if required:
            raise RuntimeError("MODEL_MANIFEST_SHA256 is required in the base profile")
        return None
    if not _SHA256.fullmatch(value):
        raise RuntimeError("MODEL_MANIFEST_SHA256 must be sha256 followed by 64 hex characters")
    return value


def _manifest_path(config: dict[str, Any]) -> Path:
    override = os.getenv("MODEL_MANIFEST_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    model_path = str(config.get("embedding", {}).get("model_path") or "").strip()
    if not model_path:
        raise RuntimeError("embedding.model_path is required to locate model-manifest.json")
    return (Path(model_path).expanduser().resolve().parent / "model-manifest.json").resolve()


def validate_runtime_model_manifest(config: dict[str, Any]) -> Path | None:
    """Verify that the mounted manifest is the release-approved byte sequence."""
    expected = configured_manifest_digest()
    if expected is None:
        return None

    path = _manifest_path(config)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read approved model manifest at {path}") from exc

    actual = "sha256:" + hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise RuntimeError(
            "runtime model manifest SHA-256 does not match MODEL_MANIFEST_SHA256"
        )
    return path
