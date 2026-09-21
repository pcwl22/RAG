"""One unambiguous dotenv contract for protected release snapshots."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import MutableMapping
from pathlib import Path
from uuid import UUID

_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PROTECTED_PREFIXES = (
    "CANARY_",
    "COMPOSE_REDIS_",
    "DEEPSEEK_",
    "INGRESS_",
    "MODEL_",
    "OIDC_",
    "POSTGRES_",
    "RAGAS_",
    "RAG_",
    "REDIS_",
    "S3_",
    "VITE_",
)
_PROTECTED_EXACT = {
    "API_IMAGE",
    "EMBEDDING_DEVICE",
    "FRONTEND_IMAGE",
    "MULTIMODAL_DEVICE",
    "OBJECT_STORAGE_ENABLED",
    "PDF_DEVICE",
    "QUEUE_PROVIDER",
    "RELEASE_ID",
    "RERANKER_DEVICE",
    "WORKER_IMAGE",
}
_PRESERVED_CONTROL_KEYS = {"RAG_ENV_FILE", "RAG_RELEASE_SNAPSHOT"}


def normalize_dotenv_key(raw_key: str, *, strict: bool) -> str:
    key = raw_key.strip()
    export_match = re.match(r"^export\s+", key)
    if export_match:
        key = key[export_match.end() :].strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1].strip()
    if strict and not _KEY_PATTERN.fullmatch(key):
        raise ValueError(f"invalid dotenv key {key!r}")
    return key


def load_env_file(path: Path, *, reject_duplicates: bool = False) -> dict[str, str]:
    """Load the supported dotenv subset; strict mode rejects ambiguous syntax."""
    values: dict[str, str] = {}
    source_lines: dict[str, int] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            if reject_duplicates:
                raise ValueError(f"invalid dotenv assignment at line {line_number}")
            continue
        raw_key, raw_value = stripped.split("=", 1)
        key = normalize_dotenv_key(raw_key, strict=reject_duplicates)
        value = raw_value.strip()
        starts_quoted = bool(value) and value[0] in {"'", '"'}
        ends_quoted = bool(value) and value[-1] in {"'", '"'}
        if starts_quoted or ends_quoted:
            if not (len(value) >= 2 and value[0] == value[-1]):
                if reject_duplicates:
                    raise ValueError(f"unbalanced dotenv quotes at line {line_number}")
            else:
                value = value[1:-1]
        elif reject_duplicates and re.search(r"\s+#", value):
            raise ValueError(
                f"inline dotenv comments are not allowed in release snapshots (line {line_number})"
            )
        if reject_duplicates and key in values:
            first_line = source_lines[key]
            raise ValueError(
                f"duplicate dotenv key {key!r} at lines {first_line} and {line_number}"
            )
        values[key] = value
        source_lines[key] = line_number
    return values


def load_release_env_file(path: Path) -> dict[str, str]:
    """Load one authoritative protected release snapshot."""
    return load_env_file(path, reject_duplicates=True)


def release_config_digest(values: dict[str, str]) -> str:
    """Hash only the public, unique release ID, never the secret snapshot."""
    release_id = values.get("RELEASE_ID", "").strip()
    try:
        parsed = UUID(release_id)
    except ValueError as exc:
        raise ValueError("RELEASE_ID must be a canonical lowercase UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != release_id:
        raise ValueError("RELEASE_ID must be a canonical lowercase UUIDv4")
    material = f"industrial-rag-release-id-v1:{release_id}".encode()
    return "sha256:" + hashlib.sha256(material).hexdigest()


def release_resource_suffix(values: dict[str, str]) -> str:
    """Return the DNS-safe suffix shared by versioned release resources."""
    return release_config_digest(values).removeprefix("sha256:")[:20]


def apply_release_env_file(
    path: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Replace protected process settings with the exact strict snapshot.

    The Secret Manager snapshot carries both migration-owner and runtime
    PostgreSQL credentials.  Application/evaluation processes must mirror the
    Kubernetes runtime mapping and never connect as the migration owner.
    """
    target = os.environ if environ is None else environ
    values = load_release_env_file(path)
    for key in list(target):
        if key in _PRESERVED_CONTROL_KEYS:
            continue
        if key in _PROTECTED_EXACT or key.startswith(_PROTECTED_PREFIXES):
            target.pop(key, None)
    target.update(values)

    runtime_user = values.get("POSTGRES_RUNTIME_USER", "").strip()
    runtime_password = values.get("POSTGRES_APP_PASSWORD", "")
    if runtime_user:
        target["POSTGRES_USER"] = runtime_user
    if runtime_password:
        target["POSTGRES_PASSWORD"] = runtime_password
    return values
