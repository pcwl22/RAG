"""Pinned Hugging Face model-source validation and runtime materialization.

Production images contain only a signed JSON descriptor. Model weights are
downloaded directly from immutable Hugging Face commit revisions into the
writable runtime cache, then checked against approved deterministic tree
digests before either model loader can use them.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_TREE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MODEL_SECTIONS = {
    "bge-m3": "embedding",
    "bge-reranker-v2-m3": "reranker",
}
_IGNORED_TREE_PARTS = {".git", ".cache"}
_RUNTIME_MARKER = ".industrial-rag-model.json"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def configured_manifest_digest(*, required: bool | None = None) -> str | None:
    """Return the normalized configured descriptor digest.

    Production (``RAG_ENV=base``) fails closed when the digest is absent.
    Local/test profiles may omit it, but an explicitly supplied value is
    always validated.
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
    """Verify that the local descriptor is the release-approved byte sequence."""
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


def _load_runtime_manifest(config: dict[str, Any]) -> dict[str, Any] | None:
    path = validate_runtime_model_manifest(config)
    if path is None:
        return None
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid model source manifest at {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise RuntimeError("model source manifest schema_version must be 2")
    hub = str(payload.get("hub") or "").rstrip("/")
    if hub != "https://huggingface.co":
        raise RuntimeError("production model source hub must be https://huggingface.co")
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != set(_MODEL_SECTIONS):
        raise RuntimeError("model source manifest must contain exactly the approved models")
    return payload


def _validated_manifest_entry(
    config: dict[str, Any],
    manifest: dict[str, Any],
    model_name: str,
) -> dict[str, str]:
    section_name = _MODEL_SECTIONS[model_name]
    section = config.get(section_name, {})
    if not isinstance(section, dict):
        raise RuntimeError(f"{section_name} configuration must be a mapping")
    models = manifest["models"]
    entry = models.get(model_name)
    if not isinstance(entry, dict):
        raise RuntimeError(f"missing model source entry: {model_name}")

    model_id = str(entry.get("model_id") or "")
    revision = str(entry.get("revision") or "")
    source_url = str(entry.get("source_url") or "").rstrip("/")
    download_url = str(entry.get("download_url") or "").rstrip("/")
    relative_path = str(entry.get("path") or "")
    tree_digest = str(entry.get("sha256") or "").lower()
    hub = str(manifest["hub"]).rstrip("/")

    if model_id != str(section.get("model_name") or ""):
        raise RuntimeError(f"{model_name} model_id does not match runtime configuration")
    if revision != str(section.get("model_revision") or ""):
        raise RuntimeError(f"{model_name} revision does not match runtime configuration")
    if not _GIT_REVISION.fullmatch(revision):
        raise RuntimeError(f"{model_name} revision must be an immutable 40-character commit")
    if source_url != f"{hub}/{model_id}":
        raise RuntimeError(f"{model_name} source_url is not the approved Hugging Face repository")
    if download_url != f"{source_url}/tree/{revision}":
        raise RuntimeError(f"{model_name} download_url is not pinned to the approved revision")
    if relative_path != model_name:
        raise RuntimeError(f"{model_name} path is invalid")
    if not _TREE_SHA256.fullmatch(tree_digest):
        raise RuntimeError(f"{model_name} tree SHA-256 is invalid")
    return {
        "model_id": model_id,
        "revision": revision,
        "source_url": source_url,
        "download_url": download_url,
        "path": relative_path,
        "sha256": tree_digest,
        "hub": hub,
    }


def _local_entry(config: dict[str, Any], model_name: str) -> dict[str, str]:
    section_name = _MODEL_SECTIONS[model_name]
    section = config.get(section_name, {})
    if not isinstance(section, dict):
        raise RuntimeError(f"{section_name} configuration must be a mapping")
    model_id = str(section.get("model_name") or "").strip()
    revision = str(section.get("model_revision") or "").strip()
    hub = os.getenv("HF_ENDPOINT", "https://huggingface.co").strip().rstrip("/")
    if not model_id or not revision:
        raise RuntimeError(f"{section_name}.model_name and model_revision are required")
    if not hub.startswith("https://"):
        raise RuntimeError("HF_ENDPOINT must be an HTTPS URL")
    source_url = f"{hub}/{model_id}"
    return {
        "model_id": model_id,
        "revision": revision,
        "source_url": source_url,
        "download_url": f"{source_url}/tree/{revision}",
        "path": model_name,
        "sha256": "",
        "hub": hub,
    }


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in path.rglob("*"):
        if item.is_symlink():
            raise RuntimeError(f"downloaded model tree contains a symlink: {item}")
    files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and not _IGNORED_TREE_PARTS.intersection(item.relative_to(path).parts)
            and item.name != _RUNTIME_MARKER
        ),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(len(chunk).to_bytes(8, "big"))
                digest.update(chunk)
    return digest.hexdigest()


def _marker_matches(target: Path, entry: dict[str, str]) -> bool:
    marker_path = target / _RUNTIME_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(marker.get(key) == entry[key] for key in ("model_id", "revision", "sha256"))


def _write_marker(target: Path, entry: dict[str, str]) -> None:
    marker = {
        "schema_version": 1,
        "model_id": entry["model_id"],
        "revision": entry["revision"],
        "source_url": entry["source_url"],
        "download_url": entry["download_url"],
        "sha256": entry["sha256"],
    }
    (target / _RUNTIME_MARKER).write_text(
        json.dumps(marker, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_existing_tree(target: Path, entry: dict[str, str]) -> None:
    if not (target / "config.json").is_file():
        raise RuntimeError(f"model cache is incomplete: {target}/config.json is missing")
    expected = entry["sha256"]
    if expected:
        actual = _tree_sha256(target)
        if not hmac.compare_digest(actual, expected):
            raise RuntimeError(f"downloaded {entry['model_id']} tree SHA-256 does not match")
    _write_marker(target, entry)


def _positive_int_env(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1 or value > maximum:
        raise RuntimeError(f"{name} must be between 1 and {maximum}")
    return value


def _download_model(target: Path, entry: dict[str, str]) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency contract catches this
        raise RuntimeError("huggingface-hub is required for model auto-download") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.download-", dir=str(target.parent))
    )
    try:
        logger.info(
            "Downloading pinned Hugging Face model",
            extra={
                "model_id": entry["model_id"],
                "revision": entry["revision"],
                "download_url": entry["download_url"],
                "target": str(target),
            },
        )
        snapshot_download(
            repo_id=entry["model_id"],
            revision=entry["revision"],
            local_dir=temporary,
            token=False,
            endpoint=entry["hub"],
            etag_timeout=float(_positive_int_env("MODEL_DOWNLOAD_ETAG_TIMEOUT", 30, 300)),
            max_workers=_positive_int_env("MODEL_DOWNLOAD_MAX_WORKERS", 4, 32),
        )
        metadata = temporary / ".cache"
        if metadata.exists():
            shutil.rmtree(metadata)
        _validate_existing_tree(temporary, entry)
        if target.exists():
            raise RuntimeError(f"model cache target appeared during download: {target}")
        temporary.replace(target)
        logger.info(
            "Pinned Hugging Face model is ready",
            extra={"model_id": entry["model_id"], "target": str(target)},
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def model_auto_download_enabled() -> bool:
    configured = os.getenv("MODEL_AUTO_DOWNLOAD", "").strip().lower()
    if configured:
        return configured in _TRUE_VALUES
    return os.getenv("RAG_ENV", "laptop").strip().lower() == "base"


def prepare_runtime_model(
    config: dict[str, Any],
    model_name: str,
    *,
    force_download: bool = False,
) -> Path:
    """Ensure one configured model exists locally and satisfies its descriptor."""
    if model_name not in _MODEL_SECTIONS:
        raise RuntimeError(f"unsupported runtime model: {model_name}")
    section_name = _MODEL_SECTIONS[model_name]
    section = config.get(section_name, {})
    if not isinstance(section, dict):
        raise RuntimeError(f"{section_name} configuration must be a mapping")
    target_value = str(section.get("model_path") or "").strip()
    if not target_value:
        raise RuntimeError(f"{section_name}.model_path is required")
    target = Path(target_value).expanduser().resolve()

    manifest = _load_runtime_manifest(config)
    if manifest is None and not force_download and not model_auto_download_enabled():
        # Preserve the developer/test contract: local model loaders may point
        # at a pre-provisioned path without enabling managed downloads.
        return target
    entry = (
        _validated_manifest_entry(config, manifest, model_name)
        if manifest is not None
        else _local_entry(config, model_name)
    )
    if target.name != entry["path"]:
        raise RuntimeError(f"{section_name}.model_path must end with {entry['path']}")

    if target.is_dir():
        if not entry["sha256"] or _marker_matches(target, entry):
            if not (target / "config.json").is_file():
                raise RuntimeError(f"model cache is incomplete: {target}/config.json is missing")
            return target
        _validate_existing_tree(target, entry)
        return target

    if target.exists():
        raise RuntimeError(f"model_path exists but is not a directory: {target}")
    if not force_download and not model_auto_download_enabled():
        return target
    _download_model(target, entry)
    return target


def prepare_runtime_models(
    config: dict[str, Any], *, force_download: bool = False
) -> dict[str, Path]:
    """Materialize every enabled production model from the signed descriptor."""
    paths = {
        "bge-m3": prepare_runtime_model(
            config, "bge-m3", force_download=force_download
        )
    }
    reranker = config.get("reranker", {})
    if isinstance(reranker, dict) and reranker.get("enabled", False):
        paths["bge-reranker-v2-m3"] = prepare_runtime_model(
            config, "bge-reranker-v2-m3", force_download=force_download
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "verify"),
        help="download and verify, or only verify an existing runtime cache",
    )
    args = parser.parse_args()

    from app.utils.config import get_settings

    config = get_settings()
    if args.command == "prepare":
        prepare_runtime_models(config, force_download=True)
        return
    for name in _MODEL_SECTIONS:
        section = config.get(_MODEL_SECTIONS[name], {})
        if name == "bge-reranker-v2-m3" and isinstance(section, dict):
            if not section.get("enabled", False):
                continue
        prepare_runtime_model(config, name, force_download=False)


if __name__ == "__main__":
    main()
