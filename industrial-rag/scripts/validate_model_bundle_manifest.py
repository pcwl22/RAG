"""Validate the pinned Hugging Face model-source manifest.

The published descriptor contains no model weights.  When ``--models-root`` is
provided (for an already downloaded runtime cache), the deterministic content
hashes are verified as an additional integrity check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

_SHA256_LENGTH = 64
_HUB = "https://huggingface.co"
_REQUIRED_MODELS = {
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "sha256": "b1634ec79b0cd6e439ac1471698e15e6a23ea0365908a1c532a8e94b8c048c55",
    },
    "bge-reranker-v2-m3": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "sha256": "5ed918e794d4ce634a4e4fa87e8d0c0436c73176d09ff7531fca5920511aa474",
    },
}
_IGNORED_TREE_PARTS = {".git", ".cache"}
_RUNTIME_MARKER = ".industrial-rag-model.json"


def tree_sha256(path: Path) -> str:
    """Hash relative names and file bytes in stable lexical order."""
    digest = hashlib.sha256()
    symlinks = [item for item in path.rglob("*") if item.is_symlink()]
    if symlinks:
        raise ValueError(f"model tree must not contain symlinks: {symlinks[0]}")
    for file_path in sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and not _IGNORED_TREE_PARTS.intersection(item.relative_to(path).parts)
            and item.name != _RUNTIME_MARKER
        ),
        key=lambda item: item.relative_to(path).as_posix(),
    ):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(len(chunk).to_bytes(8, "big"))
                digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(
    path: Path,
    models_root: Path | None = None,
    *,
    allow_placeholders: bool = False,
) -> list[str]:
    errors: list[str] = []
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid model manifest: {exc}"]
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        errors.append("schema_version must be 2")
    if isinstance(payload, dict) and payload.get("hub") != _HUB:
        errors.append(f"hub must be {_HUB}")
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, dict):
        return errors + ["models must be an object"]
    if set(models) != set(_REQUIRED_MODELS):
        errors.append("models must contain exactly bge-m3 and bge-reranker-v2-m3")
    for name, specification in _REQUIRED_MODELS.items():
        entry = models.get(name)
        if not isinstance(entry, dict):
            errors.append(f"{name} manifest entry must be an object")
            continue
        model_id = specification["model_id"]
        revision = specification["revision"]
        if entry.get("model_id") != model_id:
            errors.append(f"{name}.model_id is incorrect")
        if entry.get("revision") != revision:
            errors.append(f"{name}.revision is incorrect")
        source_url = f"{_HUB}/{model_id}"
        download_url = f"{source_url}/tree/{revision}"
        if entry.get("source_url") != source_url:
            errors.append(f"{name}.source_url is incorrect")
        if entry.get("download_url") != download_url:
            errors.append(f"{name}.download_url is incorrect")
        relative_path = entry.get("path")
        digest = str(entry.get("sha256") or "")
        if relative_path != name:
            errors.append(f"{name}.path must be {name}")
        if digest == "REPLACE_WITH_SHA256_OF_DETERMINISTIC_MODEL_TREE":
            if not allow_placeholders:
                errors.append(f"{name}.sha256 must be a 64-character hexadecimal digest")
        elif len(digest) != _SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in digest.lower()
        ):
            errors.append(f"{name}.sha256 must be a 64-character hexadecimal digest")
        elif digest.lower() != specification["sha256"]:
            errors.append(f"{name}.sha256 is not the approved deterministic tree digest")
        if models_root is not None and isinstance(relative_path, str):
            model_path = (models_root / relative_path).resolve()
            root = models_root.resolve()
            if root not in model_path.parents:
                errors.append(f"{name}.path escapes models root")
            elif not model_path.is_dir():
                errors.append(f"missing model directory: {model_path}")
            elif len(digest) == _SHA256_LENGTH:
                try:
                    actual_digest = tree_sha256(model_path)
                except ValueError as exc:
                    errors.append(f"{name} {exc}")
                else:
                    if actual_digest != digest.lower():
                        errors.append(f"{name} tree SHA-256 does not match manifest")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args()
    errors = validate_manifest(
        args.manifest,
        args.models_root,
        allow_placeholders=args.allow_placeholders,
    )
    if errors:
        raise SystemExit("\n".join(errors))
    print("Model bundle manifest is valid.")


if __name__ == "__main__":
    main()
