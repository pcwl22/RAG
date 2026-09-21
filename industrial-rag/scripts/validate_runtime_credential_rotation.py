"""Reject in-place PostgreSQL password rotation that would break live pods."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Any

import yaml


def _secret_values(path: Path) -> dict[str, str]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return {}
    document: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("kind") != "Secret":
        raise ValueError(f"{path} must contain one Kubernetes Secret")
    encoded = document.get("data") or {}
    if not isinstance(encoded, dict):
        raise ValueError(f"{path} Secret.data must be a mapping")
    values: dict[str, str] = {}
    for key, value in encoded.items():
        try:
            values[str(key)] = base64.b64decode(str(value), validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"{path} contains invalid base64 Secret data for {key}") from exc
    return values


def validate_rotation(current_path: Path, desired_path: Path) -> list[str]:
    current = _secret_values(current_path)
    desired = _secret_values(desired_path)
    if not current:
        return []
    current_user = current.get("POSTGRES_USER", "")
    desired_user = desired.get("POSTGRES_USER", "")
    current_password = current.get("POSTGRES_PASSWORD", "")
    desired_password = desired.get("POSTGRES_PASSWORD", "")
    if not desired_user or not desired_password:
        return ["desired rag-runtime Secret is missing PostgreSQL credentials"]
    if current_user == desired_user and current_password != desired_password:
        return [
            "runtime PostgreSQL password rotation must use a new versioned "
            "POSTGRES_RUNTIME_USER so old and new pods can overlap safely"
        ]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--desired", type=Path, required=True)
    args = parser.parse_args()
    errors = validate_rotation(args.current, args.desired)
    if errors:
        raise SystemExit("\n".join(errors))
    print("Runtime credential rotation contract passed.")


if __name__ == "__main__":
    main()
