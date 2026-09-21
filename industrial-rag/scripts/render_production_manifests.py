"""Inject approved release values into a rendered Kustomize manifest."""

from __future__ import annotations

import argparse
import importlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.validate_production_env import (
        load_release_env_file,
        release_config_digest,
        release_resource_suffix,
    )
else:
    _validator = importlib.import_module(
        "scripts.validate_production_env" if __package__ else "validate_production_env"
    )
    load_release_env_file: Callable[[Path], dict[str, str]] = (
        _validator.load_release_env_file
    )
    release_config_digest: Callable[[dict[str, str]], str] = (
        _validator.release_config_digest
    )
    release_resource_suffix: Callable[[dict[str, str]], str] = (
        _validator.release_resource_suffix
    )

_DIGEST_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$", re.IGNORECASE)


def _required(values: dict[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required for release manifest rendering")
    return value


def render(input_path: Path, output_path: Path, env_path: Path) -> None:
    values = load_release_env_file(env_path)
    text = input_path.read_text(encoding="utf-8")
    config_digest = release_config_digest(values)
    resource_suffix = release_resource_suffix(values)
    release_resource_names = (
        "rag-runtime-config",
        "rag-frontend-config",
        "rag-canary-script",
        "rag-migration",
        "rag-runtime",
        "rag-canary-credentials",
    )
    resource_pattern = "|".join(re.escape(name) for name in release_resource_names)
    text = re.sub(
        rf"\b(?:{resource_pattern})\b",
        lambda match: f"{match.group(0)}-{resource_suffix}",
        text,
    )
    replacements = {
        "industrial-rag-api": _required(values, "API_IMAGE"),
        "industrial-rag-worker": _required(values, "WORKER_IMAGE"),
        "industrial-rag-frontend": _required(values, "FRONTEND_IMAGE"),
    }
    for name, image in replacements.items():
        if not _DIGEST_IMAGE.fullmatch(image):
            raise ValueError(f"{name} must be referenced by an immutable digest")
        text = re.sub(
            rf"registry\.example\.invalid/{re.escape(name)}@sha256:0{{64}}",
            f"{image}",
            text,
        )

    # Bind the public release identity and all signed image references inside
    # an immutable ConfigMap.  This is intentionally separate from
    # ``config_digest``: resource names expose only a hash of RELEASE_ID and
    # never a hash of the protected secret snapshot.
    public_release_values = {
        "__RAG_RELEASE_ID__": _required(values, "RELEASE_ID"),
        "__RAG_API_IMAGE__": replacements["industrial-rag-api"],
        "__RAG_WORKER_IMAGE__": replacements["industrial-rag-worker"],
        "__RAG_FRONTEND_IMAGE__": replacements["industrial-rag-frontend"],
        "__RAG_MODEL_BUNDLE_IMAGE__": _required(values, "MODEL_BUNDLE_IMAGE"),
    }
    for placeholder, value in public_release_values.items():
        text = text.replace(f'"{placeholder}"', json.dumps(value, ensure_ascii=False))

    model_digest = _required(values, "MODEL_BUNDLE_DIGEST")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", model_digest, re.IGNORECASE):
        raise ValueError("MODEL_BUNDLE_DIGEST must be sha256:<64 hex characters>")
    text = text.replace(
        "sha256:REPLACE_WITH_APPROVED_MODEL_BUNDLE_DIGEST",
        model_digest,
    )
    manifest_digest = _required(values, "MODEL_MANIFEST_SHA256")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_digest, re.IGNORECASE):
        raise ValueError("MODEL_MANIFEST_SHA256 must be sha256:<64 hex characters>")
    text = text.replace(
        "sha256:REPLACE_WITH_APPROVED_MODEL_MANIFEST_SHA256",
        manifest_digest,
    )

    host = _required(values, "INGRESS_HOST")
    if "example.invalid" in host or any(character.isspace() for character in host):
        raise ValueError("INGRESS_HOST must be a real DNS hostname")
    text = text.replace("rag.example.invalid", host)

    oidc_connect_src = _required(values, "OIDC_CONNECT_SRC")
    text = text.replace("https://id.example.invalid", oidc_connect_src.rstrip("/"))
    text = text.replace(
        "192.0.2.0/24",
        _required(values, "INGRESS_PROXY_CIDR"),
    )
    runtime_oidc_values = {
        "__RAG_RUNTIME_OIDC_ISSUER__": _required(values, "VITE_OIDC_ISSUER").rstrip("/"),
        "__RAG_RUNTIME_OIDC_CLIENT_ID__": _required(values, "VITE_OIDC_CLIENT_ID"),
        "__RAG_RUNTIME_OIDC_AUDIENCE__": _required(values, "VITE_OIDC_AUDIENCE"),
    }
    for placeholder, value in runtime_oidc_values.items():
        text = text.replace(f'"{placeholder}"', json.dumps(value, ensure_ascii=False))
    runtime_storage_values = {
        "__RAG_S3_ADDRESSING_STYLE__": _required(values, "S3_ADDRESSING_STYLE"),
        "__RAG_S3_FAILED_PREFIX__": _required(values, "S3_FAILED_PREFIX"),
    }
    for placeholder, value in runtime_storage_values.items():
        text = text.replace(f'"{placeholder}"', json.dumps(value, ensure_ascii=False))
    text = text.replace(
        "sha256:REPLACE_WITH_RELEASE_CONFIG_DIGEST",
        config_digest,
    )
    output_path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    render(args.input, args.output, args.env_file)
    print(f"Rendered production manifest: {args.output}")


if __name__ == "__main__":
    main()
