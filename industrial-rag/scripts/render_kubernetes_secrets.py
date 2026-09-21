"""Render allow-listed Kubernetes Secrets from a protected env snapshot.

The input is supplied by a Secret Manager contract at release time.  This
script intentionally maps runtime database credentials separately from the
migration administrator credentials and never prints secret values.
"""

from __future__ import annotations

import argparse
import base64
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

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

RUNTIME_MAPPING = {
    "RAG_API_KEY": "RAG_API_KEY",
    "RAG_METRICS_TOKEN": "RAG_METRICS_TOKEN",
    "RAG_SERVICE_TENANT_ID": "RAG_SERVICE_TENANT_ID",
    "RAG_SERVICE_ROLES": "RAG_SERVICE_ROLES",
    "POSTGRES_HOST": "POSTGRES_HOST",
    "POSTGRES_PORT": "POSTGRES_PORT",
    "POSTGRES_DB": "POSTGRES_DB",
    "POSTGRES_USER": "POSTGRES_RUNTIME_USER",
    "POSTGRES_PASSWORD": "POSTGRES_APP_PASSWORD",
    "POSTGRES_SSLMODE": "POSTGRES_SSLMODE",
    "POSTGRES_SSLROOTCERT": "POSTGRES_SSLROOTCERT",
    "REDIS_URL": "REDIS_URL",
    "DEEPSEEK_API_KEY": "DEEPSEEK_API_KEY",
    "DEEPSEEK_API_URL": "DEEPSEEK_API_URL",
    "DEEPSEEK_MODEL": "DEEPSEEK_MODEL",
    "OIDC_ISSUER": "OIDC_ISSUER",
    "OIDC_AUDIENCE": "OIDC_AUDIENCE",
    "OIDC_JWKS_URL": "OIDC_JWKS_URL",
    "S3_ENDPOINT_URL": "S3_ENDPOINT_URL",
    "S3_BUCKET": "S3_BUCKET",
    "S3_REGION": "S3_REGION",
    "S3_ACCESS_KEY_ID": "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY": "S3_SECRET_ACCESS_KEY",
}
MIGRATION_MAPPING = {
    "POSTGRES_ADMIN_USER": "POSTGRES_ADMIN_USER",
    "POSTGRES_ADMIN_PASSWORD": "POSTGRES_PASSWORD",
    "POSTGRES_HOST": "POSTGRES_HOST",
    "POSTGRES_PORT": "POSTGRES_PORT",
    "POSTGRES_DB": "POSTGRES_DB",
    "POSTGRES_RUNTIME_USER": "POSTGRES_RUNTIME_USER",
    "POSTGRES_RUNTIME_PASSWORD": "POSTGRES_APP_PASSWORD",
    "POSTGRES_SSLMODE": "POSTGRES_SSLMODE",
    "POSTGRES_SSLROOTCERT": "POSTGRES_SSLROOTCERT",
}
CANARY_MAPPING = {
    "CANARY_OIDC_TOKEN_URL": "CANARY_OIDC_TOKEN_URL",
    "CANARY_OIDC_CLIENT_AUTH_METHOD": "CANARY_OIDC_CLIENT_AUTH_METHOD",
    "CANARY_PRIMARY_CLIENT_ID": "CANARY_PRIMARY_CLIENT_ID",
    "CANARY_PRIMARY_CLIENT_SECRET": "CANARY_PRIMARY_CLIENT_SECRET",
    "CANARY_SECONDARY_CLIENT_ID": "CANARY_SECONDARY_CLIENT_ID",
    "CANARY_SECONDARY_CLIENT_SECRET": "CANARY_SECONDARY_CLIENT_SECRET",
    "CANARY_UPLOAD_CONTENT_B64": "CANARY_UPLOAD_CONTENT_B64",
    "CANARY_NO_ANSWER_QUERY": "CANARY_NO_ANSWER_QUERY",
    "CANARY_NO_ANSWER_EXPECTED_TEXT": "CANARY_NO_ANSWER_EXPECTED_TEXT",
}
CANARY_OPTIONAL_VALUES = ("CANARY_OIDC_SCOPE", "CANARY_OIDC_AUDIENCE")


def _secret_document(
    name: str,
    namespace: str,
    values: dict[str, str],
    *,
    config_digest: str,
) -> dict[str, Any]:
    encoded = {
        key: base64.b64encode(value.encode("utf-8")).decode("ascii")
        for key, value in values.items()
    }
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "industrial-rag",
                "industrial-rag/release-resource": "true",
            },
            "annotations": {"industrial-rag/config-digest": config_digest},
        },
        "type": "Opaque",
        "immutable": True,
        "data": encoded,
    }


def _mapped_values(source: dict[str, str], mapping: dict[str, str]) -> dict[str, str]:
    missing = [source_name for source_name in mapping.values() if not source.get(source_name, "").strip()]
    if missing:
        raise ValueError(
            "secret-manager snapshot is missing required values: "
            + ", ".join(sorted(set(missing)))
        )
    return {target: source[source_name] for target, source_name in mapping.items()}


def render(
    env_file: Path,
    *,
    namespace: str,
    runtime_output: Path,
    migration_output: Path,
    canary_output: Path,
) -> None:
    values = load_release_env_file(env_file)
    suffix = release_resource_suffix(values)
    config_digest = release_config_digest(values)
    runtime = _mapped_values(values, RUNTIME_MAPPING)
    migration = _mapped_values(values, MIGRATION_MAPPING)
    canary = _mapped_values(values, CANARY_MAPPING)
    canary.update(
        {name: values[name] for name in CANARY_OPTIONAL_VALUES if values.get(name, "").strip()}
    )
    for output, name, mapped in (
        (runtime_output, f"rag-runtime-{suffix}", runtime),
        (migration_output, f"rag-migration-{suffix}", migration),
        (canary_output, f"rag-canary-credentials-{suffix}", canary),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(
                _secret_document(
                    name,
                    namespace,
                    mapped,
                    config_digest=config_digest,
                ),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
            newline="\n",
        )
        try:
            output.chmod(0o600)
        except OSError:
            # The release runner still protects the containing directory with
            # umask 077; Windows and some mounted filesystems do not expose
            # POSIX mode bits.
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--namespace", default="rag-production")
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--migration-output", type=Path, required=True)
    parser.add_argument("--canary-output", type=Path, required=True)
    args = parser.parse_args()
    render(
        args.env_file,
        namespace=args.namespace,
        runtime_output=args.runtime_output,
        migration_output=args.migration_output,
        canary_output=args.canary_output,
    )
    print("Rendered Kubernetes runtime, migration, and functional-canary Secret manifests.")


if __name__ == "__main__":
    main()
