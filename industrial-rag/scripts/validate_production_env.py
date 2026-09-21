"""Validate production security settings without printing secret values."""

from __future__ import annotations

import argparse
import base64
import binascii
import ipaddress
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.domain_signal_map import is_known_out_of_scope  # noqa: E402
from app.utils.strict_dotenv import (  # noqa: E402
    load_env_file,
    load_release_env_file,
    release_config_digest,
    release_resource_suffix,
)

__all__ = [
    "is_immutable_image_reference",
    "load_env_file",
    "load_release_env_file",
    "release_config_digest",
    "release_resource_suffix",
    "resolve_validation_values",
    "validate_production_values",
]

REQUIRED_VALUES = (
    "RELEASE_ID",
    "CANARY_OIDC_TOKEN_URL",
    "CANARY_OIDC_CLIENT_AUTH_METHOD",
    "CANARY_PRIMARY_CLIENT_ID",
    "CANARY_PRIMARY_CLIENT_SECRET",
    "CANARY_SECONDARY_CLIENT_ID",
    "CANARY_SECONDARY_CLIENT_SECRET",
    "CANARY_UPLOAD_CONTENT_B64",
    "CANARY_NO_ANSWER_QUERY",
    "CANARY_NO_ANSWER_EXPECTED_TEXT",
    "RAG_API_KEY",
    "RAG_METRICS_TOKEN",
    "POSTGRES_PASSWORD",
    "POSTGRES_ADMIN_USER",
    "POSTGRES_RUNTIME_USER",
    "POSTGRES_APP_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_SSLMODE",
    "POSTGRES_SSLROOTCERT",
    "REDIS_PASSWORD",
    "COMPOSE_REDIS_URL",
    "OIDC_ENABLED",
    "OIDC_ISSUER",
    "OIDC_AUDIENCE",
    "OIDC_JWKS_URL",
    "RAG_SERVICE_TENANT_ID",
    "RAG_SERVICE_ROLES",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_API_URL",
    "DEEPSEEK_MODEL",
    "VITE_OIDC_ISSUER",
    "VITE_OIDC_CLIENT_ID",
    "VITE_OIDC_AUDIENCE",
    "REDIS_URL",
    "OBJECT_STORAGE_ENABLED",
    "S3_ENDPOINT_URL",
    "S3_BUCKET",
    "S3_REGION",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "S3_REQUIRE_TLS",
    "S3_ADDRESSING_STYLE",
    "S3_FAILED_PREFIX",
    "MODEL_BUNDLE_IMAGE",
    "MODEL_BUNDLE_DIGEST",
    "MODEL_MANIFEST_SHA256",
    "API_IMAGE",
    "WORKER_IMAGE",
    "FRONTEND_IMAGE",
    "INGRESS_HOST",
    "INGRESS_PROXY_CIDR",
    "OIDC_CONNECT_SRC",
)
PLACEHOLDER_PATTERN = re.compile(
    r"change[-_ ]?me|replace[-_ ]?with|xxxxx|example|your[-_ ]|<[^>]+>", re.IGNORECASE
)
TRUE_VALUES = {"1", "true", "yes", "on"}
SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)
IMMUTABLE_IMAGE_PATTERN = re.compile(
    r"^[A-Za-z0-9.-]+(?::[0-9]+)?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
    r"@sha256:[0-9a-f]{64}$"
)


def is_immutable_image_reference(value: str) -> bool:
    """Return whether a value is a shell-safe, registry-qualified OCI digest ref."""
    return IMMUTABLE_IMAGE_PATTERN.fullmatch(value) is not None


def resolve_validation_values(
    env_file: Path | None,
    *,
    release_snapshot: bool = False,
    process_values: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve the exact values validated by the selected CLI contract."""
    if release_snapshot and env_file is None:
        raise ValueError("release_snapshot requires env_file")
    if env_file:
        loader = load_release_env_file if release_snapshot else load_env_file
        values = loader(env_file)
    else:
        values = {}
    if not release_snapshot:
        values.update(process_values if process_values is not None else os.environ)
    return values


def validate_production_values(
    values: Mapping[str, str],
    *,
    allow_placeholders: bool = False,
    allow_local_http: bool = False,
) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_VALUES:
        value = values.get(name, "").strip()
        if not value:
            errors.append(f"{name} is required")
        elif not allow_placeholders and PLACEHOLDER_PATTERN.search(value):
            errors.append(f"{name} must not use an example or placeholder value")

    release_id = values.get("RELEASE_ID", "").strip()
    if release_id and not (allow_placeholders and PLACEHOLDER_PATTERN.search(release_id)):
        try:
            parsed_release_id = UUID(release_id)
        except ValueError:
            parsed_release_id = None
        if (
            parsed_release_id is None
            or parsed_release_id.version != 4
            or str(parsed_release_id) != release_id
        ):
            errors.append("RELEASE_ID must be a canonical lowercase UUIDv4")

    for name, minimum in (
        ("RAG_API_KEY", 32),
        ("RAG_METRICS_TOKEN", 32),
        ("POSTGRES_PASSWORD", 16),
        ("POSTGRES_APP_PASSWORD", 16),
        ("REDIS_PASSWORD", 16),
        ("DEEPSEEK_API_KEY", 32),
        ("S3_SECRET_ACCESS_KEY", 16),
        ("CANARY_PRIMARY_CLIENT_SECRET", 16),
        ("CANARY_SECONDARY_CLIENT_SECRET", 16),
    ):
        value = values.get(name, "")
        if value and not (allow_placeholders and PLACEHOLDER_PATTERN.search(value)):
            if len(value) < minimum:
                errors.append(f"{name} must contain at least {minimum} characters")

    postgres_admin = values.get("POSTGRES_PASSWORD", "")
    postgres_runtime = values.get("POSTGRES_APP_PASSWORD", "")
    if postgres_admin and postgres_admin == postgres_runtime:
        errors.append("POSTGRES_PASSWORD and POSTGRES_APP_PASSWORD must be different")
    admin_user = values.get("POSTGRES_ADMIN_USER", "").strip()
    runtime_user = values.get("POSTGRES_RUNTIME_USER", "").strip()
    if admin_user and runtime_user and admin_user == runtime_user:
        errors.append("POSTGRES_ADMIN_USER and POSTGRES_RUNTIME_USER must be different")
    postgres_host = values.get("POSTGRES_HOST", "").strip()
    if postgres_host and any(character.isspace() for character in postgres_host):
        errors.append("POSTGRES_HOST must not contain whitespace")
    try:
        postgres_port = int(values.get("POSTGRES_PORT", ""))
    except ValueError:
        postgres_port = 0
    if not 1 <= postgres_port <= 65535:
        errors.append("POSTGRES_PORT must be an integer between 1 and 65535")
    postgres_db = values.get("POSTGRES_DB", "").strip()
    if postgres_db and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,62}", postgres_db):
        errors.append("POSTGRES_DB must be a safe PostgreSQL database identifier")
    postgres_sslmode = values.get("POSTGRES_SSLMODE", "").strip().lower()
    postgres_sslrootcert = values.get("POSTGRES_SSLROOTCERT", "").strip()
    if not allow_local_http and postgres_sslmode != "verify-full":
        errors.append("POSTGRES_SSLMODE must be verify-full in production")
    elif allow_local_http and postgres_sslmode not in {
        "disable", "allow", "prefer", "require", "verify-ca", "verify-full"
    }:
        errors.append("POSTGRES_SSLMODE is not a supported libpq SSL mode")
    if postgres_sslrootcert:
        normalized_rootcert = postgres_sslrootcert.replace("\\", "/")
        if not normalized_rootcert.startswith("/") or ".." in normalized_rootcert.split("/"):
            errors.append("POSTGRES_SSLROOTCERT must be an absolute container path")

    if values.get("QUEUE_PROVIDER", "").strip().lower() != "celery":
        errors.append("QUEUE_PROVIDER must be celery in production")
    if values.get("OIDC_ENABLED", "").strip().lower() not in TRUE_VALUES:
        errors.append("OIDC_ENABLED must be true in production")
    if values.get("RAG_SECURE_MODE", "").strip().lower() not in TRUE_VALUES and not allow_local_http:
        errors.append("RAG_SECURE_MODE must be true in production")
    if values.get("RAG_ENV", "").strip().lower() != "base":
        errors.append("RAG_ENV must be base in production")
    if values.get("POSTGRES_AUTO_MIGRATE", "").strip().lower() not in {"false", "0", "no", "off"}:
        errors.append("POSTGRES_AUTO_MIGRATE must be false in production")

    tenant_id = values.get("RAG_SERVICE_TENANT_ID", "")
    if tenant_id:
        try:
            UUID(tenant_id)
        except ValueError:
            errors.append("RAG_SERVICE_TENANT_ID must be a UUID")
    service_roles = {
        role.strip().lower()
        for role in values.get("RAG_SERVICE_ROLES", "").split(",")
        if role.strip()
    }
    if not service_roles or not service_roles <= {"viewer", "editor", "admin"}:
        errors.append("RAG_SERVICE_ROLES must contain only viewer, editor, or admin")

    redis_urls = {
        name: values.get(name, "")
        for name in ("REDIS_URL", "COMPOSE_REDIS_URL")
    }
    for redis_name, redis_url in redis_urls.items():
        if not redis_url:
            continue
        parsed_redis = urlsplit(redis_url)
        if parsed_redis.scheme not in {"redis", "rediss"}:
            errors.append(f"{redis_name} must use redis:// or rediss://")
        if not parsed_redis.hostname:
            errors.append(f"{redis_name} must include a host")
        if not parsed_redis.password:
            errors.append(f"{redis_name} must include a password")
        elif values.get("REDIS_PASSWORD") and unquote(parsed_redis.password) != values["REDIS_PASSWORD"]:
            errors.append(f"{redis_name} password must match REDIS_PASSWORD")
        tls_required = values.get("REDIS_REQUIRE_TLS", "").strip().lower() in TRUE_VALUES
        if (tls_required or not allow_local_http) and parsed_redis.scheme != "rediss":
            errors.append(f"{redis_name} must use rediss:// for production TLS")

    for name in ("OIDC_ISSUER", "OIDC_JWKS_URL", "VITE_OIDC_ISSUER"):
        url = values.get(name, "")
        if not url:
            continue
        parsed_url = urlsplit(url)
        if not parsed_url.hostname:
            errors.append(f"{name} must be an absolute URL")
        elif not allow_local_http and parsed_url.scheme != "https":
            errors.append(f"{name} must use https:// in production")
        elif allow_local_http and parsed_url.scheme not in {"http", "https"}:
            errors.append(f"{name} must use http:// or https://")

    llm_url = values.get("DEEPSEEK_API_URL", "")
    if llm_url:
        parsed_llm = urlsplit(llm_url)
        if not parsed_llm.hostname:
            errors.append("DEEPSEEK_API_URL must be an absolute URL")
        elif not allow_local_http and parsed_llm.scheme != "https":
            errors.append("DEEPSEEK_API_URL must use https:// in production")
        elif allow_local_http and parsed_llm.scheme not in {"http", "https"}:
            errors.append("DEEPSEEK_API_URL must use http:// or https://")

    if values.get("VITE_OIDC_ISSUER") and values.get("OIDC_ISSUER"):
        if values["VITE_OIDC_ISSUER"].rstrip("/") != values["OIDC_ISSUER"].rstrip("/"):
            errors.append("VITE_OIDC_ISSUER must match OIDC_ISSUER")
    if values.get("VITE_OIDC_AUDIENCE") and values.get("OIDC_AUDIENCE"):
        if values["VITE_OIDC_AUDIENCE"] != values["OIDC_AUDIENCE"]:
            errors.append("VITE_OIDC_AUDIENCE must match OIDC_AUDIENCE")

    canary_token_url = values.get("CANARY_OIDC_TOKEN_URL", "")
    if canary_token_url:
        parsed_token_url = urlsplit(canary_token_url)
        if not parsed_token_url.hostname:
            errors.append("CANARY_OIDC_TOKEN_URL must be an absolute URL")
        elif not allow_local_http and parsed_token_url.scheme != "https":
            errors.append("CANARY_OIDC_TOKEN_URL must use https:// in production")
        elif allow_local_http and parsed_token_url.scheme not in {"http", "https"}:
            errors.append("CANARY_OIDC_TOKEN_URL must use http:// or https://")
        issuer = urlsplit(values.get("OIDC_ISSUER", ""))
        if issuer.scheme and issuer.netloc and (
            parsed_token_url.scheme,
            parsed_token_url.netloc,
        ) != (issuer.scheme, issuer.netloc):
            errors.append("CANARY_OIDC_TOKEN_URL must share the OIDC_ISSUER origin")
    auth_method = values.get("CANARY_OIDC_CLIENT_AUTH_METHOD", "").strip()
    if auth_method not in {"client_secret_basic", "client_secret_post"}:
        errors.append(
            "CANARY_OIDC_CLIENT_AUTH_METHOD must be client_secret_basic or client_secret_post"
        )
    primary_client = values.get("CANARY_PRIMARY_CLIENT_ID", "").strip()
    secondary_client = values.get("CANARY_SECONDARY_CLIENT_ID", "").strip()
    if primary_client and primary_client == secondary_client:
        errors.append("CANARY_PRIMARY_CLIENT_ID and CANARY_SECONDARY_CLIENT_ID must be different")
    primary_secret = values.get("CANARY_PRIMARY_CLIENT_SECRET", "")
    secondary_secret = values.get("CANARY_SECONDARY_CLIENT_SECRET", "")
    if primary_secret and primary_secret == secondary_secret:
        errors.append(
            "CANARY_PRIMARY_CLIENT_SECRET and CANARY_SECONDARY_CLIENT_SECRET must be different"
        )
    encoded_sample = values.get("CANARY_UPLOAD_CONTENT_B64", "")
    if encoded_sample:
        try:
            sample = base64.b64decode(encoded_sample, validate=True)
            sample.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            errors.append("CANARY_UPLOAD_CONTENT_B64 must contain base64-encoded UTF-8")
        else:
            if len(sample) < 80 or len(sample) > 256 * 1024 or b"\x00" in sample:
                errors.append(
                    "CANARY_UPLOAD_CONTENT_B64 must decode to 80 bytes through 256 KiB of UTF-8 text"
                )
    for name in ("CANARY_NO_ANSWER_QUERY", "CANARY_NO_ANSWER_EXPECTED_TEXT"):
        value = values.get(name, "")
        if value and len(value) > 1000:
            errors.append(f"{name} must contain at most 1000 characters")
    no_answer_query = values.get("CANARY_NO_ANSWER_QUERY", "").strip()
    if no_answer_query and not is_known_out_of_scope(no_answer_query):
        errors.append("CANARY_NO_ANSWER_QUERY must match an approved out-of-scope signal rule")

    object_storage_enabled = values.get("OBJECT_STORAGE_ENABLED", "").strip().lower()
    if object_storage_enabled not in TRUE_VALUES:
        errors.append("OBJECT_STORAGE_ENABLED must be true in production")
    s3_endpoint = values.get("S3_ENDPOINT_URL", "")
    if s3_endpoint:
        parsed_s3 = urlsplit(s3_endpoint)
        if not parsed_s3.hostname:
            errors.append("S3_ENDPOINT_URL must be an absolute URL")
        elif not allow_local_http and parsed_s3.scheme != "https":
            errors.append("S3_ENDPOINT_URL must use https:// in production")
        elif allow_local_http and parsed_s3.scheme not in {"http", "https"}:
            errors.append("S3_ENDPOINT_URL must use http:// or https://")
    if values.get("S3_REQUIRE_TLS", "").strip().lower() not in TRUE_VALUES and not allow_local_http:
        errors.append("S3_REQUIRE_TLS must be true in production")
    addressing_style = values.get("S3_ADDRESSING_STYLE", "").strip().lower()
    if addressing_style not in {"path", "virtual"}:
        errors.append("S3_ADDRESSING_STYLE must be path or virtual")
    failed_prefix = values.get("S3_FAILED_PREFIX", "").strip().strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", failed_prefix):
        errors.append("S3_FAILED_PREFIX must be a single safe path segment")

    digest = values.get("MODEL_BUNDLE_DIGEST", "").strip()
    if digest and not SHA256_DIGEST_PATTERN.fullmatch(digest):
        errors.append("MODEL_BUNDLE_DIGEST must be sha256:<64 lowercase hex characters>")
    elif digest.lower() == "sha256:" + "0" * 64 and not allow_placeholders:
        errors.append("MODEL_BUNDLE_DIGEST must not use the zero placeholder digest")
    manifest_digest = values.get("MODEL_MANIFEST_SHA256", "").strip()
    if manifest_digest and not SHA256_DIGEST_PATTERN.fullmatch(manifest_digest):
        errors.append("MODEL_MANIFEST_SHA256 must be sha256:<64 lowercase hex characters>")
    elif manifest_digest.lower() == "sha256:" + "0" * 64 and not allow_placeholders:
        errors.append("MODEL_MANIFEST_SHA256 must not use the zero placeholder digest")
    image = values.get("MODEL_BUNDLE_IMAGE", "").strip()
    if image and not is_immutable_image_reference(image):
        errors.append("MODEL_BUNDLE_IMAGE must be referenced by digest")
    elif image.lower().endswith("@sha256:" + "0" * 64) and not allow_placeholders:
        errors.append("MODEL_BUNDLE_IMAGE must not use the zero placeholder digest")
    elif image and digest and image.rsplit("@", 1)[-1].lower() != digest.lower():
        errors.append("MODEL_BUNDLE_DIGEST must match the MODEL_BUNDLE_IMAGE OCI digest")
    for image_name in ("API_IMAGE", "WORKER_IMAGE", "FRONTEND_IMAGE"):
        image_value = values.get(image_name, "").strip()
        if image_value and not is_immutable_image_reference(image_value):
            errors.append(f"{image_name} must be referenced by digest")
        elif image_value.lower().endswith("@sha256:" + "0" * 64) and not allow_placeholders:
            errors.append(f"{image_name} must not use the zero placeholder digest")

    ingress_host = values.get("INGRESS_HOST", "").strip()
    if ingress_host and (
        any(character.isspace() for character in ingress_host)
        or "/" in ingress_host
        or ":" in ingress_host
    ):
        errors.append("INGRESS_HOST must be a DNS hostname")
    proxy_cidr = values.get("INGRESS_PROXY_CIDR", "").strip()
    if proxy_cidr:
        try:
            proxy_network = ipaddress.ip_network(proxy_cidr, strict=False)
        except ValueError:
            errors.append("INGRESS_PROXY_CIDR must be one explicit IPv4 or IPv6 CIDR")
        else:
            documentation_networks = {
                ipaddress.ip_network("192.0.2.0/24"),
                ipaddress.ip_network("198.51.100.0/24"),
                ipaddress.ip_network("203.0.113.0/24"),
                ipaddress.ip_network("2001:db8::/32"),
            }
            if not allow_placeholders and (
                proxy_network.prefixlen == 0 or proxy_network in documentation_networks
            ):
                errors.append(
                    "INGRESS_PROXY_CIDR must be the narrow production ingress proxy CIDR"
                )
    connect_src = values.get("OIDC_CONNECT_SRC", "").strip()
    if connect_src:
        parsed_connect = urlsplit(connect_src)
        if not parsed_connect.hostname:
            errors.append("OIDC_CONNECT_SRC must be an absolute URL")
        elif not allow_local_http and parsed_connect.scheme != "https":
            errors.append("OIDC_CONNECT_SRC must use https:// in production")
        issuer = urlsplit(values.get("VITE_OIDC_ISSUER", ""))
        if issuer.scheme and issuer.netloc:
            issuer_origin = f"{issuer.scheme}://{issuer.netloc}"
            if connect_src.rstrip("/") != issuer_origin:
                errors.append("OIDC_CONNECT_SRC must match the VITE_OIDC_ISSUER origin")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Optional dotenv file; process env wins")
    parser.add_argument(
        "--release-snapshot",
        action="store_true",
        help=(
            "Treat --env-file as the sole protected release source and reject "
            "normalized duplicate keys"
        ),
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Validate a committed example file without accepting it for deployment",
    )
    parser.add_argument(
        "--allow-local-http",
        action="store_true",
        help="Allow local http:// OIDC and redis:// for development acceptance only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        values = resolve_validation_values(
            args.env_file,
            release_snapshot=args.release_snapshot,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    errors = validate_production_values(
        values,
        allow_placeholders=args.allow_placeholders,
        allow_local_http=args.allow_local_http,
    )
    result = {"passed": not errors, "check": "production_environment", "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
