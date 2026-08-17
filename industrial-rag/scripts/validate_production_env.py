"""Validate production security settings without printing secret values."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import UUID

REQUIRED_VALUES = (
    "RAG_API_KEY",
    "RAG_METRICS_TOKEN",
    "POSTGRES_PASSWORD",
    "POSTGRES_APP_PASSWORD",
    "REDIS_PASSWORD",
    "COMPOSE_REDIS_URL",
    "OIDC_ISSUER",
    "OIDC_AUDIENCE",
    "OIDC_JWKS_URL",
    "RAG_SERVICE_TENANT_ID",
)
PLACEHOLDER_PATTERN = re.compile(
    r"change[-_ ]?me|replace[-_ ]?with|xxxxx|example|your[-_ ]|<[^>]+>", re.IGNORECASE
)
TRUE_VALUES = {"1", "true", "yes", "on"}


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
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

    for name, minimum in (
        ("RAG_API_KEY", 32),
        ("RAG_METRICS_TOKEN", 32),
        ("POSTGRES_PASSWORD", 16),
        ("POSTGRES_APP_PASSWORD", 16),
        ("REDIS_PASSWORD", 16),
    ):
        value = values.get(name, "")
        if value and not (allow_placeholders and PLACEHOLDER_PATTERN.search(value)):
            if len(value) < minimum:
                errors.append(f"{name} must contain at least {minimum} characters")

    postgres_admin = values.get("POSTGRES_PASSWORD", "")
    postgres_runtime = values.get("POSTGRES_APP_PASSWORD", "")
    if postgres_admin and postgres_admin == postgres_runtime:
        errors.append("POSTGRES_PASSWORD and POSTGRES_APP_PASSWORD must be different")

    if values.get("QUEUE_PROVIDER", "").strip().lower() != "celery":
        errors.append("QUEUE_PROVIDER must be celery in production")
    if values.get("COMPOSE_OIDC_ENABLED", "").strip().lower() not in TRUE_VALUES:
        errors.append("COMPOSE_OIDC_ENABLED must be true in production")

    tenant_id = values.get("RAG_SERVICE_TENANT_ID", "")
    if tenant_id:
        try:
            UUID(tenant_id)
        except ValueError:
            errors.append("RAG_SERVICE_TENANT_ID must be a UUID")

    redis_url = values.get("COMPOSE_REDIS_URL", "")
    if redis_url:
        parsed_redis = urlsplit(redis_url)
        if parsed_redis.scheme not in {"redis", "rediss"}:
            errors.append("COMPOSE_REDIS_URL must use redis:// or rediss://")
        if not parsed_redis.hostname:
            errors.append("COMPOSE_REDIS_URL must include a host")
        if not parsed_redis.password:
            errors.append("COMPOSE_REDIS_URL must include a password")
        elif values.get("REDIS_PASSWORD") and unquote(parsed_redis.password) != values["REDIS_PASSWORD"]:
            errors.append("COMPOSE_REDIS_URL password must match REDIS_PASSWORD")
        tls_required = values.get("REDIS_REQUIRE_TLS", "").strip().lower() in TRUE_VALUES
        if (tls_required or not allow_local_http) and parsed_redis.scheme != "rediss":
            errors.append("COMPOSE_REDIS_URL must use rediss:// for production TLS")

    for name in ("OIDC_ISSUER", "OIDC_JWKS_URL"):
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

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Optional dotenv file; process env wins")
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
    values = load_env_file(args.env_file) if args.env_file else dict(os.environ)
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
