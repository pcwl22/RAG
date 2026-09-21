"""Authenticated principals, OIDC validation, and tenant request context."""

from __future__ import annotations

import asyncio
import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"


class OIDCAuthenticationError(ValueError):
    """The bearer token itself is invalid and should receive HTTP 401."""


class OIDCProviderUnavailable(RuntimeError):
    """The identity provider/JWKS dependency could not validate any token."""


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: str
    roles: frozenset[str]
    auth_type: str


_principal_context: ContextVar[Principal | None] = ContextVar("rag_principal", default=None)


def normalize_tenant_id(value: Any) -> str:
    tenant_id = str(value or os.getenv("RAG_SERVICE_TENANT_ID") or DEFAULT_TENANT_ID).strip()
    try:
        return str(UUID(tenant_id))
    except ValueError as exc:
        raise ValueError("tenant_id must be a UUID") from exc


def current_principal() -> Principal:
    principal = _principal_context.get()
    if principal is not None:
        return principal
    return Principal(
        subject="local-system",
        tenant_id=normalize_tenant_id(None),
        roles=frozenset({"viewer", "editor", "admin"}),
        auth_type="local",
    )


def current_tenant_id() -> str:
    return current_principal().tenant_id


def set_current_principal(principal: Principal) -> Token[Principal | None]:
    return _principal_context.set(principal)


def reset_current_principal(token: Token[Principal | None]) -> None:
    _principal_context.reset(token)


def _claim(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


class OIDCValidator:
    """Validate standard OIDC access tokens using a cached JWKS client."""

    def __init__(self, config: dict[str, Any]):
        oidc = config.get("security", {}).get("oidc", {})
        self.issuer = str(os.getenv("OIDC_ISSUER") or oidc.get("issuer") or "").rstrip("/")
        self.audience = str(os.getenv("OIDC_AUDIENCE") or oidc.get("audience") or "")
        self.jwks_url = str(
            os.getenv("OIDC_JWKS_URL")
            or oidc.get("jwks_url")
            or (f"{self.issuer}/protocol/openid-connect/certs" if self.issuer else "")
        )
        self.tenant_claim = str(oidc.get("tenant_claim", "tenant_id"))
        self.roles_claim = str(oidc.get("roles_claim", "realm_access.roles"))
        self.algorithms = list(oidc.get("algorithms", ["RS256"]))
        self._configured_enabled = bool(oidc.get("enabled", False))
        self._jwks_client: Any | None = None

    @property
    def enabled(self) -> bool:
        """Resolve the environment override without discarding the JWKS cache."""
        enabled_value = os.getenv("OIDC_ENABLED")
        if enabled_value is None:
            return self._configured_enabled
        return str(enabled_value).strip().lower() in {"1", "true", "yes", "on"}

    def validate_configuration(self) -> None:
        if self.enabled and (not self.issuer or not self.audience or not self.jwks_url):
            raise RuntimeError("OIDC requires issuer, audience, and jwks_url")
        if self.enabled:
            allowed = {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512"}
            if not self.algorithms or not set(self.algorithms).issubset(allowed):
                raise RuntimeError("OIDC algorithms must use an approved asymmetric signature")
            secure_mode = str(os.getenv("RAG_SECURE_MODE", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if secure_mode:
                for name, value in (("OIDC_ISSUER", self.issuer), ("OIDC_JWKS_URL", self.jwks_url)):
                    if urlsplit(value).scheme != "https":
                        raise RuntimeError(f"{name} must use https:// when RAG_SECURE_MODE is enabled")

    def _decode(self, token: str) -> Principal:
        try:
            import jwt
        except ImportError as exc:
            raise RuntimeError("PyJWT[crypto] is required when OIDC is enabled") from exc

        if self._jwks_client is None:
            self._jwks_client = jwt.PyJWKClient(self.jwks_url, cache_jwk_set=True, lifespan=300)
        signing_key = self._jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=self.algorithms,
            audience=self.audience,
            issuer=self.issuer,
            options={"require": ["exp", "iat", "iss", "sub"]},
        )
        roles_value = _claim(payload, self.roles_claim) or []
        if isinstance(roles_value, str):
            roles_value = [roles_value]
        roles = frozenset(str(role) for role in roles_value if str(role) in {"viewer", "editor", "admin"})
        tenant_claim = _claim(payload, self.tenant_claim)
        if tenant_claim is None or not str(tenant_claim).strip():
            raise ValueError(f"OIDC token is missing required {self.tenant_claim} claim")
        return Principal(
            subject=str(payload["sub"]),
            tenant_id=normalize_tenant_id(tenant_claim),
            roles=roles,
            auth_type="oidc",
        )

    async def validate(self, token: str) -> Principal:
        try:
            return await asyncio.to_thread(self._decode, token)
        except (OIDCAuthenticationError, OIDCProviderUnavailable):
            raise
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise OIDCProviderUnavailable("OIDC provider is unavailable") from exc
        except Exception as exc:
            # PyJWT deliberately separates signature/claim failures from JWKS
            # transport failures.  Avoid importing the optional dependency at
            # module import time while still preserving that distinction.
            try:
                import jwt
            except ImportError:
                raise OIDCProviderUnavailable("OIDC validator dependency is unavailable") from exc

            invalid_token_types = tuple(
                candidate
                for candidate in (
                    getattr(jwt, "InvalidTokenError", None),
                    getattr(getattr(jwt, "exceptions", None), "InvalidTokenError", None),
                    getattr(getattr(jwt, "exceptions", None), "PyJWKClientError", None),
                )
                if isinstance(candidate, type)
            )
            connection_type = getattr(
                getattr(jwt, "exceptions", None),
                "PyJWKClientConnectionError",
                None,
            )
            jwk_set_type = getattr(getattr(jwt, "exceptions", None), "PyJWKSetError", None)
            if isinstance(connection_type, type) and isinstance(exc, connection_type):
                raise OIDCProviderUnavailable("OIDC JWKS endpoint is unavailable") from exc
            if isinstance(jwk_set_type, type) and isinstance(exc, jwk_set_type):
                raise OIDCProviderUnavailable("OIDC JWKS response is invalid") from exc
            if invalid_token_types and isinstance(exc, invalid_token_types):
                raise OIDCAuthenticationError("Invalid bearer token") from exc
            if isinstance(exc, ValueError | KeyError | TypeError):
                raise OIDCAuthenticationError("Invalid bearer token") from exc
            raise OIDCProviderUnavailable("OIDC validation is unavailable") from exc


__all__ = [
    "DEFAULT_TENANT_ID",
    "OIDCAuthenticationError",
    "OIDCProviderUnavailable",
    "OIDCValidator",
    "Principal",
    "current_principal",
    "current_tenant_id",
    "normalize_tenant_id",
    "reset_current_principal",
    "set_current_principal",
]
