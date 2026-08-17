"""Request authentication and production configuration checks."""
import hmac
import ipaddress
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from app.auth import (
    OIDCValidator,
    Principal,
    current_principal,
    normalize_tenant_id,
    reset_current_principal,
    set_current_principal,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

PUBLIC_PATHS = {"/", "/health", "/health/live", "/health/ready"}
DEFAULT_METRICS_PATH = "/internal/metrics"
VIEWER_POST_PATHS = {
    "/api/v1/query",
    "/api/v1/answer",
    "/api/v1/query_simple",
    "/api/v1/answer_simple",
    "/api/v1/query/enhanced",
    "/api/v1/chat",
    "/api/v1/chat/stream",
}
EDITOR_POST_PATHS = {"/api/v1/documents/ingest"}
LOOPBACK_HOST_NAMES = frozenset({"localhost", "localhost.localdomain", ""})


def _is_loopback_host(host: str) -> bool:
    """Return whether a configured bind address only accepts local connections."""
    normalized = host.strip().strip("[]").lower()
    if normalized in LOOPBACK_HOST_NAMES:
        return True
    try:
        # 0.0.0.0 and :: are wildcard binds, not loopback, and is_loopback
        # already reports them as False.
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        # A resolvable hostname other than localhost is treated as remote.
        return False


def required_api_roles(method: str, path: str) -> frozenset[str]:
    """Return explicitly approved roles for an API operation; unknown routes are denied."""
    normalized_method = method.upper()
    if normalized_method in {"GET", "HEAD"}:
        return frozenset({"viewer", "editor", "admin"})
    if normalized_method == "POST" and path in VIEWER_POST_PATHS:
        return frozenset({"viewer", "editor", "admin"})
    if normalized_method == "POST" and path in EDITOR_POST_PATHS:
        return frozenset({"editor", "admin"})
    if normalized_method == "DELETE" and path.startswith("/api/v1/documents/"):
        return frozenset({"admin"})
    return frozenset()


def validate_security_config(config: dict[str, Any]) -> None:
    """Reject unsafe production defaults before serving requests."""
    debug = bool(config.get("app", {}).get("debug", False))
    security = config.get("security", {})
    api_key = security.get("api_key", {})
    oidc = OIDCValidator(config)
    cors = security.get("cors", {})

    if debug:
        # Debug mode skips every check below, including the requirement that
        # some form of authentication exists. That is only acceptable while the
        # API is unreachable from the network, so the bind address decides
        # whether this is a laptop profile or an unauthenticated public service.
        host = str(config.get("app", {}).get("host", "")).strip()
        if not _is_loopback_host(host):
            raise RuntimeError(
                f"app.debug enabled with non-loopback host '{host}'. Debug mode "
                "bypasses authentication and CORS validation; bind to 127.0.0.1 "
                "or set app.debug to false."
            )
        logger.warning(
            "app.debug is enabled: security validation is bypassed and requests "
            "are served as a local administrator. Never use this profile for a "
            "shared or production deployment.",
            extra={"host": host},
        )
        return

    if api_key.get("enabled"):
        value = os.getenv("RAG_API_KEY") or api_key.get("value")
        if not value or str(value).startswith("${"):
            raise RuntimeError("RAG_API_KEY must be configured when API key authentication is enabled")
        normalized = str(value).strip()
        if len(normalized) < 32 or normalized.lower().startswith(
            ("change-me", "test-only", "validation-only")
        ):
            raise RuntimeError("RAG_API_KEY must be a non-placeholder secret of at least 32 characters")

    oidc.validate_configuration()
    if not api_key.get("enabled") and not oidc.enabled:
        raise RuntimeError("Production mode requires API key or OIDC authentication")

    origins = cors.get("allow_origins", [])
    if cors.get("allow_credentials") and "*" in origins:
        raise RuntimeError("CORS cannot use wildcard origins with credentials enabled")

    prometheus = config.get("monitoring", {}).get("prometheus", {})
    # The default must match app.main, which instruments the app when
    # monitoring.prometheus.enabled is absent. Defaulting to False here would
    # exempt exactly the configurations that still expose the endpoint.
    if prometheus.get("enabled", True):
        metrics_token = os.getenv("RAG_METRICS_TOKEN") or prometheus.get("token")
        normalized_metrics_token = str(metrics_token or "").strip()
        if (
            len(normalized_metrics_token) < 32
            or normalized_metrics_token.startswith("${")
            or normalized_metrics_token.lower().startswith(
                ("change-me", "test-only", "validation-only")
            )
        ):
            raise RuntimeError(
                "RAG_METRICS_TOKEN must be an independent non-placeholder secret "
                "of at least 32 characters"
            )
        api_key_value = str(os.getenv("RAG_API_KEY") or api_key.get("value") or "").strip()
        if api_key_value and hmac.compare_digest(normalized_metrics_token, api_key_value):
            raise RuntimeError("RAG_METRICS_TOKEN must be different from RAG_API_KEY")


def authentication_middleware(
    config: dict[str, Any],
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """Authenticate API-key service calls or standard OIDC bearer tokens."""
    security = config.get("security", {})
    api_key = security.get("api_key", {})
    enabled = bool(api_key.get("enabled", False))
    expected = os.getenv("RAG_API_KEY") or api_key.get("value")
    header_name = str(api_key.get("header_name", "X-API-Key"))
    oidc = OIDCValidator(config)
    prometheus = config.get("monitoring", {}).get("prometheus", {})
    metrics_path = str(prometheus.get("path", DEFAULT_METRICS_PATH))
    metrics_header = str(prometheus.get("header_name", "X-Metrics-Token"))
    configured_metrics_token = os.getenv("RAG_METRICS_TOKEN") or prometheus.get("token")
    if isinstance(configured_metrics_token, str) and configured_metrics_token.startswith("${"):
        configured_metrics_token = None
    def service_principal() -> Principal:
        configured_roles = os.getenv("RAG_SERVICE_ROLES", "viewer")
        return Principal(
            subject="rag-api-key-service",
            tenant_id=normalize_tenant_id(os.getenv("RAG_SERVICE_TENANT_ID")),
            roles=frozenset(role.strip() for role in configured_roles.split(",") if role.strip()),
            auth_type="api_key",
        )

    def authorized(principal: Principal, request: Request) -> bool:
        if not request.url.path.startswith("/api/"):
            return True
        return bool(principal.roles & required_api_roles(request.method, request.url.path))

    async def middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Keep metrics on an internal path and allow a dedicated scraper token.
        # The API-key fallback is retained for existing deployments while they
        # provision RAG_METRICS_TOKEN.
        if request.url.path in {metrics_path, "/metrics"}:
            metrics_token = configured_metrics_token or expected
            supplied_metrics_token = request.headers.get(metrics_header, "")
            supplied_api_key = request.headers.get(header_name, "")
            valid_metrics = bool(
                metrics_token
                and hmac.compare_digest(str(supplied_metrics_token), str(metrics_token))
            )
            valid_fallback = bool(
                not configured_metrics_token
                and expected
                and hmac.compare_digest(str(supplied_api_key), str(expected))
            )
            if not (valid_metrics or valid_fallback):
                return JSONResponse(status_code=401, content={"detail": "Metrics authentication required"})
            return await call_next(request)

        principal: Principal | None = None
        supplied = request.headers.get(header_name, "")
        if enabled and expected and hmac.compare_digest(str(supplied), str(expected)):
            principal = service_principal()
        else:
            authorization = request.headers.get("Authorization", "")
            if oidc.enabled and authorization.lower().startswith("bearer "):
                try:
                    principal = await oidc.validate(authorization.split(None, 1)[1])
                except Exception:
                    return JSONResponse(status_code=401, content={"detail": "Invalid bearer token"})

        if principal is None:
            if enabled or oidc.enabled:
                return JSONResponse(status_code=401, content={"detail": "Authentication required"})
            # Explicitly disabled authentication is the loopback-only laptop
            # mode. Preserve its local administrator identity so document
            # management remains usable without pretending to authenticate a
            # service account.
            principal = current_principal()
        if not authorized(principal, request):
            return JSONResponse(status_code=403, content={"detail": "Insufficient role"})

        request.state.principal = principal
        context_token = set_current_principal(principal)
        try:
            response = await call_next(request)
        except BaseException:
            reset_current_principal(context_token)
            raise

        # StreamingResponse bodies are consumed after ``call_next`` returns.
        # Keep the tenant context alive until the body has been fully consumed;
        # otherwise SSE generation can silently fall back to the default tenant.
        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is None:
            reset_current_principal(context_token)
            return response

        async def authenticated_body_iterator() -> AsyncIterator[Any]:
            try:
                async for chunk in body_iterator:
                    yield chunk
            finally:
                reset_current_principal(context_token)

        cast(Any, response).body_iterator = authenticated_body_iterator()
        return response

    return middleware
