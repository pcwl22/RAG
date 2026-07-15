"""Request authentication and production configuration checks."""
import hmac
import os
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

PUBLIC_PATHS = {"/", "/health", "/health/live", "/health/ready", "/docs", "/openapi.json", "/redoc"}


def validate_security_config(config: dict[str, Any]) -> None:
    """Reject unsafe production defaults before serving requests."""
    debug = bool(config.get("app", {}).get("debug", False))
    security = config.get("security", {})
    api_key = security.get("api_key", {})
    jwt = security.get("jwt", {})
    cors = security.get("cors", {})

    if debug:
        return

    if api_key.get("enabled"):
        value = os.getenv("RAG_API_KEY") or api_key.get("value")
        if not value or str(value).startswith("${"):
            raise RuntimeError("RAG_API_KEY must be configured when API key authentication is enabled")

    if jwt.get("enabled"):
        secret = os.getenv("JWT_SECRET_KEY") or jwt.get("secret_key")
        if not secret or str(secret).startswith("${") or secret == "your-secret-key-change-in-production":
            raise RuntimeError("JWT_SECRET_KEY must be changed when JWT authentication is enabled")

    origins = cors.get("allow_origins", [])
    if cors.get("allow_credentials") and "*" in origins:
        raise RuntimeError("CORS cannot use wildcard origins with credentials enabled")


def authentication_middleware(config: dict[str, Any]):
    """Build a lightweight API-key middleware from the current configuration."""
    security = config.get("security", {})
    api_key = security.get("api_key", {})
    enabled = bool(api_key.get("enabled", False))
    expected = os.getenv("RAG_API_KEY") or api_key.get("value")
    header_name = str(api_key.get("header_name", "X-API-Key"))

    async def middleware(request: Request, call_next):
        if enabled and request.url.path.startswith("/api/") and request.url.path not in PUBLIC_PATHS:
            supplied = request.headers.get(header_name, "")
            if not expected or not hmac.compare_digest(str(supplied), str(expected)):
                return JSONResponse(status_code=401, content={"detail": "Authentication required"})
        return await call_next(request)

    return middleware
