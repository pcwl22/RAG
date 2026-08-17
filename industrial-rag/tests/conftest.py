"""Pytest configuration for the migrated app package."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Importing app.main calls get_settings(), which calls load_dotenv() and injects
# the developer .env into the process. Authentication code then reads those
# values straight from os.environ, so a local .env silently decides whether OIDC
# is enabled or an API key exists, and tests pass or fail based on the machine
# they run on.
#
# load_dotenv() never overwrites a variable that is already set, so pinning a
# value here permanently shields the test run from .env. This hook runs before
# collection, and therefore before any test module imports app.main.
#
# Authentication variables are inputs to the code under test: several tests
# build a config with oidc.enabled false and assert on validate_security_config,
# and an ambient OIDC_ENABLED=true would silently replace that premise. They are
# pinned unconditionally. Tests needing authentication enabled use monkeypatch,
# which applies after this hook and is unaffected.
_PINNED_AUTH_ENV = {
    "OIDC_ENABLED": "false",
    "OIDC_ISSUER": "",
    "OIDC_AUDIENCE": "",
    "OIDC_JWKS_URL": "",
    "RAG_API_KEY": "",
    "RAG_METRICS_TOKEN": "",
    "RAG_SERVICE_ROLES": "viewer,editor,admin",
    "RAG_SERVICE_TENANT_ID": "00000000-0000-0000-0000-000000000001",
}

# Which profile to exercise is a legitimate caller choice, so this one keeps
# setdefault semantics. At this point os.environ holds only what the caller
# exported, so `RAG_ENV=base pytest ...` is honoured while .env cannot decide it.
# laptop keeps app.debug true, which is what the suite is written against.
_TEST_ENV_DEFAULTS = {
    "RAG_ENV": "laptop",
}

# Variables that must be absent rather than empty. Conda ships SSL_CERT_FILE=""
# and an empty path makes httpx raise FileNotFoundError while building its TLS
# context. Only a value that cannot work is removed, so a real bundle path set
# by the caller is preserved.
_TLS_PATH_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")


def pytest_configure(config):
    """Pin authentication-related environment before any app module is imported."""
    os.environ.update(_PINNED_AUTH_ENV)

    for name, value in _TEST_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)

    for name in _TLS_PATH_VARS:
        value = os.environ.get(name)
        if value is not None and (not value.strip() or not Path(value).is_file()):
            del os.environ[name]
