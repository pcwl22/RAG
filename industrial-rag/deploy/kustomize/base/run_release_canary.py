"""Fail-closed functional release canary executed inside the candidate API image.

The script intentionally uses only the Python standard library so it does not
add a second dependency contract to the production image.  Secrets are read
from the dedicated versioned canary Secret and are never printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class CanaryError(RuntimeError):
    """A release-blocking canary contract failure."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward OAuth credentials or bearer tokens across redirects."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect())
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_PARTITION = "release-canary"
_SOURCE_ID_PREFIX = "release-canary-functional-fixture"


@dataclass(frozen=True)
class CanarySettings:
    token_url: str
    client_auth_method: str
    primary_client_id: str
    primary_client_secret: str
    secondary_client_id: str
    secondary_client_secret: str
    upload_content: str
    no_answer_query: str
    no_answer_expected_text: str
    scope: str = ""
    audience: str = ""


def _required(environ: Mapping[str, str], name: str) -> str:
    value = str(environ.get(name, "")).strip()
    if not value:
        raise CanaryError(f"{name} is required")
    return value


def load_settings(environ: Mapping[str, str] | None = None) -> CanarySettings:
    """Load and validate the protected functional-canary contract."""
    values = os.environ if environ is None else environ
    token_url = _required(values, "CANARY_OIDC_TOKEN_URL")
    parsed_token_url = urllib.parse.urlsplit(token_url)
    if parsed_token_url.scheme != "https" or not parsed_token_url.hostname:
        raise CanaryError("CANARY_OIDC_TOKEN_URL must be an absolute https:// URL")

    auth_method = _required(values, "CANARY_OIDC_CLIENT_AUTH_METHOD")
    if auth_method not in {"client_secret_basic", "client_secret_post"}:
        raise CanaryError(
            "CANARY_OIDC_CLIENT_AUTH_METHOD must be client_secret_basic or client_secret_post"
        )

    primary_id = _required(values, "CANARY_PRIMARY_CLIENT_ID")
    primary_secret = _required(values, "CANARY_PRIMARY_CLIENT_SECRET")
    secondary_id = _required(values, "CANARY_SECONDARY_CLIENT_ID")
    secondary_secret = _required(values, "CANARY_SECONDARY_CLIENT_SECRET")
    if primary_id == secondary_id:
        raise CanaryError("canary OIDC client IDs must be distinct")
    if primary_secret == secondary_secret:
        raise CanaryError("canary OIDC client secrets must be distinct")

    encoded_sample = _required(values, "CANARY_UPLOAD_CONTENT_B64")
    try:
        sample_bytes = base64.b64decode(encoded_sample, validate=True)
        upload_content = sample_bytes.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise CanaryError("CANARY_UPLOAD_CONTENT_B64 must contain base64-encoded UTF-8") from exc
    if len(sample_bytes) < 80 or len(sample_bytes) > 256 * 1024 or "\x00" in upload_content:
        raise CanaryError("canary upload sample must be 80 bytes to 256 KiB of UTF-8 text")

    return CanarySettings(
        token_url=token_url,
        client_auth_method=auth_method,
        primary_client_id=primary_id,
        primary_client_secret=primary_secret,
        secondary_client_id=secondary_id,
        secondary_client_secret=secondary_secret,
        upload_content=upload_content,
        no_answer_query=_required(values, "CANARY_NO_ANSWER_QUERY"),
        no_answer_expected_text=_required(values, "CANARY_NO_ANSWER_EXPECTED_TEXT"),
        scope=str(values.get("CANARY_OIDC_SCOPE", "")).strip(),
        audience=str(values.get("CANARY_OIDC_AUDIENCE", "")).strip(),
    )


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    body: bytes | None = None,
    timeout: float,
) -> tuple[int, Mapping[str, str], bytes]:
    request = urllib.request.Request(
        url,
        data=body,
        headers=dict(headers or {}),
        method=method,
    )
    try:
        response = _OPENER.open(request, timeout=timeout)
        status = int(response.status)
        response_headers = response.headers
        payload = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        response_headers = exc.headers
        payload = exc.read(_MAX_RESPONSE_BYTES + 1)
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        raise CanaryError(
            f"request transport failed for {urllib.parse.urlsplit(url).path}"
        ) from exc
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise CanaryError("canary response exceeded 8 MiB")
    return status, response_headers, payload


def _json_object(payload: bytes, check: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanaryError(f"{check} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise CanaryError(f"{check} returned a non-object JSON document")
    return value


def _expect_status(status: int, expected: int, check: str) -> None:
    if status != expected:
        raise CanaryError(f"{check} returned HTTP {status}, expected {expected}")


def _token(settings: CanarySettings, client_id: str, client_secret: str, timeout: float) -> str:
    form: dict[str, str] = {"grant_type": "client_credentials"}
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    if settings.scope:
        form["scope"] = settings.scope
    if settings.audience:
        form["audience"] = settings.audience
    if settings.client_auth_method == "client_secret_basic":
        encoded = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode("ascii")
        headers["Authorization"] = f"Basic {encoded}"
    else:
        form["client_id"] = client_id
        form["client_secret"] = client_secret
    status, _, payload = _request(
        settings.token_url,
        method="POST",
        headers=headers,
        body=urllib.parse.urlencode(form).encode("ascii"),
        timeout=timeout,
    )
    _expect_status(status, 200, "OIDC client-credentials token request")
    document = _json_object(payload, "OIDC client-credentials token request")
    token = document.get("access_token")
    if not isinstance(token, str) or not token.strip():
        raise CanaryError("OIDC token response is missing access_token")
    token_type = str(document.get("token_type", "Bearer"))
    if token_type.lower() != "bearer":
        raise CanaryError("OIDC token response did not return a bearer token")
    return token


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def wait_until_ready(api_base_url: str, *, timeout: float, request_timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_status = 0
    while time.monotonic() < deadline:
        try:
            status, _, _ = _request(
                f"{api_base_url}/health/ready",
                timeout=min(request_timeout, 5.0),
            )
            last_status = status
            if status == 200:
                return
        except CanaryError:
            pass
        time.sleep(2)
    raise CanaryError(f"candidate API readiness timed out (last HTTP status {last_status})")


def _multipart_upload(
    api_base_url: str,
    token: str,
    content: str,
    marker: str,
    *,
    timeout: float,
) -> str:
    boundary = "----industrial-release-canary-" + uuid.uuid4().hex
    # ``source_id`` drives replacement semantics. A fixed value could replace
    # a legitimate tenant document using the same public API metadata, so bind
    # it to this unpredictable run marker instead.
    source_metadata = json.dumps(
        {"source_id": f"{_SOURCE_ID_PREFIX}-{marker}"},
        separators=(",", ":"),
    )
    marked_content = f"{content.rstrip()}\n\n发布验证标记：{marker}\n"
    fields = [
        ("partition", None, _PARTITION.encode("utf-8"), None),
        ("metadata", None, source_metadata.encode("utf-8"), None),
        (
            "file",
            f"release-canary-{marker}.txt",
            marked_content.encode("utf-8"),
            "text/plain; charset=utf-8",
        ),
    ]
    chunks: list[bytes] = []
    for field_name, filename, field_body, content_type in fields:
        chunks.append(f"--{boundary}\r\n".encode("ascii"))
        disposition = f'Content-Disposition: form-data; name="{field_name}"'
        if filename:
            disposition += f'; filename="{filename}"'
        chunks.append((disposition + "\r\n").encode("utf-8"))
        if content_type:
            chunks.append(f"Content-Type: {content_type}\r\n".encode("ascii"))
        chunks.append(b"\r\n")
        chunks.append(field_body)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    headers = _auth(token)
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    status, _, payload = _request(
        f"{api_base_url}/api/v1/documents/ingest",
        method="POST",
        headers=headers,
        body=b"".join(chunks),
        timeout=timeout,
    )
    _expect_status(status, 200, "authenticated document upload")
    task_id = _json_object(payload, "authenticated document upload").get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise CanaryError("authenticated document upload returned no task_id")
    return task_id


def _cross_tenant_denied(
    api_base_url: str,
    secondary_token: str,
    task_id: str,
    *,
    timeout: float,
) -> None:
    status, _, _ = _request(
        f"{api_base_url}/api/v1/documents/status/{urllib.parse.quote(task_id, safe='')}",
        headers=_auth(secondary_token),
        timeout=timeout,
    )
    _expect_status(status, 404, "cross-tenant task lookup")


def _wait_for_ingest(
    api_base_url: str,
    primary_token: str,
    task_id: str,
    *,
    timeout: float,
    request_timeout: float,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, _, payload = _request(
            f"{api_base_url}/api/v1/documents/status/{urllib.parse.quote(task_id, safe='')}",
            headers=_auth(primary_token),
            timeout=request_timeout,
        )
        if status == 503:
            time.sleep(2)
            continue
        _expect_status(status, 200, "authenticated upload status")
        document = _json_object(payload, "authenticated upload status")
        state = str(document.get("status", ""))
        if state == "completed":
            if int(document.get("total_chunks", 0) or 0) < 1:
                raise CanaryError("completed canary upload produced no chunks")
            document_id = document.get("document_id")
            if not isinstance(document_id, str) or not document_id.strip():
                raise CanaryError("completed canary upload returned no tenant-scoped document_id")
            return document_id
        if state in {"failed", "unknown"}:
            raise CanaryError(f"canary upload entered terminal state {state!r}")
        time.sleep(2)
    raise CanaryError("canary upload processing timed out")


def _post_json(
    url: str,
    token: str,
    document: Mapping[str, Any],
    *,
    timeout: float,
) -> tuple[int, Mapping[str, str], bytes]:
    headers = _auth(token)
    headers["Content-Type"] = "application/json"
    return _request(
        url,
        method="POST",
        headers=headers,
        body=json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        timeout=timeout,
    )


def _authorized_retrieval(
    api_base_url: str,
    token: str,
    marker: str,
    *,
    timeout: float,
) -> None:
    status, _, payload = _post_json(
        f"{api_base_url}/api/v1/query",
        token,
        {
            "query": marker,
            "partition": _PARTITION,
            "top_k": 10,
            "similarity_threshold": 0,
            "enable_rerank": False,
        },
        timeout=timeout,
    )
    _expect_status(status, 200, "authenticated retrieval")
    document = _json_object(payload, "authenticated retrieval")
    results = document.get("documents")
    if not isinstance(results, list) or not any(
        isinstance(item, dict) and marker in str(item.get("content", "")) for item in results
    ):
        raise CanaryError("authenticated retrieval did not return the uploaded canary marker")


def _cross_tenant_retrieval_empty(
    api_base_url: str,
    token: str,
    marker: str,
    *,
    timeout: float,
) -> None:
    status, _, payload = _post_json(
        f"{api_base_url}/api/v1/query",
        token,
        {
            "query": marker,
            "partition": _PARTITION,
            "top_k": 10,
            "similarity_threshold": 0,
            "enable_rerank": False,
        },
        timeout=timeout,
    )
    _expect_status(status, 200, "cross-tenant retrieval")
    document = _json_object(payload, "cross-tenant retrieval")
    if document.get("total") != 0 or document.get("documents") != []:
        raise CanaryError("secondary tenant retrieved the primary tenant canary fixture")


def _no_answer(
    api_base_url: str,
    token: str,
    settings: CanarySettings,
    *,
    timeout: float,
) -> None:
    status, _, payload = _post_json(
        f"{api_base_url}/api/v1/answer",
        token,
        {
            "query": settings.no_answer_query,
            "partition": _PARTITION,
            "top_k": 5,
            "similarity_threshold": 0,
            "enable_rerank": False,
            "stream": False,
        },
        timeout=timeout,
    )
    _expect_status(status, 200, "no-answer contract")
    document = _json_object(payload, "no-answer contract")
    if document.get("sources") != []:
        raise CanaryError("no-answer contract unexpectedly returned sources")
    if settings.no_answer_expected_text not in str(document.get("answer", "")):
        raise CanaryError("no-answer contract did not return the approved refusal text")


def parse_sse_events(payload: bytes) -> list[dict[str, Any]]:
    """Parse the bounded SSE response and reject malformed data events."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanaryError("SSE response was not UTF-8") from exc
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CanaryError("SSE response contained invalid JSON") from exc
        if not isinstance(event, dict):
            raise CanaryError("SSE data event was not an object")
        events.append(event)
    if not events:
        raise CanaryError("SSE response contained no data events")
    return events


def _sse(
    api_base_url: str,
    token: str,
    marker: str,
    *,
    timeout: float,
) -> None:
    status, headers, payload = _post_json(
        f"{api_base_url}/api/v1/answer",
        token,
        {
            "query": marker,
            "partition": _PARTITION,
            "top_k": 10,
            "similarity_threshold": 0,
            "enable_rerank": False,
            "stream": True,
        },
        timeout=timeout,
    )
    _expect_status(status, 200, "SSE answer")
    if not str(headers.get("Content-Type", "")).lower().startswith("text/event-stream"):
        raise CanaryError("SSE answer returned the wrong content type")
    events = parse_sse_events(payload)
    if any(event.get("type") == "error" for event in events):
        raise CanaryError("SSE answer emitted an error event")
    sources = [event.get("data") for event in events if event.get("type") == "sources"]
    if not any(
        isinstance(items, list)
        and any(isinstance(item, dict) and marker in str(item.get("content", "")) for item in items)
        for items in sources
    ):
        raise CanaryError("SSE answer did not bind the uploaded canary source")
    if not any(event.get("type") == "chunk" and str(event.get("data", "")) for event in events):
        raise CanaryError("SSE answer emitted no answer chunks")
    done = [event for event in events if event.get("type") == "done"]
    if not done or any(event.get("error") for event in done):
        raise CanaryError("SSE answer did not complete successfully")


def _cleanup_document(
    api_base_url: str,
    token: str,
    document_id: str,
    *,
    timeout: float,
) -> None:
    url = f"{api_base_url}/api/v1/documents/{urllib.parse.quote(document_id, safe='')}"
    status, _, _ = _request(
        url,
        method="DELETE",
        headers=_auth(token),
        timeout=timeout,
    )
    _expect_status(status, 200, "canary fixture cleanup")
    status, _, _ = _request(
        url,
        method="DELETE",
        headers=_auth(token),
        timeout=timeout,
    )
    _expect_status(status, 404, "canary fixture cleanup verification")


def run_canary(
    settings: CanarySettings,
    *,
    api_base_url: str,
    readiness_timeout: float,
    ingest_timeout: float,
    request_timeout: float,
) -> list[str]:
    api_base_url = api_base_url.rstrip("/")
    wait_until_ready(
        api_base_url,
        timeout=readiness_timeout,
        request_timeout=request_timeout,
    )
    primary_token = _token(
        settings,
        settings.primary_client_id,
        settings.primary_client_secret,
        request_timeout,
    )
    secondary_token = _token(
        settings,
        settings.secondary_client_id,
        settings.secondary_client_secret,
        request_timeout,
    )
    if primary_token == secondary_token:
        raise CanaryError("OIDC canary clients returned the same bearer token")

    marker = "industrial_rag_release_canary_" + uuid.uuid4().hex
    document_id: str | None = None
    check_error: CanaryError | None = None
    cleanup_error: CanaryError | None = None
    try:
        task_id = _multipart_upload(
            api_base_url,
            primary_token,
            settings.upload_content,
            marker,
            timeout=request_timeout,
        )
        _cross_tenant_denied(
            api_base_url,
            secondary_token,
            task_id,
            timeout=request_timeout,
        )
        document_id = _wait_for_ingest(
            api_base_url,
            primary_token,
            task_id,
            timeout=ingest_timeout,
            request_timeout=request_timeout,
        )
        _authorized_retrieval(api_base_url, primary_token, marker, timeout=request_timeout)
        _cross_tenant_retrieval_empty(
            api_base_url,
            secondary_token,
            marker,
            timeout=request_timeout,
        )
        _no_answer(api_base_url, primary_token, settings, timeout=request_timeout)
        _sse(api_base_url, primary_token, marker, timeout=request_timeout)
    except CanaryError as exc:
        check_error = exc
    finally:
        if document_id is not None:
            try:
                _cleanup_document(
                    api_base_url,
                    primary_token,
                    document_id,
                    timeout=request_timeout,
                )
            except CanaryError as exc:
                cleanup_error = exc
    if check_error is not None and cleanup_error is not None:
        raise CanaryError(f"{check_error}; fixture cleanup also failed") from cleanup_error
    if check_error is not None:
        raise check_error
    if cleanup_error is not None:
        raise cleanup_error
    return [
        "readiness",
        "oidc-client-credentials",
        "authenticated-upload",
        "cross-tenant-task-denied",
        "authenticated-retrieval",
        "cross-tenant-retrieval-empty",
        "no-answer",
        "sse-answer",
        "fixture-cleanup",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--readiness-timeout", type=float, default=240)
    parser.add_argument("--ingest-timeout", type=float, default=480)
    parser.add_argument("--request-timeout", type=float, default=60)
    args = parser.parse_args()
    try:
        settings = load_settings()
        checks = run_canary(
            settings,
            api_base_url=args.api_base_url,
            readiness_timeout=args.readiness_timeout,
            ingest_timeout=args.ingest_timeout,
            request_timeout=args.request_timeout,
        )
    except CanaryError as exc:
        raise SystemExit(f"release canary failed: {exc}") from exc
    print(json.dumps({"passed": True, "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
