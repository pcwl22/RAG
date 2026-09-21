"""Integrity contract for durable upload completion receipts."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from app.auth import normalize_tenant_id

COMPLETION_RECEIPT_VERSION = 2
_IMMUTABLE_MANIFEST_FIELDS = (
    "version",
    "task_id",
    "object_key",
    "tenant_id",
    "filename",
    "partition",
    "metadata",
    "created_at_unix",
)


def manifest_identity_sha256(manifest: dict[str, Any]) -> str:
    """Hash the immutable, server-owned upload intent fields."""
    identity = {field: manifest.get(field) for field in _IMMUTABLE_MANIFEST_FIELDS}
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_completion_receipt(
    manifest: dict[str, Any],
    *,
    task_id: str,
    tenant_id: str,
    object_key: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build a receipt bound to one immutable upload intent and DB result."""
    return {
        "version": COMPLETION_RECEIPT_VERSION,
        "task_id": task_id,
        "tenant_id": normalize_tenant_id(tenant_id),
        "object_key": object_key,
        "manifest_identity_sha256": manifest_identity_sha256(manifest),
        "result": {
            "document_id": result.get("document_id"),
            "source_key": result.get("source_key"),
            "total_chunks": result.get("total_chunks"),
            "status": result.get("status"),
        },
    }


def parse_completion_receipt(
    manifest: dict[str, Any],
    *,
    expected_task_id: str,
    expected_tenant_id: str,
    expected_object_key: str,
) -> dict[str, Any] | None:
    """Validate a receipt and return its normalized committed result."""
    receipt = manifest.get("completion_receipt")
    if receipt is None:
        return None
    if not isinstance(receipt, dict):
        raise ValueError("completion receipt is invalid")
    if receipt.get("version") != COMPLETION_RECEIPT_VERSION:
        raise ValueError("completion receipt version is invalid")
    if receipt.get("task_id") != expected_task_id:
        raise ValueError("completion receipt task id is invalid")
    receipt_tenant_id = receipt.get("tenant_id")
    if not isinstance(receipt_tenant_id, str) or normalize_tenant_id(
        receipt_tenant_id
    ) != normalize_tenant_id(expected_tenant_id):
        raise ValueError("completion receipt tenant is invalid")
    if receipt.get("object_key") != expected_object_key:
        raise ValueError("completion receipt object key is invalid")
    expected_fingerprint = manifest_identity_sha256(manifest)
    actual_fingerprint = receipt.get("manifest_identity_sha256")
    if not isinstance(actual_fingerprint, str) or not hmac.compare_digest(
        actual_fingerprint,
        expected_fingerprint,
    ):
        raise ValueError("completion receipt manifest identity is invalid")

    result = receipt.get("result")
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise ValueError("completion receipt result is invalid")
    document_id = result.get("document_id")
    source_key = result.get("source_key")
    total_chunks = result.get("total_chunks")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("completion receipt document id is invalid")
    if not isinstance(source_key, str) or not source_key:
        raise ValueError("completion receipt source key is invalid")
    if (
        not isinstance(total_chunks, int)
        or isinstance(total_chunks, bool)
        or total_chunks < 1
    ):
        raise ValueError("completion receipt chunk count is invalid")
    return {
        "document_id": document_id,
        "source_key": source_key,
        "total_chunks": total_chunks,
        "status": "completed",
    }


__all__ = [
    "COMPLETION_RECEIPT_VERSION",
    "build_completion_receipt",
    "manifest_identity_sha256",
    "parse_completion_receipt",
]
