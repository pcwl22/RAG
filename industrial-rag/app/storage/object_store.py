"""Small S3-compatible object-storage adapter for distributed uploads."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.auth import normalize_tenant_id
from app.utils.config import get_settings, resolve_object_storage_config

_SAFE_KEY_PART = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SAFE_FAILED_PREFIX = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SAFE_EXTENSION = re.compile(r"^[A-Za-z0-9]{1,16}$")
_UPLOAD_MANIFEST_FILENAME = ".upload-manifest.json"
_MAX_DISPLAY_FILENAME_LENGTH = 255


class ObjectStorageConfigError(ValueError):
    """Raised when S3-compatible storage is required but not usable."""


def _normalized_upload_basename(filename: str | None) -> str:
    basename = str(filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    basename = unicodedata.normalize("NFKC", basename)
    return "".join(
        character
        for character in basename
        if unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
    ).strip(" .\t\r\n") or "upload"


def build_upload_object_key(tenant_id: str, task_id: str, filename: str) -> str:
    """Build a server-owned tenant-isolated key.

    No client-controlled basename is used as storage identity.  The unique task
    path is the identity and only a validated extension is retained so workers
    can select the parser after downloading the opaque ``payload`` object.
    """
    tenant = normalize_tenant_id(tenant_id)
    task = re.sub(r"[^A-Za-z0-9_-]+", "-", str(task_id)).strip(".-")[:96]
    if not task:
        raise ValueError("task_id must not be empty")
    extension = upload_filename_extension(filename)
    storage_name = f"payload.{extension}" if extension else "payload"
    return f"tenants/{tenant}/uploads/{task}/{storage_name}"


def sanitize_upload_filename(filename: str | None) -> str:
    """Return a bounded Unicode display name with paths and controls removed.

    This value is metadata, never a storage key.  Keeping Unicode avoids
    collapsing distinct legal filenames such as ``劳动合同法.pdf`` and
    ``民法典.pdf`` to the same ASCII placeholder.  A safe extension is
    preserved even when an attacker supplies an overlong basename.
    """
    basename = _normalized_upload_basename(filename)

    extension = upload_filename_extension(basename)
    suffix = f".{extension}" if extension else ""
    stem = basename[: -len(suffix)] if suffix else basename
    available = max(1, _MAX_DISPLAY_FILENAME_LENGTH - len(suffix))
    stem = stem[:available].rstrip(" .") or "upload"
    return f"{stem}{suffix}"


def upload_filename_extension(filename: str | None) -> str:
    """Extract a short ASCII extension without trusting the client basename."""
    basename = _normalized_upload_basename(filename)
    suffix = Path(basename).suffix.removeprefix(".")
    return suffix.lower() if _SAFE_EXTENSION.fullmatch(suffix) else ""


def upload_filename_identity(filename: str | None, namespace: str = "") -> str:
    """Hash the full normalized name before display truncation.

    The namespace normally contains the document partition.  This retains the
    existing replace-on-reupload behavior without allowing two overlong names
    with the same first 255 characters to collapse to one source identity.
    """
    material = f"{str(namespace).strip().casefold()}\0{_normalized_upload_basename(filename).casefold()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def validate_upload_object_key(object_key: str) -> None:
    """Reject task keys that were not produced by the upload key contract."""
    parts = object_key.strip("/").split("/")
    if len(parts) != 5 or parts[0] != "tenants" or parts[2] != "uploads":
        raise ValueError("object key is not a service-generated tenant upload key")
    if not _SAFE_KEY_PART.fullmatch(parts[1]) or not _SAFE_KEY_PART.fullmatch(parts[3]):
        raise ValueError("object key contains an unsafe tenant or task component")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", parts[4]) or parts[4] in {".", ".."}:
        raise ValueError("object key contains an unsafe filename component")


def build_upload_manifest_key(object_key: str) -> str:
    """Return the service-owned sidecar key for one uploaded object."""
    normalized = object_key.strip("/")
    validate_upload_object_key(normalized)
    parts = normalized.split("/")
    return "/".join((*parts[:4], _UPLOAD_MANIFEST_FILENAME))


def failed_object_key(object_key: str, failed_prefix: str = "failed") -> str:
    """Map an upload key to a tenant-preserving failure-retention prefix."""
    normalized = object_key.strip("/")
    parts = normalized.split("/")
    validate_upload_object_key(normalized)
    tenant_id = parts[1]
    task_id = parts[3]
    filename = parts[-1]
    prefix = failed_prefix.strip("/") or "failed"
    if not _SAFE_FAILED_PREFIX.fullmatch(prefix):
        raise ValueError("failed prefix must be a single safe path segment")
    return f"tenants/{tenant_id}/{prefix}/{task_id}/{filename}"


class S3ObjectStore:
    """Synchronous S3 adapter; callers choose the appropriate thread/loop."""

    def __init__(self, settings: dict[str, Any], client: Any | None = None) -> None:
        self.settings = settings
        self._validate_settings(settings)
        self._client = client or self._build_client(settings)
        self._health_client = client
        self._health_client_lock = threading.Lock()
        self._health_probe_lock = threading.Lock()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> S3ObjectStore:
        settings = resolve_object_storage_config(config or get_settings())
        cls._validate_settings(settings)
        return cls(settings)

    @staticmethod
    def _validate_settings(settings: dict[str, Any]) -> None:
        if not settings.get("enabled"):
            raise ObjectStorageConfigError("OBJECT_STORAGE_ENABLED must be true")
        required = (
            "endpoint_url",
            "bucket",
            "region",
            "access_key_id",
            "secret_access_key",
        )
        missing = [key for key in required if not str(settings.get(key) or "").strip()]
        if missing:
            raise ObjectStorageConfigError(f"missing S3 settings: {', '.join(missing)}")
        parsed = urlsplit(str(settings["endpoint_url"]))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ObjectStorageConfigError("S3 endpoint_url must be an absolute URL")
        if settings.get("require_tls") and parsed.scheme != "https":
            raise ObjectStorageConfigError("S3 endpoint_url must use https://")
        if settings.get("addressing_style") not in {"path", "virtual"}:
            raise ObjectStorageConfigError("S3 addressing style must be path or virtual")
        if not _SAFE_FAILED_PREFIX.fullmatch(str(settings.get("failed_prefix") or "")):
            raise ObjectStorageConfigError(
                "S3 failed_prefix must be a single safe path segment"
            )

    @staticmethod
    def _build_client(
        settings: dict[str, Any],
        *,
        health_probe: bool = False,
    ) -> Any:
        try:
            boto3 = importlib.import_module("boto3")
            botocore_config = importlib.import_module("botocore.config")
        except ImportError as exc:  # pragma: no cover - depends on runtime image
            raise ObjectStorageConfigError(
                "boto3 is required when OBJECT_STORAGE_ENABLED is true"
            ) from exc

        connect_timeout = int(settings.get("connect_timeout_seconds", 3))
        read_timeout = int(settings.get("read_timeout_seconds", 15))
        retries = {"max_attempts": 3, "mode": "standard"}
        if health_probe:
            # One short attempt keeps readiness responsive without cancelling a
            # still-running boto thread and accumulating orphaned health calls.
            connect_timeout = min(connect_timeout, 2)
            read_timeout = min(read_timeout, 2)
            retries = {"total_max_attempts": 1, "mode": "standard"}
        client_config = botocore_config.Config(
            signature_version="s3v4",
            s3={"addressing_style": settings["addressing_style"]},
            retries=retries,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_pool_connections=int(settings.get("max_pool_connections", 20)),
        )
        kwargs: dict[str, Any] = {
            "service_name": "s3",
            "endpoint_url": settings["endpoint_url"],
            "region_name": settings["region"],
            "aws_access_key_id": settings["access_key_id"],
            "aws_secret_access_key": settings["secret_access_key"],
            "config": client_config,
        }
        if settings.get("ca_bundle"):
            kwargs["verify"] = settings["ca_bundle"]
        return boto3.client(**kwargs)

    @property
    def bucket(self) -> str:
        return str(self.settings["bucket"])

    @property
    def failed_prefix(self) -> str:
        return str(self.settings.get("failed_prefix") or "failed")

    def upload_file(self, local_path: str | Path, object_key: str) -> None:
        self._client.upload_file(str(local_path), self.bucket, object_key)

    def put_json(self, object_key: str, payload: dict[str, Any]) -> None:
        """Persist a small server-generated upload manifest."""
        validate_upload_object_key(object_key)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=body,
            ContentType="application/json",
        )

    def get_json(self, object_key: str) -> dict[str, Any]:
        """Read and decode a server-generated upload manifest."""
        validate_upload_object_key(object_key)
        response = self._client.get_object(Bucket=self.bucket, Key=object_key)
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise ValueError("object storage returned an invalid manifest body")
        try:
            raw_body = body.read()
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if not isinstance(raw_body, bytes | bytearray):
            raise ValueError("object storage returned an invalid manifest payload")
        payload = json.loads(bytes(raw_body).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("upload manifest must be a JSON object")
        return payload

    def download_file(self, object_key: str, local_path: str | Path) -> None:
        self._client.download_file(self.bucket, object_key, str(local_path))

    def delete_object(self, object_key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=object_key)

    def move_to_failed(self, object_key: str) -> str:
        destination = failed_object_key(object_key, self.failed_prefix)
        self._client.copy_object(
            Bucket=self.bucket,
            CopySource={"Bucket": self.bucket, "Key": object_key},
            Key=destination,
        )
        self.delete_object(object_key)
        return destination

    def check_health(self) -> bool:
        if not self._health_probe_lock.acquire(blocking=False):
            return False
        try:
            if self._health_client is None:
                with self._health_client_lock:
                    if self._health_client is None:
                        self._health_client = self._build_client(
                            self.settings,
                            health_probe=True,
                        )
            health_client = self._health_client
            if health_client is None:  # Defensive guard for custom client factories.
                raise RuntimeError("object storage health client is unavailable")
            health_client.head_bucket(Bucket=self.bucket)
            return True
        finally:
            self._health_probe_lock.release()

    def list_upload_manifest_keys(
        self,
        max_items: int = 100,
        continuation_token: str | None = None,
    ) -> tuple[list[str], str | None]:
        """Scan one bounded S3 page and return its next opaque cursor."""
        limit = int(max_items)
        if limit < 1 or limit > 1000:
            raise ValueError("max_items must be between 1 and 1000")
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Prefix": "tenants/",
            "MaxKeys": limit,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        page = self._client.list_objects_v2(**kwargs)
        keys: list[str] = []
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key.endswith(f"/{_UPLOAD_MANIFEST_FILENAME}"):
                continue
            try:
                validate_upload_object_key(key)
            except ValueError:
                continue
            keys.append(key)
        next_token = page.get("NextContinuationToken") if page.get("IsTruncated") else None
        return keys, str(next_token) if next_token else None


def get_object_store(config: dict[str, Any] | None = None) -> S3ObjectStore:
    """Construct a validated store for one request/worker operation."""
    return S3ObjectStore.from_config(config)


__all__ = [
    "ObjectStorageConfigError",
    "S3ObjectStore",
    "build_upload_object_key",
    "build_upload_manifest_key",
    "failed_object_key",
    "get_object_store",
    "sanitize_upload_filename",
    "upload_filename_extension",
    "upload_filename_identity",
    "validate_upload_object_key",
]
