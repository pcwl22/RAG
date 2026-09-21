from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import app.storage.object_store as object_store_module
from app.storage.object_store import (
    S3ObjectStore,
    build_upload_manifest_key,
    build_upload_object_key,
    failed_object_key,
    sanitize_upload_filename,
    upload_filename_identity,
    validate_upload_object_key,
)


class FakeS3Client:
    def __init__(self):
        self.calls = []

    def upload_file(self, local_path, bucket, key):
        self.calls.append(("upload", local_path, bucket, key))

    def download_file(self, bucket, key, local_path):
        self.calls.append(("download", bucket, key, local_path))
        Path(local_path).write_text("downloaded", encoding="utf-8")

    def delete_object(self, **kwargs):
        self.calls.append(("delete", kwargs))

    def put_object(self, **kwargs):
        self.calls.append(("put", kwargs))

    def get_object(self, **kwargs):
        import io

        self.calls.append(("get", kwargs))
        return {"Body": io.BytesIO(b'{"ok": true}')}

    def copy_object(self, **kwargs):
        self.calls.append(("copy", kwargs))

    def head_bucket(self, **kwargs):
        self.calls.append(("head", kwargs))


def _settings():
    return {
        "enabled": True,
        "endpoint_url": "https://s3.example.test",
        "bucket": "rag",
        "region": "us-east-1",
        "access_key_id": "access",
        "secret_access_key": "secret-123456789",
        "require_tls": True,
        "addressing_style": "path",
        "failed_prefix": "failed",
    }


def test_upload_key_is_server_owned_and_tenant_isolated():
    tenant_a = "00000000-0000-0000-0000-00000000000a"
    tenant_b = "00000000-0000-0000-0000-00000000000b"
    key = build_upload_object_key(
        tenant_a, "task-123", r"C:\private\contracts\agreement.txt"
    )

    assert key.startswith(f"tenants/{tenant_a}/uploads/task-123/")
    assert "private" not in key
    assert key.endswith("payload.txt")
    assert build_upload_object_key(tenant_b, "task-123", "agreement.txt") != key
    assert sanitize_upload_filename(r"C:\private\contracts\agreement.txt") == "agreement.txt"


def test_unicode_display_names_do_not_control_storage_identity_and_keep_extension():
    tenant = "00000000-0000-0000-0000-00000000000a"
    first = sanitize_upload_filename("合同/劳动合同法.pdf")
    second = sanitize_upload_filename("合同/民法典.pdf")
    overlong = sanitize_upload_filename(f"{'法' * 400}.docx")

    assert first == "劳动合同法.pdf"
    assert second == "民法典.pdf"
    assert first != second
    assert overlong.endswith(".docx")
    assert len(overlong) <= 255
    assert build_upload_object_key(tenant, "task-a", first).endswith("/payload.pdf")
    assert build_upload_object_key(tenant, "task-b", second).endswith("/payload.pdf")
    assert "劳动合同法" not in build_upload_object_key(tenant, "task-a", first)
    assert upload_filename_identity(f"{'法' * 300}甲.txt", "general") != (
        upload_filename_identity(f"{'法' * 300}乙.txt", "general")
    )


def test_upload_key_contract_rejects_unsafe_failure_paths():
    validate_upload_object_key("tenants/tenant-a/uploads/task-123/agreement.txt")

    try:
        failed_object_key(
            "tenants/tenant-a/uploads/task-123/agreement.txt",
            "../retained",
        )
    except ValueError as exc:
        assert "safe path segment" in str(exc)
    else:
        raise AssertionError("unsafe failure prefix must be rejected")

    try:
        validate_upload_object_key("tenants/tenant-a/../task-123/agreement.txt")
    except ValueError as exc:
        assert "service-generated" in str(exc) or "unsafe" in str(exc)
    else:
        raise AssertionError("unsafe object key must be rejected")


def test_failed_key_preserves_tenant_and_changes_prefix():
    key = "tenants/tenant-a/uploads/task-123/agreement.txt"
    assert failed_object_key(key) == "tenants/tenant-a/failed/task-123/agreement.txt"
    assert build_upload_manifest_key(key) == "tenants/tenant-a/uploads/task-123/.upload-manifest.json"


def test_s3_store_upload_download_and_failure_retention():
    client = FakeS3Client()
    store = S3ObjectStore(_settings(), client=client)
    source = Path("sample.txt")

    store.upload_file(source, "tenants/tenant-a/uploads/task/sample.txt")
    store.put_json("tenants/tenant-a/uploads/task/.upload-manifest.json", {"ok": True})
    assert store.get_json("tenants/tenant-a/uploads/task/.upload-manifest.json") == {"ok": True}
    store.download_file("tenants/tenant-a/uploads/task/sample.txt", source)
    failed = store.move_to_failed("tenants/tenant-a/uploads/task/sample.txt")

    assert failed == "tenants/tenant-a/failed/task/sample.txt"
    assert [call[0] for call in client.calls] == [
        "upload", "put", "get", "download", "copy", "delete"
    ]


def test_s3_client_receives_transport_timeouts_and_ca_bundle(monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    def fake_client(**kwargs):
        captured["client"] = kwargs
        return object()

    def fake_import(name):
        if name == "boto3":
            return SimpleNamespace(client=fake_client)
        if name == "botocore.config":
            return SimpleNamespace(Config=FakeConfig)
        raise ImportError(name)

    monkeypatch.setattr(object_store_module.importlib, "import_module", fake_import)
    settings = {
        **_settings(),
        "ca_bundle": "/etc/ssl/certs/internal-ca.pem",
        "connect_timeout_seconds": 4,
        "read_timeout_seconds": 19,
        "max_pool_connections": 31,
    }

    S3ObjectStore(settings)

    assert captured["config"] == {
        "signature_version": "s3v4",
        "s3": {"addressing_style": "path"},
        "retries": {"max_attempts": 3, "mode": "standard"},
        "connect_timeout": 4,
        "read_timeout": 19,
        "max_pool_connections": 31,
    }
    assert captured["client"]["verify"] == "/etc/ssl/certs/internal-ca.pem"
    assert captured["client"]["endpoint_url"] == "https://s3.example.test"


def test_s3_health_uses_reusable_single_attempt_short_timeout_client(monkeypatch):
    configs = []
    clients = []

    class FakeConfig:
        def __init__(self, **kwargs):
            configs.append(kwargs)

    class Client:
        def __init__(self):
            self.health_calls = []

        def head_bucket(self, **kwargs):
            self.health_calls.append(kwargs)

    def fake_client(**_kwargs):
        client = Client()
        clients.append(client)
        return client

    def fake_import(name):
        if name == "boto3":
            return SimpleNamespace(client=fake_client)
        if name == "botocore.config":
            return SimpleNamespace(Config=FakeConfig)
        raise ImportError(name)

    monkeypatch.setattr(object_store_module.importlib, "import_module", fake_import)
    store = S3ObjectStore(
        {
            **_settings(),
            "connect_timeout_seconds": 4,
            "read_timeout_seconds": 19,
            "max_pool_connections": 31,
        }
    )

    assert len(configs) == 1
    assert store.check_health() is True
    assert store.check_health() is True
    assert len(configs) == 2
    assert configs[1] == {
        "signature_version": "s3v4",
        "s3": {"addressing_style": "path"},
        "retries": {"total_max_attempts": 1, "mode": "standard"},
        "connect_timeout": 2,
        "read_timeout": 2,
        "max_pool_connections": 31,
    }
    assert clients[1].health_calls == [{"Bucket": "rag"}, {"Bucket": "rag"}]


def test_s3_health_does_not_queue_behind_a_stuck_probe():
    started = Event()
    release = Event()
    results = []

    class BlockingClient:
        def head_bucket(self, **_kwargs):
            started.set()
            release.wait(timeout=5)

    store = S3ObjectStore(_settings(), client=BlockingClient())
    probe = Thread(target=lambda: results.append(store.check_health()))
    probe.start()
    assert started.wait(timeout=1)

    assert store.check_health() is False
    release.set()
    probe.join(timeout=1)
    assert not probe.is_alive()
    assert results == [True]


def test_manifest_listing_is_bounded_and_filters_non_intent_objects():
    class Client:
        def list_objects_v2(self, **kwargs):
            assert kwargs == {
                "Bucket": "rag",
                "Prefix": "tenants/",
                "MaxKeys": 4,
                "ContinuationToken": "previous-page",
            }
            return {
                "IsTruncated": True,
                "NextContinuationToken": "next-page",
                "Contents": [
                    {
                        "Key": (
                            "tenants/tenant-a/uploads/task-a/"
                            ".upload-manifest.json"
                        )
                    },
                    {"Key": "tenants/tenant-a/uploads/task-a/payload.txt"},
                    {"Key": "untrusted/.upload-manifest.json"},
                    {
                        "Key": (
                            "tenants/tenant-a/uploads/task-b/"
                            ".upload-manifest.json"
                        )
                    },
                ],
            }

    store = S3ObjectStore(_settings(), client=Client())

    assert store.list_upload_manifest_keys(4, "previous-page") == (
        [
            "tenants/tenant-a/uploads/task-a/.upload-manifest.json",
            "tenants/tenant-a/uploads/task-b/.upload-manifest.json",
        ],
        "next-page",
    )
