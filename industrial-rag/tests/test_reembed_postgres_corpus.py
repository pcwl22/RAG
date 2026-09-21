"""Safety checks for the explicit PostgreSQL corpus re-embedding command."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "reembed_postgres_corpus.py"
SPEC = importlib.util.spec_from_file_location("reembed_postgres_corpus", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reembed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reembed)


def test_snapshot_digest_binds_identity_order_and_content() -> None:
    rows = [("tenant", "one", "content"), ("tenant", "two", "other")]

    assert reembed._snapshot_digest(rows) == reembed._snapshot_digest(list(rows))
    assert reembed._snapshot_digest(rows) != reembed._snapshot_digest(list(reversed(rows)))
    assert reembed._snapshot_digest(rows) != reembed._snapshot_digest(
        [("tenant", "one", "changed"), ("tenant", "two", "other")]
    )


@pytest.mark.parametrize(
    "value",
    [
        "rag_embedding_backup_20260910153000;DROP TABLE documents",
        "other_20260910153000",
        "rag_embedding_backup_latest",
    ],
)
def test_backup_table_name_rejects_identifier_injection(value: str) -> None:
    with pytest.raises(ValueError, match="backup table"):
        reembed._validate_backup_table(value)


def test_source_fingerprint_normalizes_valid_sha256() -> None:
    assert reembed._validate_fingerprint("A" * 64, "source fingerprint") == "a" * 64


def test_source_fingerprint_rejects_non_sha256() -> None:
    with pytest.raises(ValueError, match="64-character lowercase SHA-256"):
        reembed._validate_fingerprint("not-a-digest", "source fingerprint")


def test_backup_table_accepts_timestamped_repository_name() -> None:
    value = "rag_embedding_backup_20260910153000"
    assert reembed._validate_backup_table(value) == value


def test_no_change_still_validates_database_identity_and_count(monkeypatch) -> None:
    fingerprint = "a" * 64
    calls = {"connected": 0, "identity": 0, "rollback": 0, "closed": 0}

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Connection:
        def cursor(self):
            return Cursor()

        def rollback(self):
            calls["rollback"] += 1

        def close(self):
            calls["closed"] += 1

    def connect(**_options):
        calls["connected"] += 1
        return Connection()

    psycopg2 = ModuleType("psycopg2")
    psycopg2.connect = connect
    psycopg2.sql = SimpleNamespace()
    extras = ModuleType("psycopg2.extras")
    extras.execute_values = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", extras)
    monkeypatch.setattr(
        reembed,
        "_runtime_schema_metadata",
        lambda: {"embedding_fingerprint": fingerprint, "embedding_dimension": "1024"},
    )
    monkeypatch.setattr(reembed, "_connection_options", lambda: {})
    monkeypatch.setattr(
        reembed,
        "_require_migration_identity",
        lambda _cursor: calls.__setitem__("identity", calls["identity"] + 1),
    )
    monkeypatch.setattr(
        reembed,
        "_current_metadata",
        lambda _cursor: {"embedding_fingerprint": fingerprint},
    )
    monkeypatch.setattr(
        reembed,
        "_read_snapshot",
        lambda _cursor: [("tenant", "document", "content")],
    )

    result = reembed.rebuild(
        source_fingerprint=fingerprint,
        expected_count=1,
        batch_size=8,
        backup_table="rag_embedding_backup_20260910153000",
        apply=False,
    )

    assert result["status"] == "no_change"
    assert result["document_count"] == 1
    assert result["source_snapshot_sha256"]
    assert calls == {"connected": 1, "identity": 1, "rollback": 1, "closed": 1}
