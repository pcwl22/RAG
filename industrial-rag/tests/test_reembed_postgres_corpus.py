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


def test_document_embedding_dimension_reads_pgvector_typmod() -> None:
    class Cursor:
        def execute(self, _query):
            return None

        def fetchone(self):
            return ("vector(768)", True)

    assert reembed._document_embedding_dimension(Cursor()) == 768
    assert reembed._document_embedding_contract(Cursor()) == (768, True)


def test_no_change_still_validates_database_identity_and_count(monkeypatch) -> None:
    fingerprint = "a" * 64
    chunking_fingerprint = "c" * 64
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
        lambda: {
            "embedding_fingerprint": fingerprint,
            "embedding_dimension": "1024",
            "chunking_fingerprint": chunking_fingerprint,
        },
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
        lambda _cursor: {
            "embedding_fingerprint": fingerprint,
            "embedding_dimension": "1024",
            "chunking_fingerprint": chunking_fingerprint,
        },
    )
    monkeypatch.setattr(
        reembed, "_document_embedding_contract", lambda _cursor: (1024, False)
    )
    monkeypatch.setattr(
        reembed,
        "_read_snapshot",
        lambda _cursor: [("tenant", "document", "content")],
    )

    result = reembed.rebuild(
        source_dimension=1024,
        source_fingerprint=fingerprint,
        source_chunking_fingerprint=chunking_fingerprint,
        expected_count=1,
        batch_size=8,
        backup_table="rag_embedding_backup_20260910153000",
        apply=False,
    )

    assert result["status"] == "no_change"
    assert result["document_count"] == 1
    assert result["source_snapshot_sha256"]
    assert calls == {"connected": 1, "identity": 1, "rollback": 1, "closed": 1}


def test_inspect_corpus_returns_copyable_review_and_apply_commands(monkeypatch) -> None:
    source_fingerprint = "a" * 64
    target_fingerprint = "b" * 64
    chunking_fingerprint = "c" * 64
    calls = {"identity": 0, "rollback": 0, "closed": 0}

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

    psycopg2 = ModuleType("psycopg2")
    psycopg2.connect = lambda **_options: Connection()
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2)
    monkeypatch.setattr(reembed, "_connection_options", lambda: {})
    monkeypatch.setattr(
        reembed,
        "_require_migration_identity",
        lambda _cursor: calls.__setitem__("identity", calls["identity"] + 1),
    )
    monkeypatch.setattr(
        reembed,
        "_current_metadata",
        lambda _cursor: {
            "embedding_fingerprint": source_fingerprint,
            "embedding_dimension": "1024",
            "chunking_fingerprint": chunking_fingerprint,
        },
    )
    monkeypatch.setattr(
        reembed, "_document_embedding_contract", lambda _cursor: (1024, False)
    )
    monkeypatch.setattr(reembed, "_document_count", lambda _cursor: 42)
    monkeypatch.setattr(
        reembed,
        "_runtime_schema_metadata",
        lambda: {
            "embedding_fingerprint": target_fingerprint,
            "embedding_dimension": "1024",
            "chunking_fingerprint": chunking_fingerprint,
        },
    )
    monkeypatch.setattr(
        reembed,
        "_default_backup_table",
        lambda: "rag_embedding_backup_20260924150000",
    )

    result = reembed.inspect_corpus()

    assert result["status"] == "migration_required"
    assert result["document_count"] == 42
    assert "--from-dimension 1024" in result["plan_command"]
    assert f"--from-fingerprint {source_fingerprint}" in result["plan_command"]
    assert (
        f"--from-chunking-fingerprint {chunking_fingerprint}" in result["plan_command"]
    )
    assert "--confirm-document-count 42" in result["plan_command"]
    assert result["apply_command"] == f'{result["plan_command"]} --apply'
    assert calls == {"identity": 1, "rollback": 1, "closed": 1}


def test_inspect_requires_reingest_for_chunking_only_change(monkeypatch) -> None:
    embedding_fingerprint = "a" * 64
    source_chunking_fingerprint = "b" * 64
    target_chunking_fingerprint = "c" * 64

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Connection:
        def cursor(self):
            return Cursor()

        def rollback(self):
            return None

        def close(self):
            return None

    psycopg2 = ModuleType("psycopg2")
    psycopg2.connect = lambda **_options: Connection()
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2)
    monkeypatch.setattr(reembed, "_connection_options", lambda: {})
    monkeypatch.setattr(reembed, "_require_migration_identity", lambda _cursor: None)
    monkeypatch.setattr(
        reembed,
        "_current_metadata",
        lambda _cursor: {
            "embedding_fingerprint": embedding_fingerprint,
            "embedding_dimension": "1024",
            "chunking_fingerprint": source_chunking_fingerprint,
        },
    )
    monkeypatch.setattr(
        reembed, "_document_embedding_contract", lambda _cursor: (1024, False)
    )
    monkeypatch.setattr(reembed, "_document_count", lambda _cursor: 42)
    monkeypatch.setattr(
        reembed,
        "_runtime_schema_metadata",
        lambda: {
            "embedding_fingerprint": embedding_fingerprint,
            "embedding_dimension": "1024",
            "chunking_fingerprint": target_chunking_fingerprint,
        },
    )

    result = reembed.inspect_corpus()

    assert result["status"] == "reingest_required"
    assert "chunking configuration changed" in result["reason"]
    assert "plan_command" not in result
    assert "apply_command" not in result


def test_rebuild_refuses_to_bless_changed_chunking_metadata(monkeypatch) -> None:
    source_fingerprint = "a" * 64
    source_chunking_fingerprint = "b" * 64
    target_chunking_fingerprint = "c" * 64

    psycopg2 = ModuleType("psycopg2")
    psycopg2.connect = lambda **_options: pytest.fail("database connection must not be opened")
    psycopg2.sql = SimpleNamespace()
    extras = ModuleType("psycopg2.extras")
    extras.execute_values = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", extras)
    monkeypatch.setattr(
        reembed,
        "_runtime_schema_metadata",
        lambda: {
            "embedding_fingerprint": source_fingerprint,
            "embedding_dimension": "1024",
            "chunking_fingerprint": target_chunking_fingerprint,
        },
    )

    with pytest.raises(RuntimeError, match="Re-ingest the source corpus"):
        reembed.rebuild(
            source_dimension=1024,
            source_fingerprint=source_fingerprint,
            source_chunking_fingerprint=source_chunking_fingerprint,
            expected_count=1,
            batch_size=8,
            backup_table="rag_embedding_backup_20260910153000",
            apply=False,
        )
