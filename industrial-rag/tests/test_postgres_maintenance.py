import hashlib
from pathlib import Path

import pytest

from app.vectorstore import postgres_store
from scripts.backup_postgres import backup_database
from scripts.postgres_cli import PostgresTarget
from scripts.restore_postgres import restore_database


def test_postgres_target_prefers_standard_pg_environment():
    target = PostgresTarget.from_env(
        {
            "PGHOST": "database.internal",
            "PGPORT": "6432",
            "PGDATABASE": "production_rag",
            "PGUSER": "backup_user",
            "PGPASSWORD": "secret",
            "POSTGRES_HOST": "ignored",
        }
    )

    assert target.host == "database.internal"
    assert target.port == "6432"
    assert target.database == "production_rag"
    assert target.user == "backup_user"


def test_backup_is_atomic_and_password_is_not_in_command(tmp_path):
    calls = []
    target = PostgresTarget("localhost", "5432", "rag_db", "backup", "secret")
    output = tmp_path / "rag.dump"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        partial = Path(command[command.index("--file") + 1])
        partial.write_bytes(b"postgres archive")

    result = backup_database(target, output, runner=fake_run)

    assert result == output.resolve()
    assert output.read_bytes() == b"postgres archive"
    assert "secret" not in calls[0][0]
    assert calls[0][1]["env"]["PGPASSWORD"] == "secret"
    assert calls[0][1]["check"] is True


def test_restore_requires_exact_database_confirmation(tmp_path):
    archive = tmp_path / "rag.dump"
    archive.write_bytes(b"postgres archive")
    target = PostgresTarget("localhost", "5432", "rag_db", "postgres")

    with pytest.raises(ValueError, match="confirmation does not match"):
        restore_database(
            target,
            archive,
            confirmed_database="another_database",
            runner=lambda *_args, **_kwargs: pytest.fail("restore must not run"),
        )


def test_restore_validates_archive_before_clean_restore(tmp_path):
    archive = tmp_path / "rag.dump"
    archive.write_bytes(b"postgres archive")
    target = PostgresTarget("localhost", "5432", "rag_db", "postgres", "secret")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    restore_database(
        target,
        archive,
        confirmed_database="rag_db",
        clean=True,
        runner=fake_run,
    )

    assert calls[0][0] == ["pg_restore", "--list", str(archive.resolve())]
    assert "--clean" in calls[1][0]
    assert "--if-exists" in calls[1][0]
    assert "--single-transaction" in calls[1][0]
    assert all(call[1]["check"] is True for call in calls)


def test_sql_migration_checksum_is_derived_from_file_content():
    migration_id, checksum, sql_text = postgres_store.CONTENT_SCHEMA_MIGRATIONS[0]

    assert migration_id == "20260805_01_schema_v3_contract"
    assert checksum == "sha256:" + hashlib.sha256(sql_text.encode("utf-8")).hexdigest()
    assert "documents_tenant_isolation" in sql_text
