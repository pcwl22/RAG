"""Real PostgreSQL coverage for atomic cross-dimension corpus re-embedding."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "reembed_postgres_corpus.py"
SPEC = importlib.util.spec_from_file_location("reembed_postgres_corpus_integration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reembed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reembed)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="requires a disposable pgvector PostgreSQL service",
)


def _connection_options() -> dict[str, object]:
    return {
        "host": os.getenv("PGHOST", "127.0.0.1"),
        "port": int(os.getenv("PGPORT", "5432")),
        "dbname": os.getenv("PGDATABASE", "rag_test"),
        "user": os.getenv("PGUSER", "postgres"),
        "password": os.getenv("PGPASSWORD", "test-only"),
        "connect_timeout": 5,
    }


def test_cross_dimension_rebuild_is_atomic_and_preserves_source_backup(monkeypatch) -> None:
    source_fingerprint = "a" * 64
    target_fingerprint = "b" * 64
    chunking_fingerprint = "c" * 64
    backup_table = "rag_embedding_backup_20990101000000"
    options = _connection_options()
    setup = psycopg2.connect(**options)
    setup.autocommit = True
    try:
        with setup.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {}, documents, rag_schema_metadata CASCADE").format(
                    sql.Identifier(backup_table)
                )
            )
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                """
                CREATE TABLE documents (
                    tenant_id uuid NOT NULL,
                    id text NOT NULL,
                    content text NOT NULL,
                    embedding vector(3) NOT NULL,
                    PRIMARY KEY (tenant_id, id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE rag_schema_metadata (
                    key text PRIMARY KEY,
                    value text NOT NULL,
                    updated_at timestamp DEFAULT NOW()
                )
                """
            )
            cursor.executemany(
                "INSERT INTO rag_schema_metadata (key, value) VALUES (%s, %s)",
                [
                    ("schema_version", "3"),
                    ("embedding_dimension", "3"),
                    ("embedding_fingerprint", source_fingerprint),
                    ("chunking_fingerprint", chunking_fingerprint),
                ],
            )
            cursor.execute(
                """
                INSERT INTO documents (tenant_id, id, content, embedding) VALUES
                    ('00000000-0000-0000-0000-000000000001', 'one', 'first', '[1,2,3]'),
                    ('00000000-0000-0000-0000-000000000001', 'two', 'second', '[4,5,6]')
                """
            )
            cursor.execute(
                """
                CREATE INDEX idx_documents_embedding_ivfflat
                ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1)
                """
            )

        monkeypatch.setattr(reembed, "_connection_options", lambda: options)
        monkeypatch.setattr(
            reembed,
            "_runtime_schema_metadata",
            lambda: {
                "schema_version": "3",
                "embedding_dimension": "2",
                "embedding_fingerprint": target_fingerprint,
                "chunking_fingerprint": chunking_fingerprint,
            },
        )
        monkeypatch.setattr(
            reembed,
            "encode_texts",
            lambda texts, *, batch_size: [
                [float(index + 1), float(index + 2)] for index, _text in enumerate(texts)
            ],
        )

        result = reembed.rebuild(
            source_dimension=3,
            source_fingerprint=source_fingerprint,
            source_chunking_fingerprint=chunking_fingerprint,
            expected_count=2,
            batch_size=8,
            backup_table=backup_table,
            apply=True,
        )

        assert result["status"] == "completed"
        assert result["source_embedding_dimension"] == 3
        assert result["target_embedding_dimension"] == 2
        with setup.cursor() as cursor:
            cursor.execute(
                """
                SELECT format_type(atttypid, atttypmod), attnotnull
                FROM pg_attribute
                WHERE attrelid = 'documents'::regclass AND attname = 'embedding'
                """
            )
            assert cursor.fetchone() == ("vector(2)", True)
            cursor.execute(
                sql.SQL(
                    "SELECT format_type(atttypid, atttypmod), attnotnull FROM pg_attribute "
                    "WHERE attrelid = {}::regclass AND attname = 'embedding'"
                ).format(sql.Literal(backup_table))
            )
            assert cursor.fetchone() == ("vector(3)", True)
            cursor.execute(
                sql.SQL("SELECT COUNT(*), MIN(vector_dims(embedding)) FROM {}").format(
                    sql.Identifier(backup_table)
                )
            )
            assert cursor.fetchone() == (2, 3)
            cursor.execute("SELECT COUNT(*), MIN(vector_dims(embedding)) FROM documents")
            assert cursor.fetchone() == (2, 2)
            cursor.execute("SELECT key, value FROM rag_schema_metadata")
            metadata = dict(cursor.fetchall())
            assert metadata["embedding_dimension"] == "2"
            assert metadata["embedding_fingerprint"] == target_fingerprint
            assert metadata["chunking_fingerprint"] == chunking_fingerprint
            cursor.execute(
                """
                SELECT index_row.indisvalid, index_row.indisready
                FROM pg_index AS index_row
                JOIN pg_class AS index_class ON index_class.oid = index_row.indexrelid
                WHERE index_class.relname = 'idx_documents_embedding_ivfflat'
                """
            )
            assert cursor.fetchone() == (True, True)
    finally:
        with setup.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {}, documents, rag_schema_metadata CASCADE").format(
                    sql.Identifier(backup_table)
                )
            )
        setup.close()


def test_cross_dimension_failure_rolls_back_column_index_metadata_and_backup(monkeypatch) -> None:
    source_fingerprint = "d" * 64
    target_fingerprint = "e" * 64
    chunking_fingerprint = "f" * 64
    backup_table = "rag_embedding_backup_20990101000001"
    options = _connection_options()
    setup = psycopg2.connect(**options)
    setup.autocommit = True
    try:
        with setup.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {}, documents, rag_schema_metadata CASCADE").format(
                    sql.Identifier(backup_table)
                )
            )
            cursor.execute("DROP FUNCTION IF EXISTS reject_embedding_update() CASCADE")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                """
                CREATE TABLE documents (
                    tenant_id uuid NOT NULL,
                    id text NOT NULL,
                    content text NOT NULL,
                    embedding vector(3) NOT NULL,
                    PRIMARY KEY (tenant_id, id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE rag_schema_metadata (
                    key text PRIMARY KEY,
                    value text NOT NULL,
                    updated_at timestamp DEFAULT NOW()
                )
                """
            )
            cursor.executemany(
                "INSERT INTO rag_schema_metadata (key, value) VALUES (%s, %s)",
                [
                    ("schema_version", "3"),
                    ("embedding_dimension", "3"),
                    ("embedding_fingerprint", source_fingerprint),
                    ("chunking_fingerprint", chunking_fingerprint),
                ],
            )
            cursor.execute(
                """
                INSERT INTO documents (tenant_id, id, content, embedding)
                VALUES ('00000000-0000-0000-0000-000000000001', 'one', 'first', '[1,2,3]')
                """
            )
            cursor.execute(
                """
                CREATE INDEX idx_documents_embedding_ivfflat
                ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1)
                """
            )
            cursor.execute(
                """
                CREATE FUNCTION reject_embedding_update() RETURNS trigger
                LANGUAGE plpgsql AS $$
                BEGIN
                    RAISE EXCEPTION 'injected cutover failure';
                END
                $$
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER reject_embedding_update
                BEFORE UPDATE OF embedding ON documents
                FOR EACH ROW EXECUTE FUNCTION reject_embedding_update()
                """
            )

        monkeypatch.setattr(reembed, "_connection_options", lambda: options)
        monkeypatch.setattr(
            reembed,
            "_runtime_schema_metadata",
            lambda: {
                "schema_version": "3",
                "embedding_dimension": "2",
                "embedding_fingerprint": target_fingerprint,
                "chunking_fingerprint": chunking_fingerprint,
            },
        )
        monkeypatch.setattr(
            reembed,
            "encode_texts",
            lambda texts, *, batch_size: [[9.0, 10.0] for _text in texts],
        )

        with pytest.raises(psycopg2.Error, match="cannot alter type"):
            reembed.rebuild(
                source_dimension=3,
                source_fingerprint=source_fingerprint,
                source_chunking_fingerprint=chunking_fingerprint,
                expected_count=1,
                batch_size=8,
                backup_table=backup_table,
                apply=True,
            )

        with setup.cursor() as cursor:
            cursor.execute(
                """
                SELECT format_type(atttypid, atttypmod), attnotnull
                FROM pg_attribute
                WHERE attrelid = 'documents'::regclass AND attname = 'embedding'
                """
            )
            assert cursor.fetchone() == ("vector(3)", True)
            cursor.execute("SELECT embedding::text FROM documents")
            assert cursor.fetchone()[0] == "[1,2,3]"
            cursor.execute("SELECT key, value FROM rag_schema_metadata")
            metadata = dict(cursor.fetchall())
            assert metadata["embedding_dimension"] == "3"
            assert metadata["embedding_fingerprint"] == source_fingerprint
            cursor.execute(
                """
                SELECT index_row.indisvalid, index_row.indisready
                FROM pg_index AS index_row
                JOIN pg_class AS index_class ON index_class.oid = index_row.indexrelid
                WHERE index_class.relname = 'idx_documents_embedding_ivfflat'
                """
            )
            assert cursor.fetchone() == (True, True)
            cursor.execute("SELECT to_regclass(%s)", (backup_table,))
            assert cursor.fetchone()[0] is None
    finally:
        with setup.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {}, documents, rag_schema_metadata CASCADE").format(
                    sql.Identifier(backup_table)
                )
            )
            cursor.execute("DROP FUNCTION IF EXISTS reject_embedding_update() CASCADE")
        setup.close()
