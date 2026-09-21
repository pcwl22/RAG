"""Real pgvector integration coverage, enabled explicitly in CI."""
import asyncio
import json
import os

import psycopg2
import pytest

from app.auth import Principal, reset_current_principal, set_current_principal
from app.utils.config import get_settings
from app.vectorstore import postgres_store

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="requires a disposable pgvector PostgreSQL service",
)


def _integration_config() -> dict:
    configured = get_settings().get("postgres", {})
    return {
        "host": os.getenv("PGHOST") or configured.get("host", "127.0.0.1"),
        "port": int(os.getenv("PGPORT") or configured.get("port", 5432)),
        "database": os.getenv("PGDATABASE") or configured.get("database", "rag_test"),
        "user": os.getenv("PGUSER") or configured.get("user", "postgres"),
        "password": os.getenv("PGPASSWORD") or configured.get("password", "test-only"),
        "min_pool_size": 1,
        "max_pool_size": 3,
        "connect_timeout": 5,
        "command_timeout": 30,
        "auto_migrate": True,
    }


def test_real_postgres_fresh_schema_is_created_at_the_final_contract(monkeypatch):
    cfg = _integration_config()
    setup = psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
    )
    setup.autocommit = True
    cur = setup.cursor()
    try:
        cur.execute(
            "DROP TABLE IF EXISTS documents, rag_schema_metadata, "
            "rag_schema_migrations, rag_tenants CASCADE"
        )
    finally:
        cur.close()
        setup.close()

    monkeypatch.setattr(postgres_store, "_postgres_config", _integration_config)
    monkeypatch.setattr(postgres_store, "_embedding_dimension", lambda: 1024)

    async def run():
        await postgres_store.close_postgres_store()
        await postgres_store.init_postgres_store()
        conn = postgres_store._connection()
        cur = conn.cursor()
        try:
            assert postgres_store._primary_key_state(cur) == (
                "documents_pkey",
                ("tenant_id", "id"),
            )
            cur.execute("SELECT COUNT(*) FROM documents WHERE tenant_id IS NULL")
            assert cur.fetchone()[0] == 0
            cur.execute(
                "SELECT migration_id, checksum FROM rag_schema_migrations "
                "WHERE migration_id = ANY(%s)",
                ([migration[0] for migration in postgres_store.SCHEMA_MIGRATIONS],),
            )
            assert dict(cur.fetchall()) == dict(postgres_store.SCHEMA_MIGRATIONS)

            await postgres_store.replace_document(
                source_key="receipt-source-1",
                filename="receipt.txt",
                ids=["receipt-row-1"],
                embeddings=[[0.0] * 1024],
                documents=["receipt verification content"],
                metadatas=[
                    {
                        "tenant_id": "00000000-0000-0000-0000-000000000001",
                        "document_id": "receipt-doc-1",
                        "source_key": "receipt-source-1",
                        "upload_task_id": "receipt-task-1",
                        "upload_object_key": "tenants/t/uploads/receipt-task-1/payload.txt",
                        "total_chunks": 1,
                    }
                ],
                partition="text",
            )
            assert postgres_store.document_commit_matches(
                tenant_id="00000000-0000-0000-0000-000000000001",
                document_id="receipt-doc-1",
                source_key="receipt-source-1",
                upload_task_id="receipt-task-1",
                upload_object_key="tenants/t/uploads/receipt-task-1/payload.txt",
                total_chunks=1,
            )
            assert not postgres_store.document_commit_matches(
                tenant_id="00000000-0000-0000-0000-000000000001",
                document_id="receipt-doc-1",
                source_key="receipt-source-1",
                upload_task_id="different-task",
                upload_object_key="tenants/t/uploads/receipt-task-1/payload.txt",
                total_chunks=1,
            )
        finally:
            cur.close()
            postgres_store._return_connection(conn)
            await postgres_store.close_postgres_store()

    asyncio.run(run())


def test_real_postgres_legacy_upgrade_is_staged_and_lock_failure_is_rerunnable(
    monkeypatch,
):
    cfg = _integration_config()
    setup = psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
    )
    setup.autocommit = True
    cur = setup.cursor()
    try:
        cur.execute(
            "DROP TABLE IF EXISTS documents, rag_schema_metadata, "
            "rag_schema_migrations, rag_tenants CASCADE"
        )
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(1024),
                metadata JSONB DEFAULT '{}',
                partition TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            INSERT INTO documents (id, content)
            SELECT 'legacy-' || value::text, 'legacy content ' || value::text
            FROM generate_series(1, 2505) AS value
            """
        )
    finally:
        cur.close()
        setup.close()

    monkeypatch.setattr(postgres_store, "_postgres_config", _integration_config)
    monkeypatch.setattr(postgres_store, "_embedding_dimension", lambda: 1024)
    monkeypatch.setenv("POSTGRES_MIGRATION_BATCH_SIZE", "1000")
    monkeypatch.setenv("POSTGRES_PK_CUTOVER_LOCK_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("POSTGRES_ALLOW_LEGACY_PK_CUTOVER", "false")

    async def run():
        await postgres_store.close_postgres_store()
        with pytest.raises(RuntimeError, match="not authorized"):
            await postgres_store.init_postgres_store()

        inspect_conn = psycopg2.connect(
            host=cfg["host"],
            port=cfg["port"],
            database=cfg["database"],
            user=cfg["user"],
            password=cfg["password"],
        )
        inspect_cur = inspect_conn.cursor()
        try:
            inspect_cur.execute("SELECT COUNT(*) FROM documents WHERE tenant_id IS NULL")
            assert inspect_cur.fetchone()[0] == 0
            inspect_cur.execute(
                "SELECT attnotnull FROM pg_attribute "
                "WHERE attrelid = 'documents'::regclass AND attname = 'tenant_id'"
            )
            assert inspect_cur.fetchone()[0] is True
            inspect_cur.execute(
                """
                SELECT array_agg(attribute.attname ORDER BY key_column.ordinality)
                FROM pg_constraint AS constraint_row
                CROSS JOIN LATERAL unnest(constraint_row.conkey)
                    WITH ORDINALITY AS key_column(attnum, ordinality)
                JOIN pg_attribute AS attribute
                  ON attribute.attrelid = constraint_row.conrelid
                 AND attribute.attnum = key_column.attnum
                WHERE constraint_row.conrelid = 'documents'::regclass
                  AND constraint_row.contype = 'p'
                """
            )
            assert inspect_cur.fetchone()[0] == ["id"]
            inspect_cur.execute(
                "SELECT migration_id FROM rag_schema_migrations "
                "WHERE migration_id = ANY(%s)",
                ([migration[0] for migration in postgres_store.STAGED_TENANT_MIGRATIONS],),
            )
            applied = {row[0] for row in inspect_cur.fetchall()}
            assert applied == {
                migration[0]
                for migration in postgres_store.STAGED_TENANT_MIGRATIONS[:4]
            }
        finally:
            inspect_cur.close()
            inspect_conn.rollback()
            inspect_conn.close()

        blocker = psycopg2.connect(
            host=cfg["host"],
            port=cfg["port"],
            database=cfg["database"],
            user=cfg["user"],
            password=cfg["password"],
        )
        blocker_cur = blocker.cursor()
        blocker_cur.execute("SELECT COUNT(*) FROM documents")
        monkeypatch.setenv("POSTGRES_ALLOW_LEGACY_PK_CUTOVER", "true")
        try:
            with pytest.raises(psycopg2.errors.LockNotAvailable):
                await postgres_store.init_postgres_store()
        finally:
            blocker_cur.close()
            blocker.rollback()
            blocker.close()

        await postgres_store.init_postgres_store()
        assert await postgres_store.check_postgres_health() is True
        final_conn = postgres_store._connection()
        final_cur = final_conn.cursor()
        try:
            assert postgres_store._primary_key_state(final_cur)[1] == (
                "tenant_id",
                "id",
            )
            final_cur.execute(
                "SELECT migration_id, checksum FROM rag_schema_migrations "
                "WHERE migration_id = ANY(%s)",
                ([migration[0] for migration in postgres_store.SCHEMA_MIGRATIONS],),
            )
            assert dict(final_cur.fetchall()) == dict(postgres_store.SCHEMA_MIGRATIONS)
        finally:
            final_cur.close()
            postgres_store._return_connection(final_conn)
            await postgres_store.close_postgres_store()

    asyncio.run(run())


def test_real_postgres_runtime_role_has_no_owner_privileges(monkeypatch):
    runtime_user = os.getenv("POSTGRES_RUNTIME_USER")
    runtime_password = os.getenv("POSTGRES_RUNTIME_PASSWORD")
    if not runtime_user or not runtime_password:
        pytest.skip("runtime-role provisioning is not configured")

    monkeypatch.setattr(postgres_store, "_postgres_config", _integration_config)
    monkeypatch.setattr(postgres_store, "_embedding_dimension", lambda: 1024)

    async def migrate():
        await postgres_store.close_postgres_store()
        await postgres_store.init_postgres_store()
        await postgres_store.close_postgres_store()

    asyncio.run(migrate())

    cfg = _integration_config()
    conn = psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        user=runtime_user,
        password=runtime_password,
    )
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT rolsuper, rolcreatedb, rolcreaterole, rolbypassrls "
            "FROM pg_roles WHERE rolname = current_user"
        )
        assert cur.fetchone() == (False, False, False, False)

        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("SELECT COUNT(*) FROM documents")
        conn.rollback()

        cur.execute("SET LOCAL ROLE rag_app")
        cur.execute(
            "SELECT set_config('app.tenant_id', %s, true)",
            ("00000000-0000-0000-0000-000000000001",),
        )
        cur.execute("SELECT COUNT(*) FROM documents")
        assert int(cur.fetchone()[0]) >= 0
    finally:
        cur.close()
        conn.rollback()
        conn.close()


def test_real_postgres_schema_health_and_citation_pairing(monkeypatch):
    monkeypatch.setattr(postgres_store, "_postgres_config", _integration_config)
    monkeypatch.setattr(postgres_store, "_embedding_dimension", lambda: 1024)

    async def run():
        await postgres_store.close_postgres_store()
        await postgres_store.init_postgres_store()
        assert await postgres_store.check_postgres_health() is True

        conn = postgres_store._connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid = 'documents'::regclass"
            )
            assert cur.fetchone() == (True, True)
            cur.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = 'rag_app'"
            )
            assert cur.fetchone() == (False, False)
            cur.execute(
                """
                SELECT array_agg(attribute.attname ORDER BY key_column.ordinality)
                FROM pg_constraint AS constraint_row
                CROSS JOIN LATERAL unnest(constraint_row.conkey)
                    WITH ORDINALITY AS key_column(attnum, ordinality)
                JOIN pg_attribute AS attribute
                  ON attribute.attrelid = constraint_row.conrelid
                 AND attribute.attnum = key_column.attnum
                WHERE constraint_row.conrelid = 'public.documents'::regclass
                  AND constraint_row.contype = 'p'
                """
            )
            assert cur.fetchone()[0] == ["tenant_id", "id"]
            cur.execute(
                "SELECT migration_id, checksum FROM rag_schema_migrations ORDER BY migration_id"
            )
            assert cur.fetchall() == sorted(postgres_store.SCHEMA_MIGRATIONS)
        finally:
            cur.close()
            postgres_store._return_connection(conn)

        rows = [
            ("pair-civil-1", "民法典第一条", "中华人民共和国民法典", "第一条"),
            ("pair-civil-2", "民法典第二条", "中华人民共和国民法典", "第二条"),
            ("pair-criminal-1", "刑法第一条", "中华人民共和国刑法", "第一条"),
            ("pair-criminal-2", "刑法第二条", "中华人民共和国刑法", "第二条"),
        ]
        conn = postgres_store._connection()
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM documents WHERE id = ANY(%s)", ([row[0] for row in rows],))
            cur.executemany(
                """
                INSERT INTO documents (id, content, metadata, partition)
                VALUES (%s, %s, %s::jsonb, 'integration')
                """,
                [
                    (
                        row_id,
                        content,
                        json.dumps(
                            {
                                "law_name": law_name,
                                "article_number": article_number,
                                "document_id": row_id,
                            },
                            ensure_ascii=False,
                        ),
                    )
                    for row_id, content, law_name, article_number in rows
                ],
            )
            conn.commit()
        finally:
            cur.close()
            postgres_store._return_connection(conn)

        try:
            docs = await postgres_store.get_documents_by_citations(
                [
                    ("中华人民共和国民法典", "第一条"),
                    ("中华人民共和国刑法", "第二条"),
                ],
                partition="integration",
            )
            assert [doc["id"] for doc in docs] == ["pair-civil-1", "pair-criminal-2"]

            vector = [0.0] * 1024
            source_key = "integration-replace-source"
            await postgres_store.replace_document(
                source_key,
                "replace.txt",
                ["replace-v1-1", "replace-v1-2"],
                [vector, vector],
                ["旧切片一", "旧切片二"],
                [
                    {
                        "source_key": source_key,
                        "filename": "replace.txt",
                        "document_id": "replace-v1",
                        "semantic_chunk_id": "replace-semantic-1",
                    },
                    {
                        "source_key": source_key,
                        "filename": "replace.txt",
                        "document_id": "replace-v1",
                        "semantic_chunk_id": "replace-semantic-2",
                    },
                ],
                "integration",
            )
            await postgres_store.replace_document(
                source_key,
                "replace.txt",
                ["replace-v2-1"],
                [vector],
                ["新切片"],
                [
                    {
                        "source_key": source_key,
                        "filename": "replace.txt",
                        "document_id": "replace-v2",
                        "semantic_chunk_id": "replace-semantic-1",
                    }
                ],
                "integration",
            )
            replacement = await postgres_store.get_documents_by_ids(
                ["replace-semantic-1"], partition="integration"
            )
            assert [doc["id"] for doc in replacement] == ["replace-v2-1"]

            conn = postgres_store._connection()
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT id FROM documents WHERE metadata->>'source_key' = %s ORDER BY id",
                    (source_key,),
                )
                assert cur.fetchall() == [("replace-v2-1",)]
            finally:
                cur.close()
                postgres_store._return_connection(conn)
        finally:
            conn = postgres_store._connection()
            cur = conn.cursor()
            try:
                cur.execute(
                    "DELETE FROM documents WHERE id = ANY(%s) OR metadata->>'source_key' = %s",
                    ([row[0] for row in rows], "integration-replace-source"),
                )
                conn.commit()
            finally:
                cur.close()
                postgres_store._return_connection(conn)
            await postgres_store.close_postgres_store()

    asyncio.run(run())


def test_real_postgres_prevents_cross_tenant_read_and_delete(monkeypatch):
    monkeypatch.setattr(postgres_store, "_postgres_config", _integration_config)
    monkeypatch.setattr(postgres_store, "_embedding_dimension", lambda: 1024)
    tenant_a = "00000000-0000-0000-0000-00000000000a"
    tenant_b = "00000000-0000-0000-0000-00000000000b"
    row_a = "tenant-isolation-a"
    row_b = "tenant-isolation-b"

    async def as_tenant(tenant_id, operation):
        token = set_current_principal(
            Principal(f"user-{tenant_id[-1]}", tenant_id, frozenset({"admin"}), "test")
        )
        try:
            return await operation()
        finally:
            reset_current_principal(token)

    async def run():
        await postgres_store.close_postgres_store()
        await postgres_store.init_postgres_store()
        conn = postgres_store._connection()
        cur = conn.cursor()
        try:
            cur.executemany(
                """
                INSERT INTO rag_tenants (id, name) VALUES (%s::uuid, %s)
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
                """,
                [(tenant_a, "Tenant A"), (tenant_b, "Tenant B")],
            )
            cur.execute("DELETE FROM documents WHERE id = ANY(%s)", ([row_a, row_b],))
            conn.commit()
        finally:
            cur.close()
            postgres_store._return_connection(conn)

        vector = [0.0] * 1024
        try:
            await as_tenant(
                tenant_a,
                lambda: postgres_store.add_documents(
                    [row_a],
                    [vector],
                    ["Tenant A private content"],
                    [{"document_id": row_a}],
                    "tenant-test",
                ),
            )
            await as_tenant(
                tenant_b,
                lambda: postgres_store.add_documents(
                    [row_b],
                    [vector],
                    ["Tenant B private content"],
                    [{"document_id": row_b}],
                    "tenant-test",
                ),
            )

            docs_a = await as_tenant(
                tenant_a,
                lambda: postgres_store.get_documents_by_ids([row_a, row_b], "tenant-test"),
            )
            assert [doc["id"] for doc in docs_a] == [row_a]
            assert await as_tenant(
                tenant_a, lambda: postgres_store.delete_document(row_b)
            ) is False

            docs_b = await as_tenant(
                tenant_b,
                lambda: postgres_store.get_documents_by_ids([row_a, row_b], "tenant-test"),
            )
            assert [doc["id"] for doc in docs_b] == [row_b]
        finally:
            conn = postgres_store._connection()
            cur = conn.cursor()
            try:
                cur.execute("DELETE FROM documents WHERE id = ANY(%s)", ([row_a, row_b],))
                conn.commit()
            finally:
                cur.close()
                postgres_store._return_connection(conn)
            await postgres_store.close_postgres_store()

    asyncio.run(run())
