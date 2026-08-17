"""PostgreSQL + pgvector storage with hybrid retrieval."""
import asyncio
import contextvars
import functools
import hashlib
import json
import os
from asyncio import Future
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from app.auth import DEFAULT_TENANT_ID, current_tenant_id
from app.utils.config import get_config_section, get_settings
from app.utils.logger import get_logger

# Optional at import time so the module can be imported without the driver;
# every entry point raises RuntimeError if the store was never initialized.
psycopg2: Any
sql: Any
try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
    from psycopg2 import sql
except ImportError:
    psycopg2 = None
    sql = None

logger = get_logger(__name__)

_T = TypeVar("_T")
_pool: Any | None = None
_executor: ThreadPoolExecutor | None = None
SCHEMA_VERSION = "3"
LEGACY_SCHEMA_MIGRATION_REVISIONS = (
    ("20260731_01_tenant_rls", "v1"),
    ("20260731_02_composite_document_key", "v1"),
    ("20260731_03_search_indexes", "v1"),
)
LEGACY_SCHEMA_MIGRATIONS = tuple(
    (
        migration_id,
        "sha256:" + hashlib.sha256(f"{migration_id}:{revision}".encode()).hexdigest(),
    )
    for migration_id, revision in LEGACY_SCHEMA_MIGRATION_REVISIONS
)
MIGRATION_VERSIONS_DIR = Path(__file__).with_name("migrations") / "versions"


def _load_content_migrations() -> tuple[tuple[str, str, str], ...]:
    """Load immutable SQL migrations and derive checksums from their exact bytes."""
    migrations: list[tuple[str, str, str]] = []
    for path in sorted(MIGRATION_VERSIONS_DIR.glob("*.sql")):
        # Normalize checkout-specific line endings so Windows and Linux derive
        # the same immutable migration checksum.
        sql_text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        raw = sql_text.encode("utf-8")
        migrations.append(
            (
                path.stem,
                "sha256:" + hashlib.sha256(raw).hexdigest(),
                sql_text,
            )
        )
    return tuple(migrations)


CONTENT_SCHEMA_MIGRATIONS = _load_content_migrations()
SCHEMA_MIGRATIONS = LEGACY_SCHEMA_MIGRATIONS + tuple(
    (migration_id, checksum)
    for migration_id, checksum, _sql_text in CONTENT_SCHEMA_MIGRATIONS
)
SEARCH_TEXT_EXPRESSION = """(
    content || ' ' ||
    COALESCE(metadata->>'filename', '') || ' ' ||
    COALESCE(metadata->>'law_name', '') || ' ' ||
    COALESCE(metadata->>'legal_citation', '') || ' ' ||
    COALESCE(metadata->>'semantic_chunk_id', '')
)"""


def _postgres_config() -> dict[str, Any]:
    return get_config_section("postgres")


def _embedding_dimension() -> int:
    return int(get_config_section("embedding").get("dimension", 1024))


def _auto_migrate_enabled() -> bool:
    value = os.getenv("POSTGRES_AUTO_MIGRATE")
    if value is None:
        value = _postgres_config().get("auto_migrate", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _runtime_schema_metadata() -> dict[str, str]:
    settings = get_settings()
    embedding = settings.get("embedding", {})
    chunking = settings.get("document_processing", {}).get("chunking", {})
    embedding_identity = {
        "model_name": embedding.get("model_name"),
        "model_revision": embedding.get("model_revision"),
        "dimension": _embedding_dimension(),
        "normalize_embeddings": bool(embedding.get("normalize_embeddings", True)),
        "max_length": embedding.get("max_length"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_dimension": str(_embedding_dimension()),
        "embedding_fingerprint": _fingerprint(embedding_identity),
        "chunking_fingerprint": _fingerprint(chunking),
    }


def _validate_schema_metadata(existing: dict[str, str], document_count: int) -> None:
    expected = _runtime_schema_metadata()
    schema_version = existing.get("schema_version")
    supported_upgrade = (schema_version, SCHEMA_VERSION) in {("1", "3"), ("2", "3")}
    if schema_version is not None and schema_version != SCHEMA_VERSION and not supported_upgrade:
        raise RuntimeError(
            f"PostgreSQL schema version mismatch: expected {SCHEMA_VERSION}, "
            f"found {schema_version}. Run an explicit schema migration."
        )
    for key in ("embedding_dimension", "embedding_fingerprint", "chunking_fingerprint"):
        actual = existing.get(key)
        if actual is not None and actual != expected[key] and document_count > 0:
            raise RuntimeError(
                f"PostgreSQL corpus fingerprint mismatch for {key}. "
                "Rebuild the corpus with an explicit re-embedding migration before startup."
            )


def _migration_required(cur: Any, migration_id: str, checksum: str) -> bool:
    """Return whether a migration must run, rejecting edited applied migrations."""
    cur.execute(
        "SELECT checksum FROM rag_schema_migrations WHERE migration_id = %s",
        (migration_id,),
    )
    row = cur.fetchone()
    if row is None:
        return True
    if str(row[0]) != checksum:
        raise RuntimeError(
            f"PostgreSQL migration checksum mismatch for {migration_id}. "
            "Applied migrations are immutable; add a new migration instead."
        )
    return False


def _record_migration(cur: Any, migration_id: str, checksum: str) -> None:
    cur.execute(
        """
        INSERT INTO rag_schema_migrations (migration_id, checksum)
        VALUES (%s, %s)
        ON CONFLICT (migration_id) DO NOTHING
        """,
        (migration_id, checksum),
    )


async def init_postgres_store() -> None:
    """Initialize connection pool and schema."""
    global _pool, _executor
    if _pool is not None:
        return
    if psycopg2 is None:
        raise ImportError("psycopg2 is required for PostgreSQL retrieval. Install psycopg2-binary.")

    cfg = _postgres_config()
    try:
        _executor = ThreadPoolExecutor(max_workers=int(cfg.get("max_pool_size", 5)))
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=int(cfg.get("min_pool_size", 1)),
            maxconn=int(cfg.get("max_pool_size", 5)),
            host=cfg.get("host", "localhost"),
            port=int(cfg.get("port", 5432)),
            database=cfg.get("database", "rag_db"),
            user=cfg.get("user", "postgres"),
            password=cfg.get("password", ""),
            client_encoding="utf8",
            connect_timeout=int(cfg.get("connect_timeout", 5)),
            keepalives=1,
            keepalives_idle=int(cfg.get("keepalives_idle", 30)),
            keepalives_interval=int(cfg.get("keepalives_interval", 10)),
            keepalives_count=int(cfg.get("keepalives_count", 3)),
            options=f"-c statement_timeout={int(cfg.get('command_timeout', 60)) * 1000}",
        )
        if _auto_migrate_enabled():
            await _execute_sync(_create_schema_sync)
        else:
            await _execute_sync(_require_schema_ready_sync)
    except Exception:
        _dispose_postgres_runtime()
        raise
    logger.info("PostgreSQL store initialized")


def _execute_sync(func: Callable[..., _T], *args: Any, **kwargs: Any) -> "Future[_T]":
    """Run a blocking DB callable on the pool executor, preserving its result type."""
    if _executor is None:
        raise RuntimeError("PostgreSQL store is not initialized")
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    callback = functools.partial(func, *args, **kwargs)
    return loop.run_in_executor(_executor, context.run, callback)


def _connection() -> Any:
    if _pool is None:
        raise RuntimeError("PostgreSQL store is not initialized")
    return _pool.getconn()


def _return_connection(conn: Any) -> None:
    """Return a clean connection to the pool, discarding it if rollback fails."""
    if _pool is None:
        return
    discard = False
    try:
        # psycopg2 leaves failed statements in an aborted transaction.  A
        # rollback is also safe after a commit/read-only transaction.
        conn.rollback()
    except Exception:
        discard = True
        logger.warning("Discarding PostgreSQL connection after rollback failure", exc_info=True)
    _pool.putconn(conn, close=discard)


def _set_tenant(cur: Any, tenant_id: str | None = None) -> str:
    """Bind this transaction to one tenant for PostgreSQL RLS."""
    resolved = tenant_id or current_tenant_id()
    # Connections may use an administrative migration account. Drop its
    # BYPASSRLS/superuser privileges for every application data transaction.
    cur.execute("SET LOCAL ROLE rag_app", ())
    cur.execute("SELECT set_config('app.tenant_id', %s, true)", (resolved,))
    return resolved


def _provision_runtime_role(cur: Any) -> None:
    """Provision the non-owner login used by API and worker containers."""
    runtime_user = str(os.getenv("POSTGRES_RUNTIME_USER") or "").strip()
    runtime_password = os.getenv("POSTGRES_RUNTIME_PASSWORD")
    if not runtime_user and not runtime_password:
        return
    if not runtime_user or not runtime_password:
        raise RuntimeError(
            "POSTGRES_RUNTIME_USER and POSTGRES_RUNTIME_PASSWORD must be configured together"
        )
    if runtime_user in {"postgres", "rag_app"}:
        raise RuntimeError("PostgreSQL runtime user must be a dedicated non-owner role")
    if sql is None:
        raise RuntimeError("psycopg2 SQL helpers are unavailable")

    cur.execute("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = %s)", (runtime_user,))
    if cur.fetchone()[0]:
        cur.execute(
            sql.SQL(
                "ALTER ROLE {} LOGIN PASSWORD %s NOSUPERUSER NOCREATEDB "
                "NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS"
            ).format(sql.Identifier(runtime_user)),
            (runtime_password,),
        )
    else:
        cur.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN PASSWORD %s NOSUPERUSER NOCREATEDB "
                "NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS"
            ).format(sql.Identifier(runtime_user)),
            (runtime_password,),
        )
    cur.execute(sql.SQL("GRANT rag_app TO {}").format(sql.Identifier(runtime_user)))


def _create_schema_sync() -> None:
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('industrial-rag-schema-v3'))")
        dimension = _embedding_dimension()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_schema_migrations (
                migration_id TEXT PRIMARY KEY,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_tenants (
                id UUID PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            INSERT INTO rag_tenants (id, name)
            VALUES (%s::uuid, 'Default tenant')
            ON CONFLICT (id) DO NOTHING
            """,
            (DEFAULT_TENANT_ID,),
        )
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_app') THEN
                    CREATE ROLE rag_app NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE
                        NOINHERIT NOREPLICATION NOBYPASSRLS;
                END IF;
            END
            $$
            """
        )
        cur.execute(
            """
            ALTER ROLE rag_app NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE
                NOINHERIT NOREPLICATION NOBYPASSRLS
            """
        )
        cur.execute("GRANT SELECT ON rag_schema_migrations TO rag_app")
        _provision_runtime_role(cur)
        cur.execute(
            """
            DO $$
            BEGIN
                IF current_user <> 'rag_app' THEN
                    EXECUTE format('GRANT rag_app TO %I', current_user);
                END IF;
            END
            $$
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT NOT NULL,
                tenant_id UUID NOT NULL DEFAULT '{DEFAULT_TENANT_ID}'::uuid
                    REFERENCES rag_tenants(id),
                content TEXT NOT NULL,
                embedding vector({dimension}),
                metadata JSONB DEFAULT '{{}}',
                partition TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (tenant_id, id)
            );
            """
        )
        # Upgrade schema v1 in place. Existing rows belong to the documented
        # default tenant; no corpus rebuild or destructive migration is needed.
        cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS tenant_id UUID")
        cur.execute(
            "UPDATE documents SET tenant_id = %s::uuid WHERE tenant_id IS NULL",
            (DEFAULT_TENANT_ID,),
        )
        cur.execute(
            f"ALTER TABLE documents ALTER COLUMN tenant_id SET DEFAULT '{DEFAULT_TENANT_ID}'::uuid"
        )
        cur.execute("ALTER TABLE documents ALTER COLUMN tenant_id SET NOT NULL")
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'documents_tenant_id_fkey'
                      AND conrelid = 'public.documents'::regclass
                ) THEN
                    ALTER TABLE documents
                    ADD CONSTRAINT documents_tenant_id_fkey
                    FOREIGN KEY (tenant_id) REFERENCES rag_tenants(id);
                END IF;
            END
            $$
            """
        )
        tenant_migration = SCHEMA_MIGRATIONS[0]
        composite_key_migration = SCHEMA_MIGRATIONS[1]
        index_migration = SCHEMA_MIGRATIONS[2]

        if _migration_required(cur, *composite_key_migration):
            # Schema v3 removes the cross-tenant availability coupling created by
            # the legacy global id primary key. The block also handles databases
            # whose primary-key constraint was renamed.
            cur.execute(
                """
            DO $$
            DECLARE
                current_pk TEXT;
            BEGIN
                SELECT constraint_name INTO current_pk
                FROM information_schema.table_constraints
                WHERE table_schema = 'public'
                  AND table_name = 'documents'
                  AND constraint_type = 'PRIMARY KEY';

                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint AS constraint_row
                    WHERE constraint_row.conrelid = 'public.documents'::regclass
                      AND constraint_row.contype = 'p'
                      AND (
                        SELECT array_agg(attribute.attname ORDER BY key_column.ordinality)
                        FROM unnest(constraint_row.conkey) WITH ORDINALITY AS key_column(attnum, ordinality)
                        JOIN pg_attribute AS attribute
                          ON attribute.attrelid = constraint_row.conrelid
                         AND attribute.attnum = key_column.attnum
                      ) = ARRAY['tenant_id', 'id']::name[]
                ) THEN
                    IF current_pk IS NOT NULL THEN
                        EXECUTE format('ALTER TABLE public.documents DROP CONSTRAINT %I', current_pk);
                    END IF;
                    ALTER TABLE public.documents
                    ADD CONSTRAINT documents_pkey PRIMARY KEY (tenant_id, id);
                END IF;
            END
            $$
            """
            )
            _record_migration(cur, *composite_key_migration)

        if _migration_required(cur, *index_migration):
            # Index creation can scan the full corpus. Keep it behind a migration
            # record so routine migration jobs do not repeat catalog-heavy DDL.
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_documents_search_fields_trgm
                ON documents USING gin ({SEARCH_TEXT_EXPRESSION} gin_trgm_ops)
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents (tenant_id)"
            )
            cur.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_documents_tenant_partition
            ON documents (tenant_id, partition)
            """
            )
            # Superseded indexes included the entire metadata JSON (including repeated
            # parent_content) and duplicated the content trigram index.
            cur.execute("DROP INDEX IF EXISTS idx_documents_search_trgm")
            cur.execute("DROP INDEX IF EXISTS idx_documents_content_trgm")
            cur.execute("DROP INDEX IF EXISTS idx_documents_metadata")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_partition ON documents (partition)"
            )
            cur.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_documents_document_id
            ON documents ((metadata->>'document_id'))
            """
            )
            cur.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_documents_source_key
            ON documents ((metadata->>'source_key'))
            """
            )
            cur.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_documents_semantic_chunk_id
            ON documents ((metadata->>'semantic_chunk_id'))
            """
            )
            cur.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_documents_citation
            ON documents ((metadata->>'law_name'), (metadata->>'article_number'))
            """
            )
            cur.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_documents_tenant_document_id
            ON documents (tenant_id, (metadata->>'document_id'))
            """
            )
            cur.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_documents_tenant_source_key
            ON documents (tenant_id, (metadata->>'source_key'))
            """
            )
            cur.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_documents_tenant_semantic_chunk_id
            ON documents (tenant_id, (metadata->>'semantic_chunk_id'))
            """
            )
            cur.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_documents_tenant_citation
            ON documents (tenant_id, (metadata->>'law_name'), (metadata->>'article_number'))
            """
            )
            cur.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat
            ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """
            )
            _record_migration(cur, *index_migration)

        if _migration_required(cur, *tenant_migration):
            cur.execute("ALTER TABLE documents ENABLE ROW LEVEL SECURITY")
            cur.execute("ALTER TABLE documents FORCE ROW LEVEL SECURITY")
            cur.execute("DROP POLICY IF EXISTS documents_tenant_isolation ON documents")
            cur.execute(
                """
            CREATE POLICY documents_tenant_isolation ON documents
            USING (
                tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
            )
            WITH CHECK (
                tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
            )
            """
            )
            _record_migration(cur, *tenant_migration)

        for migration_id, checksum, sql_text in CONTENT_SCHEMA_MIGRATIONS:
            if _migration_required(cur, migration_id, checksum):
                cur.execute(sql_text)
                _record_migration(cur, migration_id, checksum)
        cur.execute("GRANT USAGE ON SCHEMA public TO rag_app")
        cur.execute("GRANT SELECT ON rag_tenants TO rag_app")
        cur.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON documents TO rag_app")
        cur.execute(
            "SELECT set_config('app.tenant_id', %s, true)",
            (DEFAULT_TENANT_ID,),
        )
        cur.execute(
            """
            SELECT format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_attribute AS attribute
            WHERE attribute.attrelid = 'public.documents'::regclass
              AND attribute.attname = 'embedding'
              AND NOT attribute.attisdropped
            """
        )
        row = cur.fetchone()
        actual_vector_type = str(row[0]) if row else "missing"
        expected_vector_type = f"vector({dimension})"
        if actual_vector_type != expected_vector_type:
            raise RuntimeError(
                "PostgreSQL embedding dimension mismatch: "
                f"expected {expected_vector_type}, found {actual_vector_type}. "
                "Run an explicit schema migration and re-embed the corpus."
            )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """
        )
        cur.execute("GRANT SELECT ON rag_schema_metadata TO rag_app")
        cur.execute("SELECT key, value FROM rag_schema_metadata")
        existing_metadata = {str(key): str(value) for key, value in cur.fetchall()}
        cur.execute("SELECT COUNT(*) FROM documents")
        document_count = int(cur.fetchone()[0])
        _validate_schema_metadata(existing_metadata, document_count)
        cur.executemany(
            """
            INSERT INTO rag_schema_metadata (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value, updated_at = NOW()
            """,
            list(_runtime_schema_metadata().items()),
        )
        conn.commit()
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


def _dispose_postgres_runtime() -> None:
    global _pool, _executor
    if _pool is not None:
        _pool.closeall()
        _pool = None
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


async def close_postgres_store() -> None:
    _dispose_postgres_runtime()


def _require_schema_ready_sync() -> None:
    if not _check_postgres_health_sync():
        raise RuntimeError(
            "PostgreSQL schema is not ready for the configured runtime. "
            "Run the database migration job before starting the application."
        )


def _check_postgres_health_sync() -> bool:
    """Verify that the pool can execute a query, not merely that it exists."""
    if _pool is None:
        return False
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor()
        _set_tenant(cur)
        cur.execute(
            """
            SELECT
                to_regclass('public.documents') IS NOT NULL,
                to_regclass('public.rag_schema_metadata') IS NOT NULL,
                to_regclass('public.rag_schema_migrations') IS NOT NULL,
                to_regclass('public.rag_tenants') IS NOT NULL,
                EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')
            """
        )
        documents_ready, metadata_ready, migrations_ready, tenants_ready, vector_ready = cur.fetchone()
        if not (
            documents_ready
            and metadata_ready
            and migrations_ready
            and tenants_ready
            and vector_ready
        ):
            return False
        cur.execute(
            "SELECT migration_id, checksum FROM rag_schema_migrations WHERE migration_id = ANY(%s)",
            ([migration_id for migration_id, _checksum in SCHEMA_MIGRATIONS],),
        )
        applied_migrations = {str(migration_id): str(checksum) for migration_id, checksum in cur.fetchall()}
        if any(applied_migrations.get(migration_id) != checksum for migration_id, checksum in SCHEMA_MIGRATIONS):
            return False
        cur.execute(
            """
            SELECT
                format_type(attribute.atttypid, attribute.atttypmod),
                (SELECT value FROM rag_schema_metadata WHERE key = 'schema_version'),
                (SELECT value FROM rag_schema_metadata WHERE key = 'embedding_dimension'),
                (SELECT value FROM rag_schema_metadata WHERE key = 'embedding_fingerprint'),
                (SELECT value FROM rag_schema_metadata WHERE key = 'chunking_fingerprint'),
                (SELECT relrowsecurity FROM pg_class WHERE oid = 'documents'::regclass),
                (SELECT relforcerowsecurity FROM pg_class WHERE oid = 'documents'::regclass),
                EXISTS (
                    SELECT 1 FROM pg_roles
                    WHERE rolname = 'rag_app' AND NOT rolsuper AND NOT rolbypassrls
                ),
                EXISTS (
                    SELECT 1 FROM pg_policies
                    WHERE schemaname = 'public'
                      AND tablename = 'documents'
                      AND policyname = 'documents_tenant_isolation'
                ),
                EXISTS (
                    SELECT 1
                    FROM pg_constraint AS constraint_row
                    WHERE constraint_row.conrelid = 'public.documents'::regclass
                      AND constraint_row.contype = 'p'
                      AND (
                        SELECT array_agg(attribute.attname ORDER BY key_column.ordinality)
                        FROM unnest(constraint_row.conkey) WITH ORDINALITY AS key_column(attnum, ordinality)
                        JOIN pg_attribute AS attribute
                          ON attribute.attrelid = constraint_row.conrelid
                         AND attribute.attnum = key_column.attnum
                      ) = ARRAY['tenant_id', 'id']::name[]
                )
            FROM pg_attribute AS attribute
            WHERE attribute.attrelid = 'public.documents'::regclass
              AND attribute.attname = 'embedding'
              AND NOT attribute.attisdropped
            """
        )
        row = cur.fetchone()
        expected_dimension = _embedding_dimension()
        runtime_metadata = _runtime_schema_metadata()
        return bool(
            row
            and row[0] == f"vector({expected_dimension})"
            and row[1] == SCHEMA_VERSION
            and row[2] == str(expected_dimension)
            and row[3] == runtime_metadata["embedding_fingerprint"]
            and row[4] == runtime_metadata["chunking_fingerprint"]
            and row[5] is True
            and row[6] is True
            and row[7] is True
            and row[8] is True
            and row[9] is True
        )
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


async def check_postgres_health() -> bool:
    return await _execute_sync(_check_postgres_health_sync)


def _row_to_doc(row: dict, score_key: str = "score") -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    content = metadata.get("parent_content") or row["content"]
    return {
        "id": row["id"],
        "content": content,
        "child_content": row["content"],
        "score": float(row.get(score_key) or 0.0),
        "metadata": metadata,
        "partition": row.get("partition"),
        "chunk_index": int(metadata.get("chunk_index", 0) or 0),
    }


def _add_documents_sync(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict] | None,
    partition: str,
) -> int:
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor()
        tenant_id = _set_tenant(cur)
        metadatas = metadatas or [{} for _ in ids]
        records = []
        for index, doc_id in enumerate(ids):
            embedding = f"[{','.join(str(x) for x in embeddings[index])}]"
            records.append(
                (
                    doc_id,
                    tenant_id,
                    documents[index],
                    embedding,
                    json.dumps(metadatas[index], ensure_ascii=False),
                    partition,
                )
            )

        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO documents (id, tenant_id, content, embedding, metadata, partition)
            VALUES (%s, %s::uuid, %s, %s::vector, %s::jsonb, %s)
            ON CONFLICT (tenant_id, id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                partition = EXCLUDED.partition,
                created_at = NOW();
            """,
            records,
        )
        conn.commit()
        return len(ids)
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


async def add_documents(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict] | None = None,
    partition: str = "general",
) -> int:
    return await _execute_sync(_add_documents_sync, ids, embeddings, documents, metadatas, partition)


def _replace_document_sync(
    source_key: str,
    filename: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
    partition: str,
) -> int:
    """Replace a source in one transaction, including legacy rows without source_key."""
    if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
        raise ValueError("Document IDs, embeddings, contents and metadata must have equal lengths")

    conn = _connection()
    cur = None
    try:
        cur = conn.cursor()
        tenant_id = _set_tenant(cur)
        cur.execute(
            """
            DELETE FROM documents
            WHERE tenant_id = %s::uuid
              AND (
                   metadata->>'source_key' = %s
                   OR (
                    metadata->>'source_key' IS NULL
                    AND metadata->>'filename' = %s
                    AND partition = %s
                   )
               )
            """,
            (tenant_id, source_key, filename, partition),
        )
        records = [
            (
                doc_id,
                tenant_id,
                documents[index],
                f"[{','.join(str(value) for value in embeddings[index])}]",
                json.dumps(metadatas[index], ensure_ascii=False),
                partition,
            )
            for index, doc_id in enumerate(ids)
        ]
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO documents
                (id, tenant_id, content, embedding, metadata, partition, created_at)
            VALUES (%s, %s::uuid, %s, %s::vector, %s::jsonb, %s, NOW())
            """,
            records,
        )
        conn.commit()
        return len(records)
    except Exception:
        conn.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


async def replace_document(
    source_key: str,
    filename: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
    partition: str = "general",
) -> int:
    return await _execute_sync(
        _replace_document_sync,
        source_key,
        filename,
        ids,
        embeddings,
        documents,
        metadatas,
        partition,
    )


def _vector_search_sync(
    query_embedding: list[float],
    top_k: int,
    partition: str | None,
) -> list[dict]:
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        tenant_id = _set_tenant(cur)
        probes = max(1, int(_postgres_config().get("ivfflat_probes", 10)))
        cur.execute("SELECT set_config('ivfflat.probes', %s, true)", (str(probes),))
        embedding = f"[{','.join(str(x) for x in query_embedding)}]"
        sql = """
            SELECT id, content, 1 - (embedding <=> %s::vector) AS score, metadata, partition
            FROM documents
            WHERE tenant_id = %s::uuid
              AND embedding IS NOT NULL
        """
        params: list[Any] = [embedding, tenant_id]
        if partition:
            sql += " AND partition = %s"
            params.append(partition)
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([embedding, top_k])
        cur.execute(sql, params)
        return [_row_to_doc(row) for row in cur.fetchall()]
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


async def vector_search(
    query_embedding: list[float],
    top_k: int = 10,
    partition: str | None = None,
) -> list[dict]:
    return await _execute_sync(_vector_search_sync, query_embedding, top_k, partition)


def _get_documents_by_ids_sync(ids: list[str], partition: str | None) -> list[dict]:
    if not ids:
        return []
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        tenant_id = _set_tenant(cur)
        sql = """
            SELECT id, content, metadata, partition
            FROM documents
            WHERE tenant_id = %s::uuid
              AND (id = ANY(%s) OR metadata->>'semantic_chunk_id' = ANY(%s))
        """
        params: list[Any] = [tenant_id, ids, ids]
        if partition:
            sql += " AND partition = %s"
            params.append(partition)
        sql += """
            ORDER BY COALESCE(
                array_position(%s::text[], metadata->>'semantic_chunk_id'),
                array_position(%s::text[], id)
            )
        """
        params.extend([ids, ids])
        cur.execute(sql, params)
        docs = [_row_to_doc(row) for row in cur.fetchall()]
        for doc in docs:
            doc["score"] = 1.0
            doc["mapped_article_exact_match"] = True
        return docs
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


async def get_documents_by_ids(ids: list[str], partition: str | None = None) -> list[dict]:
    """Fetch controlled statute chunks by row ID or semantic identity."""
    return await _execute_sync(_get_documents_by_ids_sync, ids, partition)


def _get_documents_by_citations_sync(
    citation_pairs: list[tuple[str, str]], partition: str | None
) -> list[dict]:
    if not citation_pairs:
        return []
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        tenant_id = _set_tenant(cur)
        pair_conditions = " OR ".join(
            "(metadata->>'law_name' = %s AND metadata->>'article_number' = %s)"
            for _ in citation_pairs
        )
        sql = f"""
            SELECT id, content, metadata, partition
            FROM documents
            WHERE tenant_id = %s::uuid
              AND ({pair_conditions})
        """
        params: list[Any] = [tenant_id]
        params.extend(value for pair in citation_pairs for value in pair)
        if partition:
            sql += " AND partition = %s"
            params.append(partition)
        ordered_keys = [f"{law}\x1f{article}" for law, article in citation_pairs]
        sql += """
            ORDER BY array_position(
                %s::text[],
                (metadata->>'law_name') || E'\\x1f' || (metadata->>'article_number')
            )
        """
        params.append(ordered_keys)
        cur.execute(sql, params)
        docs = [_row_to_doc(row) for row in cur.fetchall()]
        for doc in docs:
            doc["score"] = 1.0
            doc["explicit_citation_exact_match"] = True
        return docs
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


async def get_documents_by_citations(
    citation_pairs: list[tuple[str, str]], partition: str | None = None
) -> list[dict]:
    """Fetch provisions explicitly named by law and article in the query."""
    return await _execute_sync(_get_documents_by_citations_sync, citation_pairs, partition)


def _extract_chinese_keywords(query: str) -> list[str]:
    """Extract higher-signal keywords for legal-style Chinese queries."""
    import re

    cleaned = re.sub(r"[，。？！、\s?!.,;；：:\"'（）()\n]+", " ", query).strip()

    stop_phrases = [
        "根据",
        "中华人民共和国",
        "什么",
        "如何",
        "哪些",
        "哪个",
        "是否",
        "可以",
        "应当",
        "需要",
        "有关",
        "情形",
        "规定",
        "核心区别",
        "区别",
        "张三",
        "李四",
        "王五",
        "但在",
        "过程",
        "实际伤害",
    ]
    for phrase in sorted(stop_phrases, key=len, reverse=True):
        cleaned = cleaned.replace(phrase, " ")

    law_titles = [
        "中华人民共和国民法典",
        "中华人民共和国刑法",
        "中华人民共和国劳动合同法",
        "民法典",
        "刑法",
        "劳动合同法",
    ]

    keywords: list[str] = [title for title in law_titles if title in query]
    keywords.extend(
        re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+(?:_[\u4e00-\u9fffA-Za-z0-9]+)+", query)
    )
    keywords.extend(re.findall(r"第[一二三四五六七八九十百千万零〇两0-9]+条", query))
    keywords.extend(re.findall(r"[\u4e00-\u9fff]{2,12}(?:罪|合同|劳动合同|解除劳动合同|定义|刑罚)", query))

    parts = [p.strip() for p in cleaned.split() if len(p.strip()) >= 2]
    for part in parts:
        if len(part) <= 8:
            keywords.append(part)
        else:
            for size in (6, 4, 3, 2):
                for i in range(0, len(part) - size + 1):
                    keywords.append(part[i : i + size])

    seen = set()
    unique: list[str] = []
    for keyword in keywords:
        keyword = keyword.strip()
        if len(keyword) < 2:
            continue
        if any(token in keyword for token in ("什么", "区别是", "的核心", "但在犯罪", "张三")):
            continue
        if "与" in keyword and len(keyword) > 3:
            continue
        if keyword not in seen:
            seen.add(keyword)
            unique.append(keyword)
    return unique[:20]


def _keyword_match_weight(keyword: str) -> float:
    """Assign stronger weights to titles, article numbers and offense names."""
    if "_" in keyword and keyword.endswith("条"):
        return 1.5
    if keyword.startswith("第") and keyword.endswith("条"):
        return 1.2
    if keyword in {
        "职务上的便利",
        "本单位财物",
        "非法占为己有",
        "盗窃公私财物",
        "被胁迫参加犯罪",
        "减轻处罚",
        "免除处罚",
        "用人单位的工作人员",
        "执行工作任务",
        "用人单位承担侵权责任",
    }:
        return 1.1
    if keyword.endswith("罪"):
        return 1.0
    if keyword in {
        "中华人民共和国民法典",
        "中华人民共和国刑法",
        "中华人民共和国劳动合同法",
        "民法典",
        "刑法",
        "劳动合同法",
    }:
        return 0.9
    if len(keyword) >= 6:
        return 0.6
    if len(keyword) >= 4:
        return 0.4
    return 0.25


def _keyword_search_sync(query: str, top_k: int, partition: str | None) -> list[dict]:
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        tenant_id = _set_tenant(cur)
        keywords = _extract_chinese_keywords(query)
        if not keywords:
            return []

        search_expr = SEARCH_TEXT_EXPRESSION
        weights = [_keyword_match_weight(keyword) for keyword in keywords]
        score_parts = " + ".join(
            [f"CASE WHEN {search_expr} ILIKE %s THEN {weight} ELSE 0 END" for weight in weights]
        )
        where_parts = " OR ".join([f"{search_expr} ILIKE %s" for _ in keywords])

        sql = f"""
            SELECT
                id,
                content,
                ({score_parts}) AS score,
                metadata,
                partition
            FROM documents
            WHERE tenant_id = %s::uuid
              AND ({where_parts})
        """

        params: list[Any] = [f"%{kw}%" for kw in keywords]
        params.append(tenant_id)
        params.extend(f"%{kw}%" for kw in keywords)
        if partition:
            sql += " AND partition = %s"
            params.append(partition)

        sql += " ORDER BY score DESC LIMIT %s"
        params.append(top_k)

        cur.execute(sql, params)
        rows = cur.fetchall()
        logger.info(
            "Keyword search matched %s rows",
            len(rows),
            extra={"query": query, "keywords": keywords[:8]},
        )
        return [_row_to_doc(row) for row in rows]
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


async def bm25_search(query: str, top_k: int = 10, partition: str | None = None) -> list[dict]:
    """Keyword retrieval implemented with ILIKE-based legal term matching."""
    return await _execute_sync(_keyword_search_sync, query, top_k, partition)


def _rrf_fusion(vector_results: list[dict], keyword_results: list[dict], k: int = 60) -> list[dict]:
    scores: dict[str, dict[str, Any]] = {}

    for rank, doc in enumerate(vector_results, 1):
        scores[doc["id"]] = {
            "doc": doc,
            "rrf_score": 1.0 / (k + rank),
            "vector_rank": rank,
            "vector_score": doc.get("score", 0.0),
            "bm25_rank": None,
            "bm25_score": 0.0,
        }

    for rank, doc in enumerate(keyword_results, 1):
        info = scores.setdefault(
            doc["id"],
            {
                "doc": doc,
                "rrf_score": 0.0,
                "vector_rank": None,
                "vector_score": 0.0,
                "bm25_rank": None,
                "bm25_score": 0.0,
            },
        )
        info["rrf_score"] += 1.0 / (k + rank)
        info["bm25_rank"] = rank
        info["bm25_score"] = doc.get("score", 0.0)

    fused = sorted(scores.values(), key=lambda item: item["rrf_score"], reverse=True)
    results: list[dict] = []
    for item in fused:
        doc = item["doc"].copy()
        doc.update(
            {
                "rrf_score": item["rrf_score"],
                "vector_rank": item["vector_rank"],
                "vector_score": item["vector_score"],
                "bm25_rank": item["bm25_rank"],
                "bm25_score": item["bm25_score"],
                "score": item["rrf_score"],
            }
        )
        results.append(doc)
    return results


def _dynamic_topk(results: list[dict], max_k: int, threshold_ratio: float) -> list[dict]:
    if not results:
        return []
    best = float(results[0].get("rrf_score", results[0].get("score", 0.0)))
    cutoff = best * threshold_ratio
    kept = []
    for doc in results[:max_k]:
        score = float(doc.get("rrf_score", doc.get("score", 0.0)))
        if score < cutoff:
            break
        kept.append(doc)
    return kept or results[:1]


async def hybrid_search(
    query: str,
    query_embedding: list[float],
    top_k: int = 10,
    partition: str | None = None,
    enable_rrf: bool = True,
    enable_dynamic_topk: bool = True,
    rrf_k: int = 60,
    threshold_ratio: float = 0.5,
) -> list[dict]:
    candidate_k = max(top_k * 2, 10)
    vector_task = vector_search(query_embedding, candidate_k, partition)
    keyword_task = bm25_search(query, candidate_k, partition)
    vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

    if enable_rrf:
        results = _rrf_fusion(vector_results, keyword_results, k=rrf_k)
    else:
        results = vector_results[:top_k]

    if enable_dynamic_topk:
        results = _dynamic_topk(results, max_k=top_k, threshold_ratio=threshold_ratio)
    else:
        results = results[:top_k]

    logger.info(
        "Hybrid search finished: vector=%s keyword=%s returned=%s",
        len(vector_results),
        len(keyword_results),
        len(results),
    )
    return results


def _delete_document_sync(document_id: str) -> bool:
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor()
        tenant_id = _set_tenant(cur)
        cur.execute(
            """
            DELETE FROM documents
            WHERE tenant_id = %s::uuid AND metadata->>'document_id' = %s
            """,
            (tenant_id, document_id),
        )
        deleted: bool = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


async def delete_document(document_id: str) -> bool:
    deleted = await _execute_sync(_delete_document_sync, document_id)
    if not deleted:
        return False
    try:
        from app.utils.cache import invalidate_semantic_cache

        await invalidate_semantic_cache()
    except Exception:
        logger.debug("Semantic cache invalidation unavailable", exc_info=True)
    return True


def _list_documents_sync(skip: int, limit: int) -> list[dict]:
    skip = max(0, int(skip))
    limit = max(1, min(int(limit), 1000))
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        tenant_id = _set_tenant(cur)
        cur.execute(
            """
            SELECT
                metadata->>'document_id' AS document_id,
                metadata->>'filename' AS filename,
                partition,
                COUNT(*) AS chunk_count,
                MAX(created_at) AS updated_at
            FROM documents
            WHERE tenant_id = %s::uuid
              AND metadata->>'document_id' IS NOT NULL
            GROUP BY metadata->>'document_id', metadata->>'filename', partition
            ORDER BY MAX(created_at) DESC
            OFFSET %s LIMIT %s
            """,
            (tenant_id, skip, limit),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


async def list_documents(skip: int = 0, limit: int = 100) -> list[dict]:
    return await _execute_sync(_list_documents_sync, skip, limit)


def _count_documents_sync() -> int:
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor()
        tenant_id = _set_tenant(cur)
        cur.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT metadata->>'document_id', metadata->>'filename', partition
                FROM documents
                WHERE tenant_id = %s::uuid
                  AND metadata->>'document_id' IS NOT NULL
                GROUP BY metadata->>'document_id', metadata->>'filename', partition
            ) AS grouped_documents
            """,
            (tenant_id,),
        )
        return int(cur.fetchone()[0])
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


async def count_documents() -> int:
    return await _execute_sync(_count_documents_sync)


def _row_to_chunk(row: dict) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    content = row.get("content") or ""
    return {
        "id": row["id"],
        "content": content,
        "content_length": len(content),
        "metadata": metadata,
        "partition": row.get("partition"),
        "chunk_index": int(metadata.get("chunk_index", 0) or 0),
        "chunk_strategy": metadata.get("chunk_strategy"),
        "created_at": row.get("created_at"),
    }


def _list_document_chunks_sync(document_id: str, skip: int, limit: int) -> dict[str, Any]:
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        tenant_id = _set_tenant(cur)
        cur.execute(
            """
            SELECT COUNT(*) AS total
            FROM documents
            WHERE tenant_id = %s::uuid
              AND metadata->>'document_id' = %s
            """,
            (tenant_id, document_id),
        )
        total = int(cur.fetchone()["total"])

        cur.execute(
            """
            SELECT id, content, metadata, partition, created_at
            FROM documents
            WHERE tenant_id = %s::uuid
              AND metadata->>'document_id' = %s
            ORDER BY
                CASE
                    WHEN metadata->>'chunk_index' ~ '^[0-9]+$'
                    THEN (metadata->>'chunk_index')::int
                    ELSE 0
                END,
                id
            OFFSET %s LIMIT %s
            """,
            (tenant_id, document_id, skip, limit),
        )
        return {
            "total": total,
            "chunks": [_row_to_chunk(row) for row in cur.fetchall()],
        }
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


async def list_document_chunks(
    document_id: str,
    skip: int = 0,
    limit: int = 2000,
) -> dict[str, Any]:
    return await _execute_sync(_list_document_chunks_sync, document_id, skip, limit)
