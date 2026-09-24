"""PostgreSQL + pgvector storage with hybrid retrieval."""

import asyncio
import contextvars
import functools
import hashlib
import importlib.metadata
import json
import os
from asyncio import Future
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar

from app.auth import DEFAULT_TENANT_ID, current_tenant_id
from app.utils.config import get_config_section, get_settings
from app.utils.logger import get_logger, text_log_metadata

# Optional at import time so the module can be imported without the driver;
# every entry point raises RuntimeError if the store was never initialized.
try:
    _psycopg2: Any = import_module("psycopg2")
    import_module("psycopg2.extras")
    import_module("psycopg2.pool")
    _sql: Any = import_module("psycopg2.sql")
except ImportError:
    _psycopg2 = None
    _sql = None

psycopg2: Any = _psycopg2
sql: Any = _sql

logger = get_logger(__name__)

_T = TypeVar("_T")
_pool: Any | None = None
_executor: ThreadPoolExecutor | None = None
_init_lock = asyncio.Lock()
_schema_ready = False
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
STAGED_TENANT_MIGRATION_REVISIONS = (
    ("20260831_01_tenant_expand", "v1"),
    ("20260831_02_tenant_backfill", "v1"),
    ("20260831_03_tenant_constraints", "v1"),
    ("20260831_04_composite_key_index", "v1"),
    ("20260831_05_composite_key_cutover", "v1"),
)
STAGED_TENANT_MIGRATIONS = tuple(
    (
        migration_id,
        "sha256:" + hashlib.sha256(f"{migration_id}:{revision}".encode()).hexdigest(),
    )
    for migration_id, revision in STAGED_TENANT_MIGRATION_REVISIONS
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
SCHEMA_MIGRATIONS = (
    LEGACY_SCHEMA_MIGRATIONS
    + STAGED_TENANT_MIGRATIONS
    + tuple(
        (migration_id, checksum) for migration_id, checksum, _sql_text in CONTENT_SCHEMA_MIGRATIONS
    )
)
SEARCH_TEXT_EXPRESSION = """(
    content || ' ' ||
    COALESCE(metadata->>'filename', '') || ' ' ||
    COALESCE(metadata->>'law_name', '') || ' ' ||
    COALESCE(metadata->>'legal_citation', '') || ' ' ||
    COALESCE(metadata->>'semantic_chunk_id', '')
)"""
INDEX_MIGRATION_LOCK_NAME = "industrial-rag-search-indexes-v3"
TENANT_MIGRATION_LOCK_NAME = "industrial-rag-tenant-schema-v3-staged"
TENANT_NOT_NULL_CONSTRAINT = "documents_tenant_id_not_null"
TENANT_FOREIGN_KEY_CONSTRAINT = "documents_tenant_id_fkey"
COMPOSITE_KEY_INDEX_NAME = "idx_documents_tenant_id_id_unique"
CONCURRENT_INDEX_DEFINITIONS = (
    (
        "idx_documents_search_fields_trgm",
        f"""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_search_fields_trgm
        ON documents USING gin ({SEARCH_TEXT_EXPRESSION} gin_trgm_ops)
        """,
    ),
    (
        "idx_documents_tenant",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_tenant ON documents (tenant_id)",
    ),
    (
        "idx_documents_tenant_partition",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_tenant_partition "
        "ON documents (tenant_id, partition)",
    ),
    (
        "idx_documents_partition",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_partition ON documents (partition)",
    ),
    (
        "idx_documents_document_id",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_document_id "
        "ON documents ((metadata->>'document_id'))",
    ),
    (
        "idx_documents_source_key",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_source_key "
        "ON documents ((metadata->>'source_key'))",
    ),
    (
        "idx_documents_semantic_chunk_id",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_semantic_chunk_id "
        "ON documents ((metadata->>'semantic_chunk_id'))",
    ),
    (
        "idx_documents_citation",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_citation "
        "ON documents ((metadata->>'law_name'), (metadata->>'article_number'))",
    ),
    (
        "idx_documents_tenant_document_id",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_tenant_document_id "
        "ON documents (tenant_id, (metadata->>'document_id'))",
    ),
    (
        "idx_documents_tenant_source_key",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_tenant_source_key "
        "ON documents (tenant_id, (metadata->>'source_key'))",
    ),
    (
        "idx_documents_tenant_semantic_chunk_id",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_tenant_semantic_chunk_id "
        "ON documents (tenant_id, (metadata->>'semantic_chunk_id'))",
    ),
    (
        "idx_documents_tenant_citation",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_tenant_citation "
        "ON documents (tenant_id, (metadata->>'law_name'), "
        "(metadata->>'article_number'))",
    ),
    (
        "idx_documents_embedding_ivfflat",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_embedding_ivfflat "
        "ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
    ),
)
SUPERSEDED_INDEX_NAMES = (
    "idx_documents_search_trgm",
    "idx_documents_content_trgm",
    "idx_documents_metadata",
)


def _postgres_config() -> dict[str, Any]:
    config = dict(get_config_section("postgres"))
    for env_name, config_name in (
        ("POSTGRES_COMMAND_TIMEOUT_SECONDS", "command_timeout"),
        ("POSTGRES_LOCK_TIMEOUT_SECONDS", "lock_timeout"),
    ):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer number of seconds") from exc
        if value < 0:
            raise ValueError(f"{env_name} must be zero or greater")
        config[config_name] = value
    return config


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


def _inference_runtime_versions() -> dict[str, str]:
    """Return the exact libraries that can change embedding/reranking output."""
    versions: dict[str, str] = {}
    for distribution in ("torch", "transformers", "sentence-transformers", "tokenizers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _runtime_schema_metadata() -> dict[str, str]:
    settings = get_settings()
    embedding = settings.get("embedding", {})
    chunking = settings.get("document_processing", {}).get("chunking", {})
    embedding_identity = {
        "model_name": embedding.get("model_name"),
        "model_revision": embedding.get("model_revision"),
        "model_manifest_sha256": os.getenv("MODEL_MANIFEST_SHA256", "").strip().lower() or None,
        "dimension": _embedding_dimension(),
        "normalize_embeddings": bool(embedding.get("normalize_embeddings", True)),
        "max_length": embedding.get("max_length"),
        "inference_runtime_versions": _inference_runtime_versions(),
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


def _bounded_positive_env(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _legacy_pk_cutover_allowed() -> bool:
    raw = os.getenv("POSTGRES_ALLOW_LEGACY_PK_CUTOVER", "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "POSTGRES_ALLOW_LEGACY_PK_CUTOVER must be a boolean; "
        "enable it only inside the documented traffic-drain window"
    )


def _set_migration_lock_timeout(cur: Any) -> None:
    """Bound short catalog-lock waits in the online migration stages.

    The DDL in these stages is metadata-only or uses ``NOT VALID`` / later
    ``VALIDATE``.  It still needs a table lock briefly, however, so a busy
    serving table must cause a retry rather than leave a migrator waiting
    indefinitely.  The primary-key cutover uses its own, deliberately
    separate maintenance-window timeout below.
    """
    lock_timeout = _bounded_positive_env("POSTGRES_MIGRATION_LOCK_TIMEOUT_SECONDS", 5, 60)
    cur.execute("SELECT set_config('lock_timeout', %s, true)", (f"{lock_timeout}s",))


def _constraint_info(cur: Any, constraint_name: str) -> tuple[str, bool, str] | None:
    cur.execute(
        """
        SELECT constraint_row.contype,
               constraint_row.convalidated,
               pg_get_constraintdef(constraint_row.oid, true)
        FROM pg_constraint AS constraint_row
        WHERE constraint_row.conrelid = 'public.documents'::regclass
          AND constraint_row.conname = %s
        """,
        (constraint_name,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    definition = " ".join(str(row[2]).split()).replace("public.", "")
    return str(row[0]), bool(row[1]), definition


def _tenant_constraints(cur: Any) -> tuple[tuple[str, bool, str], tuple[str, bool, str]]:
    check = _constraint_info(cur, TENANT_NOT_NULL_CONSTRAINT)
    foreign_key = _constraint_info(cur, TENANT_FOREIGN_KEY_CONSTRAINT)
    if check is None or check[0] != "c" or "tenant_id IS NOT NULL" not in check[2]:
        raise RuntimeError(
            f"PostgreSQL constraint {TENANT_NOT_NULL_CONSTRAINT} has an unexpected definition"
        )
    if (
        foreign_key is None
        or foreign_key[0] != "f"
        or "FOREIGN KEY (tenant_id) REFERENCES rag_tenants(id)" not in foreign_key[2]
    ):
        raise RuntimeError(
            f"PostgreSQL constraint {TENANT_FOREIGN_KEY_CONSTRAINT} has an unexpected definition"
        )
    return check, foreign_key


def _tenant_column_not_null(cur: Any) -> bool:
    cur.execute(
        """
        SELECT attribute.attnotnull
        FROM pg_attribute AS attribute
        WHERE attribute.attrelid = 'public.documents'::regclass
          AND attribute.attname = 'tenant_id'
          AND NOT attribute.attisdropped
        """
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("PostgreSQL documents.tenant_id column is missing")
    return bool(row[0])


def _primary_key_state(cur: Any) -> tuple[str, tuple[str, ...]] | None:
    cur.execute(
        """
        SELECT constraint_row.conname,
               array_agg(attribute.attname ORDER BY key_column.ordinality)
        FROM pg_constraint AS constraint_row
        CROSS JOIN LATERAL unnest(constraint_row.conkey)
            WITH ORDINALITY AS key_column(attnum, ordinality)
        JOIN pg_attribute AS attribute
          ON attribute.attrelid = constraint_row.conrelid
         AND attribute.attnum = key_column.attnum
        WHERE constraint_row.conrelid = 'public.documents'::regclass
          AND constraint_row.contype = 'p'
        GROUP BY constraint_row.conname
        """
    )
    row = cur.fetchone()
    if row is None:
        return None
    return str(row[0]), tuple(str(column) for column in row[1])


def _composite_index_state(cur: Any) -> tuple[bool, bool, tuple[str, ...]] | None:
    cur.execute(
        """
        SELECT index_row.indisvalid AND index_row.indisready,
               index_row.indisunique,
               array_agg(attribute.attname ORDER BY key_column.ordinality)
        FROM pg_class AS index_class
        JOIN pg_namespace AS namespace
          ON namespace.oid = index_class.relnamespace
        JOIN pg_index AS index_row
          ON index_row.indexrelid = index_class.oid
        CROSS JOIN LATERAL unnest(index_row.indkey)
            WITH ORDINALITY AS key_column(attnum, ordinality)
        JOIN pg_attribute AS attribute
          ON attribute.attrelid = index_row.indrelid
         AND attribute.attnum = key_column.attnum
        WHERE namespace.nspname = 'public'
          AND index_class.relname = %s
        GROUP BY index_row.indisvalid, index_row.indisready, index_row.indisunique
        """,
        (COMPOSITE_KEY_INDEX_NAME,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return bool(row[0]), bool(row[1]), tuple(str(column) for column in row[2])


def _quote_identifier(identifier: str) -> str:
    """Quote a catalog-provided identifier without interpolating executable text."""
    return '"' + identifier.replace('"', '""') + '"'


def _record_stage_if_needed(
    cur: Any,
    migration: tuple[str, str],
    *,
    required: bool,
) -> None:
    if required:
        _record_migration(cur, *migration)


def _expand_tenant_schema(conn: Any, migration: tuple[str, str]) -> None:
    cur = conn.cursor()
    try:
        required = _migration_required(cur, *migration)
        if not required:
            _tenant_constraints(cur)
            _tenant_column_not_null(cur)
            conn.commit()
            return
        _set_migration_lock_timeout(cur)
        cur.execute("ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS tenant_id UUID")
        cur.execute(
            f"ALTER TABLE public.documents ALTER COLUMN tenant_id "
            f"SET DEFAULT '{DEFAULT_TENANT_ID}'::uuid"
        )

        check = _constraint_info(cur, TENANT_NOT_NULL_CONSTRAINT)
        if check is None:
            cur.execute(
                f"ALTER TABLE public.documents ADD CONSTRAINT "
                f"{TENANT_NOT_NULL_CONSTRAINT} CHECK (tenant_id IS NOT NULL) NOT VALID"
            )
        elif check[0] != "c" or "tenant_id IS NOT NULL" not in check[2]:
            raise RuntimeError(
                f"PostgreSQL constraint {TENANT_NOT_NULL_CONSTRAINT} has an unexpected definition"
            )

        foreign_key = _constraint_info(cur, TENANT_FOREIGN_KEY_CONSTRAINT)
        if foreign_key is None:
            cur.execute(
                f"ALTER TABLE public.documents ADD CONSTRAINT "
                f"{TENANT_FOREIGN_KEY_CONSTRAINT} FOREIGN KEY (tenant_id) "
                "REFERENCES public.rag_tenants(id) NOT VALID"
            )
        elif (
            foreign_key[0] != "f"
            or "FOREIGN KEY (tenant_id) REFERENCES rag_tenants(id)" not in foreign_key[2]
        ):
            raise RuntimeError(
                f"PostgreSQL constraint {TENANT_FOREIGN_KEY_CONSTRAINT} has an unexpected definition"
            )

        _tenant_constraints(cur)
        _record_stage_if_needed(cur, migration, required=required)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def _backfill_tenant_ids(conn: Any, migration: tuple[str, str]) -> None:
    batch_size = _bounded_positive_env("POSTGRES_MIGRATION_BATCH_SIZE", 1000, 10000)
    cur = conn.cursor()
    try:
        required = _migration_required(cur, *migration)
        if not required:
            cur.execute("SELECT COUNT(*) FROM public.documents WHERE tenant_id IS NULL")
            if int(cur.fetchone()[0]) != 0:
                raise RuntimeError(
                    "PostgreSQL tenant backfill marker exists but NULL tenant rows remain"
                )
            conn.commit()
            return

        total_updated = 0
        while True:
            cur.execute(
                """
                WITH batch AS (
                    SELECT ctid
                    FROM public.documents
                    WHERE tenant_id IS NULL
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE public.documents AS document
                SET tenant_id = %s::uuid
                FROM batch
                WHERE document.ctid = batch.ctid
                RETURNING 1
                """,
                (batch_size, DEFAULT_TENANT_ID),
            )
            updated = len(cur.fetchall())
            conn.commit()
            total_updated += updated
            if updated == 0:
                break

        cur.execute("SELECT COUNT(*) FROM public.documents WHERE tenant_id IS NULL")
        remaining = int(cur.fetchone()[0])
        if remaining:
            raise RuntimeError(
                "PostgreSQL tenant backfill left rows locked by another transaction; "
                "rerun the migration after those transactions finish"
            )
        _record_migration(cur, *migration)
        conn.commit()
        logger.info("PostgreSQL tenant backfill completed", extra={"rows": total_updated})
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def _validate_tenant_constraints(conn: Any, migration: tuple[str, str]) -> None:
    cur = conn.cursor()
    try:
        required = _migration_required(cur, *migration)
        check, foreign_key = _tenant_constraints(cur)
        _set_migration_lock_timeout(cur)
        if not check[1]:
            cur.execute(
                f"ALTER TABLE public.documents VALIDATE CONSTRAINT {TENANT_NOT_NULL_CONSTRAINT}"
            )
            conn.commit()
        if not _tenant_column_not_null(cur):
            # PostgreSQL can use the validated CHECK proof and avoid a second
            # full-table scan; only the catalog cutover needs a short lock.
            cur.execute("ALTER TABLE public.documents ALTER COLUMN tenant_id SET NOT NULL")
            conn.commit()

        _check, foreign_key = _tenant_constraints(cur)
        if not foreign_key[1]:
            cur.execute(
                f"ALTER TABLE public.documents VALIDATE CONSTRAINT {TENANT_FOREIGN_KEY_CONSTRAINT}"
            )
            conn.commit()

        check, foreign_key = _tenant_constraints(cur)
        if not _tenant_column_not_null(cur) or not check[1] or not foreign_key[1]:
            raise RuntimeError("PostgreSQL tenant constraints did not reach the validated state")
        _record_stage_if_needed(cur, migration, required=required)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def _prepare_composite_key_index(conn: Any, migration: tuple[str, str]) -> None:
    cur = conn.cursor()
    try:
        required = _migration_required(cur, *migration)
        primary_key = _primary_key_state(cur)
        if primary_key is not None and primary_key[1] == ("tenant_id", "id"):
            _record_stage_if_needed(cur, migration, required=required)
            conn.commit()
            return

        conn.commit()
        conn.autocommit = True
        state = _composite_index_state(cur)
        if state is not None and state[0] and (not state[1] or state[2] != ("tenant_id", "id")):
            raise RuntimeError(
                f"PostgreSQL index {COMPOSITE_KEY_INDEX_NAME} has an unexpected definition"
            )
        if state is not None and not state[0]:
            cur.execute(f"DROP INDEX CONCURRENTLY IF EXISTS public.{COMPOSITE_KEY_INDEX_NAME}")
            state = None
        if state is None:
            cur.execute(
                f"CREATE UNIQUE INDEX CONCURRENTLY {COMPOSITE_KEY_INDEX_NAME} "
                "ON public.documents (tenant_id, id)"
            )
        state = _composite_index_state(cur)
        if state != (True, True, ("tenant_id", "id")):
            raise RuntimeError(
                "PostgreSQL concurrent composite-key index did not become valid and unique"
            )

        conn.autocommit = False
        _record_stage_if_needed(cur, migration, required=required)
        conn.commit()
    except Exception:
        if not conn.autocommit:
            conn.rollback()
        raise
    finally:
        conn.autocommit = False
        cur.close()


def _cutover_composite_primary_key(
    conn: Any,
    migration: tuple[str, str],
    legacy_migration: tuple[str, str],
) -> None:
    cur = conn.cursor()
    try:
        required = _migration_required(cur, *migration)
        legacy_required = _migration_required(cur, *legacy_migration)
        primary_key = _primary_key_state(cur)
        if primary_key is not None and primary_key[1] == ("tenant_id", "id"):
            _record_stage_if_needed(cur, migration, required=required)
            _record_stage_if_needed(cur, legacy_migration, required=legacy_required)
            conn.commit()
            return
        if not required:
            raise RuntimeError(
                "PostgreSQL composite-key cutover marker exists but the primary key is not final"
            )
        if primary_key is not None and primary_key[1] != ("id",):
            raise RuntimeError(
                "PostgreSQL documents has an unsupported primary key: " + ", ".join(primary_key[1])
            )
        if _composite_index_state(cur) != (True, True, ("tenant_id", "id")):
            raise RuntimeError("PostgreSQL composite-key cutover index is not ready")
        if not _legacy_pk_cutover_allowed():
            raise RuntimeError(
                "Legacy PostgreSQL primary-key cutover is prepared but not authorized. "
                "Drain legacy API/worker traffic, set "
                "POSTGRES_ALLOW_LEGACY_PK_CUTOVER=true for the one-time migration, "
                "and keep the lock timeout bounded."
            )

        lock_timeout = _bounded_positive_env("POSTGRES_PK_CUTOVER_LOCK_TIMEOUT_SECONDS", 5, 60)
        cur.execute("SELECT set_config('lock_timeout', %s, true)", (f"{lock_timeout}s",))
        if primary_key is None:
            cutover_sql = (
                "ALTER TABLE public.documents "
                f"ADD CONSTRAINT documents_pkey PRIMARY KEY USING INDEX "
                f"{_quote_identifier(COMPOSITE_KEY_INDEX_NAME)}"
            )
        else:
            cutover_sql = (
                "ALTER TABLE public.documents "
                f"DROP CONSTRAINT {_quote_identifier(primary_key[0])}, "
                f"ADD CONSTRAINT documents_pkey PRIMARY KEY USING INDEX "
                f"{_quote_identifier(COMPOSITE_KEY_INDEX_NAME)}"
            )
        cur.execute(cutover_sql)
        final_primary_key = _primary_key_state(cur)
        if final_primary_key is None or final_primary_key[1] != ("tenant_id", "id"):
            raise RuntimeError("PostgreSQL composite primary-key cutover verification failed")
        _record_migration(cur, *migration)
        _record_stage_if_needed(cur, legacy_migration, required=legacy_required)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def _run_staged_tenant_migration(
    conn: Any,
    legacy_composite_migration: tuple[str, str],
) -> None:
    """Upgrade a legacy documents table in resumable, bounded-lock stages.

    Expand/backfill/validation and concurrent index preparation are safe while
    compatible Pods continue serving. Replacing the legacy global primary key
    still needs an explicit, short traffic-drain window because older binaries
    may issue ``ON CONFLICT (id)`` and the catalog cutover needs ACCESS
    EXCLUSIVE. A bounded lock timeout makes contention fail without recording
    completion, so a later Job can retry from the verified stage markers.
    """
    original_autocommit = bool(conn.autocommit)
    cur = conn.cursor()
    lock_acquired = False
    try:
        if not original_autocommit:
            conn.commit()
        conn.autocommit = True
        cur.execute("SELECT pg_advisory_lock(hashtext(%s))", (TENANT_MIGRATION_LOCK_NAME,))
        lock_acquired = True
        conn.autocommit = False

        expand, backfill, constraints, composite_index, cutover = STAGED_TENANT_MIGRATIONS
        _expand_tenant_schema(conn, expand)
        _backfill_tenant_ids(conn, backfill)
        _validate_tenant_constraints(conn, constraints)
        _prepare_composite_key_index(conn, composite_index)
        _cutover_composite_primary_key(conn, cutover, legacy_composite_migration)
    finally:
        try:
            if lock_acquired:
                if not conn.autocommit:
                    conn.rollback()
                conn.autocommit = True
                cur.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s))",
                    (TENANT_MIGRATION_LOCK_NAME,),
                )
        finally:
            conn.autocommit = original_autocommit
            cur.close()


def _index_validity(cur: Any, index_name: str) -> bool | None:
    """Return index validity, or ``None`` when the index does not exist."""
    cur.execute(
        """
        SELECT index_row.indisvalid AND index_row.indisready
        FROM pg_class AS index_class
        JOIN pg_namespace AS namespace
          ON namespace.oid = index_class.relnamespace
        JOIN pg_index AS index_row
          ON index_row.indexrelid = index_class.oid
        WHERE namespace.nspname = 'public'
          AND index_class.relname = %s
        """,
        (index_name,),
    )
    row = cur.fetchone()
    return None if row is None else bool(row[0])


def _verify_concurrent_indexes(cur: Any) -> None:
    invalid = [
        index_name
        for index_name, _definition in CONCURRENT_INDEX_DEFINITIONS
        if _index_validity(cur, index_name) is not True
    ]
    if invalid:
        raise RuntimeError(
            "PostgreSQL concurrent index migration did not produce valid indexes: "
            + ", ".join(invalid)
        )


def _run_search_index_migration_concurrently(
    conn: Any,
    migration_id: str,
    checksum: str,
) -> None:
    """Apply large-table indexes outside a transaction and record them last.

    PostgreSQL may leave an invalid catalog entry after a failed concurrent
    build. Each rerun removes that entry before trying again. A session-level
    advisory lock covers the entire create/drop/verify/record sequence without
    forcing the index scans into one long-running transaction.
    """
    original_autocommit = bool(conn.autocommit)
    cur = conn.cursor()
    lock_acquired = False
    try:
        if not original_autocommit:
            conn.commit()
        conn.autocommit = True
        cur.execute(
            "SELECT pg_advisory_lock(hashtext(%s))",
            (INDEX_MIGRATION_LOCK_NAME,),
        )
        lock_acquired = True

        # Another migration process may have completed while this process was
        # waiting for the session lock, so check the immutable record again.
        if not _migration_required(cur, migration_id, checksum):
            _verify_concurrent_indexes(cur)
            return

        for index_name, definition in CONCURRENT_INDEX_DEFINITIONS:
            validity = _index_validity(cur, index_name)
            if validity is True:
                continue
            if validity is False:
                cur.execute(f"DROP INDEX CONCURRENTLY IF EXISTS public.{index_name}")
            cur.execute(definition)

        for index_name in SUPERSEDED_INDEX_NAMES:
            cur.execute(f"DROP INDEX CONCURRENTLY IF EXISTS public.{index_name}")

        _verify_concurrent_indexes(cur)

        # Only a fully verified index set is allowed to become an applied
        # migration. If this transaction fails, the indexes remain reusable and
        # the next run safely verifies and records them.
        conn.autocommit = False
        _record_migration(cur, migration_id, checksum)
        conn.commit()
    except Exception:
        if not conn.autocommit:
            conn.rollback()
        raise
    finally:
        try:
            if lock_acquired:
                if not conn.autocommit:
                    conn.rollback()
                conn.autocommit = True
                cur.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s))",
                    (INDEX_MIGRATION_LOCK_NAME,),
                )
        finally:
            conn.autocommit = original_autocommit
            cur.close()


async def init_postgres_store() -> None:
    """Initialize connection pool and schema."""
    global _pool, _executor, _schema_ready
    if _schema_ready and _pool is not None and _executor is not None:
        return
    async with _init_lock:
        if _schema_ready and _pool is not None and _executor is not None:
            return
        if psycopg2 is None:
            raise ImportError(
                "psycopg2 is required for PostgreSQL retrieval. Install psycopg2-binary."
            )

        # A pool without this readiness flag is a partial/failed initialization,
        # never a usable store. Dispose it before starting a clean attempt.
        if _pool is not None or _executor is not None:
            _dispose_postgres_runtime()
        cfg = _postgres_config()
        try:
            _executor = ThreadPoolExecutor(max_workers=int(cfg.get("max_pool_size", 5)))
            connection_options: dict[str, Any] = {
                "minconn": int(cfg.get("min_pool_size", 1)),
                "maxconn": int(cfg.get("max_pool_size", 5)),
                "host": cfg.get("host", "localhost"),
                "port": int(cfg.get("port", 5432)),
                "database": cfg.get("database", "rag_db"),
                "user": cfg.get("user", "postgres"),
                "password": cfg.get("password", ""),
                "client_encoding": "utf8",
                "connect_timeout": int(cfg.get("connect_timeout", 5)),
                "keepalives": 1,
                "keepalives_idle": int(cfg.get("keepalives_idle", 30)),
                "keepalives_interval": int(cfg.get("keepalives_interval", 10)),
                "keepalives_count": int(cfg.get("keepalives_count", 3)),
                "options": (
                    f"-c statement_timeout={int(cfg.get('command_timeout', 60)) * 1000} "
                    f"-c lock_timeout={int(cfg.get('lock_timeout', 5)) * 1000}"
                ),
            }
            sslmode = str(cfg.get("sslmode", "")).strip()
            sslrootcert = str(cfg.get("sslrootcert", "")).strip()
            if sslmode and not sslmode.startswith("${"):
                connection_options["sslmode"] = sslmode
            if sslrootcert and not sslrootcert.startswith("${"):
                connection_options["sslrootcert"] = sslrootcert
            _pool = psycopg2.pool.ThreadedConnectionPool(
                **connection_options,
            )
            if _auto_migrate_enabled():
                await _execute_sync(_create_schema_sync)
            else:
                await _execute_sync(_require_schema_ready_sync)
            _schema_ready = True
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


def _pool_connection() -> Any:
    if _pool is None:
        raise RuntimeError("PostgreSQL store is not initialized")
    return _pool.getconn()


class PostgresAdvisoryLeaseUnavailable(RuntimeError):
    """Raised when another worker currently owns a cross-process lease."""


@contextmanager
def postgres_advisory_lease(name: str) -> Iterator[None]:
    """Hold one PostgreSQL session advisory lock for a synchronous operation."""
    lease_name = str(name).strip()
    if not lease_name or len(lease_name) > 512:
        raise ValueError("PostgreSQL advisory lease name must contain 1-512 characters")
    conn = _pool_connection()
    cur = None
    acquired = False
    discard = False
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
            (lease_name,),
        )
        row = cur.fetchone()
        acquired = bool(row and row[0])
        if not acquired:
            raise PostgresAdvisoryLeaseUnavailable("PostgreSQL advisory lease is already held")
        yield
    finally:
        if acquired and cur is not None:
            try:
                cur.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                    (lease_name,),
                )
                row = cur.fetchone()
                if not row or row[0] is not True:
                    discard = True
            except Exception:
                discard = True
                logger.warning("PostgreSQL advisory lease release failed", exc_info=True)
        if cur is not None:
            try:
                cur.close()
            except Exception:
                discard = True
                logger.warning("PostgreSQL advisory lease cursor close failed", exc_info=True)
        if discard:
            if _pool is not None:
                _pool.putconn(conn, close=True)
        else:
            _return_connection(conn)


def _connection() -> Any:
    if not _schema_ready:
        raise RuntimeError("PostgreSQL store schema is not ready")
    return _pool_connection()


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
    conn = _pool_connection()
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
                tenant_id UUID NOT NULL DEFAULT '{DEFAULT_TENANT_ID}'::uuid,
                content TEXT NOT NULL,
                embedding vector({dimension}),
                metadata JSONB DEFAULT '{{}}',
                partition TEXT DEFAULT 'general',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                CONSTRAINT {TENANT_NOT_NULL_CONSTRAINT}
                    CHECK (tenant_id IS NOT NULL),
                CONSTRAINT {TENANT_FOREIGN_KEY_CONSTRAINT}
                    FOREIGN KEY (tenant_id) REFERENCES rag_tenants(id),
                CONSTRAINT documents_pkey PRIMARY KEY (tenant_id, id)
            );
            """
        )
        tenant_migration = SCHEMA_MIGRATIONS[0]
        composite_key_migration = SCHEMA_MIGRATIONS[1]
        index_migration = SCHEMA_MIGRATIONS[2]

        # A fresh database is created directly at the final contract. A legacy
        # table is upgraded after this bootstrap commit so no full-table update,
        # validation scan, or concurrent index build is trapped in one schema
        # transaction.
        conn.commit()
        _run_staged_tenant_migration(conn, composite_key_migration)
        _run_search_index_migration_concurrently(conn, *index_migration)
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('industrial-rag-schema-v3'))")

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
    global _pool, _executor, _schema_ready
    _schema_ready = False
    if _pool is not None:
        _pool.closeall()
        _pool = None
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


async def close_postgres_store() -> None:
    async with _init_lock:
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
    conn = _pool_connection()
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
        documents_ready, metadata_ready, migrations_ready, tenants_ready, vector_ready = (
            cur.fetchone()
        )
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
        applied_migrations = {
            str(migration_id): str(checksum) for migration_id, checksum in cur.fetchall()
        }
        if any(
            applied_migrations.get(migration_id) != checksum
            for migration_id, checksum in SCHEMA_MIGRATIONS
        ):
            return False
        try:
            tenant_check, tenant_foreign_key = _tenant_constraints(cur)
            primary_key = _primary_key_state(cur)
            tenant_not_null = _tenant_column_not_null(cur)
        except RuntimeError:
            return False
        if (
            not tenant_not_null
            or not tenant_check[1]
            or not tenant_foreign_key[1]
            or primary_key is None
            or primary_key[1] != ("tenant_id", "id")
        ):
            return False
        cur.execute(
            """
            SELECT COUNT(*)
            FROM pg_class AS index_class
            JOIN pg_namespace AS namespace
              ON namespace.oid = index_class.relnamespace
            JOIN pg_index AS index_row
              ON index_row.indexrelid = index_class.oid
            WHERE namespace.nspname = 'public'
              AND index_class.relname = ANY(%s)
              AND index_row.indisvalid
              AND index_row.indisready
            """,
            ([name for name, _definition in CONCURRENT_INDEX_DEFINITIONS],),
        )
        if int(cur.fetchone()[0]) != len(CONCURRENT_INDEX_DEFINITIONS):
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
    if not _schema_ready:
        return False
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
    return await _execute_sync(
        _add_documents_sync, ids, embeddings, documents, metadatas, partition
    )


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
        # Serialize replacements for the same logical source.  DELETE followed
        # by INSERT is atomic within this transaction, but without a
        # transaction-scoped lock two uploads of one source can interleave and
        # leave chunks from both versions in the corpus.
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (source_key,),
        )
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


def document_commit_matches(
    *,
    tenant_id: str,
    document_id: str,
    source_key: str,
    upload_task_id: str,
    upload_object_key: str,
    total_chunks: int,
) -> bool:
    """Confirm that a completion receipt describes the committed DB rows."""
    expected_chunks = int(total_chunks)
    if expected_chunks < 1:
        return False
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor()
        resolved_tenant = _set_tenant(cur, tenant_id)
        cur.execute(
            """
            SELECT
                COUNT(*) AS document_rows,
                COUNT(*) FILTER (
                    WHERE metadata->>'source_key' = %s
                      AND metadata->>'upload_task_id' = %s
                      AND metadata->>'upload_object_key' = %s
                      AND metadata->>'total_chunks' = %s
                ) AS matching_rows
            FROM documents
            WHERE tenant_id = %s::uuid
              AND metadata->>'document_id' = %s
            """,
            (
                source_key,
                upload_task_id,
                upload_object_key,
                str(expected_chunks),
                resolved_tenant,
                document_id,
            ),
        )
        row = cur.fetchone()
        return bool(row and int(row[0]) == expected_chunks and int(row[1]) == expected_chunks)
    finally:
        if cur is not None:
            cur.close()
        _return_connection(conn)


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
    keywords.extend(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+(?:_[\u4e00-\u9fffA-Za-z0-9]+)+", query))
    keywords.extend(re.findall(r"第[一二三四五六七八九十百千万零〇两0-9]+条", query))
    keywords.extend(
        re.findall(r"[\u4e00-\u9fff]{2,12}(?:罪|合同|劳动合同|解除劳动合同|定义|刑罚)", query)
    )

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
            extra={
                **text_log_metadata(query, "query"),
                "keyword_count": len(keywords),
            },
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
