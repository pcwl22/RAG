"""Atomically rebuild PostgreSQL document embeddings for the current runtime.

The command deliberately bypasses normal vector-store startup because startup
must fail closed when the recorded corpus fingerprint differs.  It requires an
explicit source fingerprint and document count, computes every replacement
vector before taking a table lock, preserves the old vectors in a backup table,
and changes the fingerprint only in the same transaction as the vector swap.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding.embedder import encode_texts  # noqa: E402
from app.utils.config import get_config_section  # noqa: E402
from app.utils.strict_dotenv import apply_release_env_file  # noqa: E402
from app.vectorstore.postgres_store import _runtime_schema_metadata  # noqa: E402

_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_BACKUP_TABLE = re.compile(r"^rag_embedding_backup_[0-9]{14}$")
_VECTOR_TYPE = re.compile(r"^vector\(([0-9]+)\)$")
_EMBEDDING_INDEX = "idx_documents_embedding_ivfflat"


def _load_environment() -> None:
    configured = os.getenv("RAG_ENV_FILE", "").strip()
    env_file = Path(configured) if configured else PROJECT_ROOT / ".env"
    if os.getenv("RAG_RELEASE_SNAPSHOT", "").strip().lower() in {"1", "true", "yes", "on"}:
        apply_release_env_file(env_file)
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_file, override=False)


def _connection_options() -> dict[str, Any]:
    config = get_config_section("postgres")
    options: dict[str, Any] = {
        "host": config.get("host", "localhost"),
        "port": int(config.get("port", 5432)),
        "dbname": config.get("database", "rag_db"),
        "user": config.get("user", "postgres"),
        "password": config.get("password", ""),
        "client_encoding": "utf8",
        "connect_timeout": int(config.get("connect_timeout", 5)),
        "application_name": "industrial-rag-corpus-reembed",
    }
    sslmode = str(config.get("sslmode", "")).strip()
    sslrootcert = str(config.get("sslrootcert", "")).strip()
    if sslmode and not sslmode.startswith("${"):
        options["sslmode"] = sslmode
    if sslrootcert and not sslrootcert.startswith("${"):
        options["sslrootcert"] = sslrootcert
    return options


def _snapshot_digest(rows: list[tuple[str, str, str]]) -> str:
    digest = hashlib.sha256()
    for tenant_id, document_id, content in rows:
        for value in (tenant_id, document_id, content):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _read_snapshot(cursor: Any) -> list[tuple[str, str, str]]:
    cursor.execute(
        "SELECT tenant_id::text, id, content FROM documents ORDER BY tenant_id, id"
    )
    return [(str(tenant), str(document_id), str(content)) for tenant, document_id, content in cursor]


def _validate_fingerprint(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if not _FINGERPRINT.fullmatch(normalized):
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return normalized


def _validate_backup_table(value: str) -> str:
    if not _BACKUP_TABLE.fullmatch(value):
        raise ValueError(
            "backup table must match rag_embedding_backup_YYYYMMDDhhmmss"
        )
    return value


def _validate_dimension(value: Any, label: str) -> int:
    try:
        dimension = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if dimension < 1:
        raise ValueError(f"{label} must be a positive integer")
    return dimension


def _current_metadata(cursor: Any) -> dict[str, str]:
    cursor.execute("SELECT key, value FROM rag_schema_metadata")
    return {str(key): str(value) for key, value in cursor.fetchall()}


def _document_embedding_contract(cursor: Any) -> tuple[int, bool]:
    cursor.execute(
        """
        SELECT
            format_type(attribute.atttypid, attribute.atttypmod),
            attribute.attnotnull
        FROM pg_attribute AS attribute
        WHERE attribute.attrelid = 'public.documents'::regclass
          AND attribute.attname = 'embedding'
          AND NOT attribute.attisdropped
        """
    )
    row = cursor.fetchone()
    vector_type = str(row[0]) if row else "missing"
    match = _VECTOR_TYPE.fullmatch(vector_type)
    if match is None:
        raise RuntimeError(
            "documents.embedding must be a dimensioned pgvector column; "
            f"found {vector_type}"
        )
    return int(match.group(1)), bool(row[1])


def _document_embedding_dimension(cursor: Any) -> int:
    return _document_embedding_contract(cursor)[0]


def _corpus_identity(cursor: Any) -> dict[str, Any]:
    metadata = _current_metadata(cursor)
    metadata_dimension = _validate_dimension(
        metadata.get("embedding_dimension"), "database embedding dimension"
    )
    column_dimension, embedding_not_null = _document_embedding_contract(cursor)
    if metadata_dimension != column_dimension:
        raise RuntimeError(
            "database embedding dimension metadata does not match documents.embedding: "
            f"metadata={metadata_dimension}, column={column_dimension}"
        )
    return {
        "metadata": metadata,
        "embedding_dimension": column_dimension,
        "embedding_not_null": embedding_not_null,
        "embedding_fingerprint": _validate_fingerprint(
            metadata.get("embedding_fingerprint", ""), "database embedding fingerprint"
        ),
        "chunking_fingerprint": _validate_fingerprint(
            metadata.get("chunking_fingerprint", ""), "database chunking fingerprint"
        ),
    }


def _document_count(cursor: Any) -> int:
    cursor.execute("SELECT COUNT(*) FROM documents")
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("database did not return a document count")
    return int(row[0])


def _require_migration_identity(cursor: Any) -> None:
    cursor.execute(
        "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
    )
    row = cursor.fetchone()
    if not row or not (bool(row[0]) or bool(row[1])):
        raise RuntimeError("re-embedding requires a migration identity with BYPASSRLS")


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _default_backup_table() -> str:
    return "rag_embedding_backup_" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")


def inspect_corpus() -> dict[str, Any]:
    """Return a read-only, copyable migration plan for the configured corpus."""
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover - guarded by runtime lock
        raise RuntimeError("psycopg2-binary is required for corpus inspection") from exc

    target_metadata = _runtime_schema_metadata()
    target_fingerprint = _validate_fingerprint(
        target_metadata["embedding_fingerprint"], "target embedding fingerprint"
    )
    target_chunking_fingerprint = _validate_fingerprint(
        target_metadata["chunking_fingerprint"], "target chunking fingerprint"
    )
    target_dimension = _validate_dimension(
        target_metadata["embedding_dimension"], "target embedding dimension"
    )
    connection = psycopg2.connect(**_connection_options())
    try:
        with connection.cursor() as cursor:
            _require_migration_identity(cursor)
            source_identity = _corpus_identity(cursor)
            document_count = _document_count(cursor)
        connection.rollback()

        source_fingerprint = str(source_identity["embedding_fingerprint"])
        source_chunking_fingerprint = str(source_identity["chunking_fingerprint"])
        source_dimension = int(source_identity["embedding_dimension"])
        embedding_changed = (
            source_fingerprint != target_fingerprint or source_dimension != target_dimension
        )
        chunking_changed = source_chunking_fingerprint != target_chunking_fingerprint
        result: dict[str, Any] = {
            "status": "ready",
            "document_count": document_count,
            "source_embedding_dimension": source_dimension,
            "target_embedding_dimension": target_dimension,
            "source_embedding_fingerprint": source_fingerprint,
            "target_embedding_fingerprint": target_fingerprint,
            "source_chunking_fingerprint": source_chunking_fingerprint,
            "target_chunking_fingerprint": target_chunking_fingerprint,
        }
        if chunking_changed:
            result.update(
                {
                    "status": "reingest_required",
                    "reason": (
                        "chunking configuration changed; re-embedding existing chunks cannot "
                        "apply the new chunking policy. Re-ingest the source corpus."
                    ),
                }
            )
        elif embedding_changed:
            backup_table = _default_backup_table()
            command = (
                "python scripts/reembed_postgres_corpus.py "
                f"--from-dimension {source_dimension} "
                f"--from-fingerprint {source_fingerprint} "
                f"--from-chunking-fingerprint {source_chunking_fingerprint} "
                f"--confirm-document-count {document_count} "
                f"--backup-table {backup_table}"
            )
            result.update(
                {
                    "status": "migration_required",
                    "backup_table": backup_table,
                    "plan_command": command,
                    "apply_command": f"{command} --apply",
                }
            )
        return result
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def rebuild(
    *,
    source_dimension: int,
    source_fingerprint: str,
    source_chunking_fingerprint: str,
    expected_count: int,
    batch_size: int,
    backup_table: str,
    apply: bool,
) -> dict[str, Any]:
    try:
        import psycopg2
        from psycopg2 import sql
        from psycopg2.extras import execute_values
    except ImportError as exc:  # pragma: no cover - guarded by runtime lock
        raise RuntimeError("psycopg2-binary is required for corpus re-embedding") from exc

    source_dimension = _validate_dimension(source_dimension, "source embedding dimension")
    source_fingerprint = _validate_fingerprint(
        source_fingerprint, "source embedding fingerprint"
    )
    source_chunking_fingerprint = _validate_fingerprint(
        source_chunking_fingerprint, "source chunking fingerprint"
    )
    backup_table = _validate_backup_table(backup_table)
    if expected_count < 1:
        raise ValueError("expected document count must be positive")
    if batch_size < 1 or batch_size > 256:
        raise ValueError("batch size must be between 1 and 256")

    target_metadata = _runtime_schema_metadata()
    target_fingerprint = _validate_fingerprint(
        target_metadata["embedding_fingerprint"], "target embedding fingerprint"
    )
    target_chunking_fingerprint = _validate_fingerprint(
        target_metadata["chunking_fingerprint"], "target chunking fingerprint"
    )
    target_dimension = _validate_dimension(
        target_metadata["embedding_dimension"], "target embedding dimension"
    )
    if target_chunking_fingerprint != source_chunking_fingerprint:
        raise RuntimeError(
            "chunking fingerprint changed; re-embedding cannot apply a new chunking policy. "
            "Re-ingest the source corpus instead."
        )

    connection = psycopg2.connect(**_connection_options())
    try:
        with connection.cursor() as cursor:
            _require_migration_identity(cursor)
            source_identity = _corpus_identity(cursor)
            if (
                source_identity["embedding_dimension"] != source_dimension
                or source_identity["embedding_fingerprint"] != source_fingerprint
                or source_identity["chunking_fingerprint"] != source_chunking_fingerprint
            ):
                raise RuntimeError(
                    "database corpus identity changed; refusing the requested migration"
                )
            rows = _read_snapshot(cursor)
        connection.rollback()
        if len(rows) != expected_count:
            raise RuntimeError(
                f"database document count changed: expected {expected_count}, found {len(rows)}"
            )
        source_digest = _snapshot_digest(rows)
        if (
            target_fingerprint == source_fingerprint
            and target_dimension == source_dimension
        ):
            return {
                "status": "no_change",
                "document_count": len(rows),
                "source_snapshot_sha256": source_digest,
                "embedding_dimension": target_dimension,
                "embedding_fingerprint": target_fingerprint,
                "chunking_fingerprint": target_chunking_fingerprint,
            }
        if not apply:
            return {
                "status": "planned",
                "document_count": len(rows),
                "source_snapshot_sha256": source_digest,
                "source_embedding_dimension": source_dimension,
                "target_embedding_dimension": target_dimension,
                "source_embedding_fingerprint": source_fingerprint,
                "target_embedding_fingerprint": target_fingerprint,
                "chunking_fingerprint": source_chunking_fingerprint,
                "backup_table": backup_table,
            }

        vectors: list[list[float]] = []
        for offset in range(0, len(rows), batch_size):
            contents = [row[2] for row in rows[offset : offset + batch_size]]
            vectors.extend(encode_texts(contents, batch_size=batch_size))
        if len(vectors) != len(rows) or any(
            len(vector) != target_dimension for vector in vectors
        ):
            raise RuntimeError("embedding model returned an unexpected vector count or dimension")

        with connection.cursor() as cursor:
            cursor.execute("LOCK TABLE documents IN ACCESS EXCLUSIVE MODE")
            locked_rows = _read_snapshot(cursor)
            if len(locked_rows) != expected_count or _snapshot_digest(locked_rows) != source_digest:
                raise RuntimeError("corpus changed while embeddings were being generated")
            locked_identity = _corpus_identity(cursor)
            if (
                locked_identity["embedding_dimension"] != source_dimension
                or locked_identity["embedding_fingerprint"] != source_fingerprint
                or locked_identity["chunking_fingerprint"] != source_chunking_fingerprint
            ):
                raise RuntimeError("database corpus identity changed before cutover")

            backup_identifier = sql.Identifier(backup_table)
            cursor.execute(
                sql.SQL(
                    "CREATE TABLE {} (tenant_id uuid NOT NULL, id text NOT NULL, "
                    "embedding vector({}){}, PRIMARY KEY (tenant_id, id))"
                ).format(
                    backup_identifier,
                    sql.Literal(source_dimension),
                    sql.SQL(" NOT NULL")
                    if locked_identity["embedding_not_null"]
                    else sql.SQL(""),
                )
            )
            cursor.execute(
                sql.SQL("INSERT INTO {} SELECT tenant_id, id, embedding FROM documents").format(
                    backup_identifier
                )
            )
            if cursor.rowcount != expected_count:
                raise RuntimeError("backup table did not capture every document embedding")

            cursor.execute(
                sql.SQL(
                    "CREATE TEMP TABLE rag_embedding_rebuild_stage "
                    "(tenant_id uuid NOT NULL, id text NOT NULL, embedding vector({}) NOT NULL, "
                    "PRIMARY KEY (tenant_id, id)) ON COMMIT DROP"
                ).format(sql.Literal(target_dimension))
            )
            execute_values(
                cursor,
                "INSERT INTO rag_embedding_rebuild_stage (tenant_id, id, embedding) VALUES %s",
                [
                    (tenant_id, document_id, _vector_literal(vector))
                    for (tenant_id, document_id, _content), vector in zip(rows, vectors, strict=True)
                ],
                template="(%s::uuid, %s, %s::vector)",
                page_size=batch_size,
            )
            cursor.execute(
                sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(_EMBEDDING_INDEX))
            )
            if source_dimension != target_dimension:
                if locked_identity["embedding_not_null"]:
                    cursor.execute(
                        "ALTER TABLE documents ALTER COLUMN embedding DROP NOT NULL"
                    )
                cursor.execute(
                    sql.SQL(
                        "ALTER TABLE documents ALTER COLUMN embedding TYPE vector({}) "
                        "USING NULL::vector({})"
                    ).format(
                        sql.Literal(target_dimension),
                        sql.Literal(target_dimension),
                    )
                )
            cursor.execute(
                """
                UPDATE documents AS document
                   SET embedding = stage.embedding
                  FROM rag_embedding_rebuild_stage AS stage
                 WHERE document.tenant_id = stage.tenant_id
                   AND document.id = stage.id
                """
            )
            if cursor.rowcount != expected_count:
                raise RuntimeError("vector cutover did not update every document")
            if source_dimension != target_dimension and locked_identity["embedding_not_null"]:
                cursor.execute("ALTER TABLE documents ALTER COLUMN embedding SET NOT NULL")
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX {} ON documents USING ivfflat "
                    "(embedding vector_cosine_ops) WITH (lists = 100)"
                ).format(sql.Identifier(_EMBEDDING_INDEX))
            )
            execute_values(
                cursor,
                """
                INSERT INTO rag_schema_metadata (key, value, updated_at) VALUES %s
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                """,
                [
                    (key, value, datetime.now(UTC).replace(tzinfo=None))
                    for key, value in (
                        ("embedding_dimension", str(target_dimension)),
                        ("embedding_fingerprint", target_fingerprint),
                    )
                ],
            )
        connection.commit()
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("ANALYZE documents")
        return {
            "status": "completed",
            "document_count": expected_count,
            "source_snapshot_sha256": source_digest,
            "source_embedding_dimension": source_dimension,
            "target_embedding_dimension": target_dimension,
            "source_embedding_fingerprint": source_fingerprint,
            "target_embedding_fingerprint": target_fingerprint,
            "chunking_fingerprint": source_chunking_fingerprint,
            "backup_table": backup_table,
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="read the configured corpus identity and print exact plan/apply commands",
    )
    parser.add_argument("--from-dimension", type=int)
    parser.add_argument("--from-fingerprint")
    parser.add_argument("--from-chunking-fingerprint")
    parser.add_argument("--confirm-document-count", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--backup-table",
        default=_default_backup_table(),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        if (
            args.from_dimension is not None
            or args.from_fingerprint is not None
            or args.from_chunking_fingerprint is not None
            or args.confirm_document_count is not None
            or args.apply
        ):
            parser.error("--inspect cannot be combined with migration or --apply arguments")
    elif (
        args.from_dimension is None
        or args.from_fingerprint is None
        or args.from_chunking_fingerprint is None
        or args.confirm_document_count is None
    ):
        parser.error(
            "migration requires --from-dimension, --from-fingerprint, "
            "--from-chunking-fingerprint, and --confirm-document-count"
        )
    return args


def main() -> None:
    _load_environment()
    args = parse_args()
    if args.inspect:
        result = inspect_corpus()
    else:
        assert args.from_dimension is not None
        assert args.from_fingerprint is not None
        assert args.from_chunking_fingerprint is not None
        assert args.confirm_document_count is not None
        result = rebuild(
            source_dimension=args.from_dimension,
            source_fingerprint=args.from_fingerprint,
            source_chunking_fingerprint=args.from_chunking_fingerprint,
            expected_count=args.confirm_document_count,
            batch_size=args.batch_size,
            backup_table=args.backup_table,
            apply=args.apply,
        )
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
