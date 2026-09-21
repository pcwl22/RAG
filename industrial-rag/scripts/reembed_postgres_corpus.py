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


def _current_metadata(cursor: Any) -> dict[str, str]:
    cursor.execute("SELECT key, value FROM rag_schema_metadata")
    return {str(key): str(value) for key, value in cursor.fetchall()}


def _require_migration_identity(cursor: Any) -> None:
    cursor.execute(
        "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
    )
    row = cursor.fetchone()
    if not row or not (bool(row[0]) or bool(row[1])):
        raise RuntimeError("re-embedding requires a migration identity with BYPASSRLS")


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def rebuild(
    *,
    source_fingerprint: str,
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

    source_fingerprint = _validate_fingerprint(source_fingerprint, "source fingerprint")
    backup_table = _validate_backup_table(backup_table)
    if expected_count < 1:
        raise ValueError("expected document count must be positive")
    if batch_size < 1 or batch_size > 256:
        raise ValueError("batch size must be between 1 and 256")

    target_metadata = _runtime_schema_metadata()
    target_fingerprint = _validate_fingerprint(
        target_metadata["embedding_fingerprint"], "target fingerprint"
    )
    dimension = int(target_metadata["embedding_dimension"])

    connection = psycopg2.connect(**_connection_options())
    try:
        with connection.cursor() as cursor:
            _require_migration_identity(cursor)
            metadata = _current_metadata(cursor)
            actual_fingerprint = metadata.get("embedding_fingerprint", "")
            if actual_fingerprint != source_fingerprint:
                raise RuntimeError(
                    "database embedding fingerprint changed; refusing the requested migration"
                )
            rows = _read_snapshot(cursor)
        connection.rollback()
        if len(rows) != expected_count:
            raise RuntimeError(
                f"database document count changed: expected {expected_count}, found {len(rows)}"
            )
        source_digest = _snapshot_digest(rows)
        if target_fingerprint == source_fingerprint:
            return {
                "status": "no_change",
                "document_count": len(rows),
                "source_snapshot_sha256": source_digest,
                "embedding_fingerprint": target_fingerprint,
            }
        if not apply:
            return {
                "status": "planned",
                "document_count": len(rows),
                "source_snapshot_sha256": source_digest,
                "source_embedding_fingerprint": source_fingerprint,
                "target_embedding_fingerprint": target_fingerprint,
                "backup_table": backup_table,
            }

        vectors: list[list[float]] = []
        for offset in range(0, len(rows), batch_size):
            contents = [row[2] for row in rows[offset : offset + batch_size]]
            vectors.extend(encode_texts(contents, batch_size=batch_size))
        if len(vectors) != len(rows) or any(len(vector) != dimension for vector in vectors):
            raise RuntimeError("embedding model returned an unexpected vector count or dimension")

        with connection.cursor() as cursor:
            cursor.execute("LOCK TABLE documents IN ACCESS EXCLUSIVE MODE")
            locked_rows = _read_snapshot(cursor)
            if len(locked_rows) != expected_count or _snapshot_digest(locked_rows) != source_digest:
                raise RuntimeError("corpus changed while embeddings were being generated")
            locked_metadata = _current_metadata(cursor)
            if locked_metadata.get("embedding_fingerprint") != source_fingerprint:
                raise RuntimeError("database embedding fingerprint changed before cutover")

            backup_identifier = sql.Identifier(backup_table)
            cursor.execute(
                sql.SQL(
                    "CREATE TABLE {} (tenant_id uuid NOT NULL, id text NOT NULL, "
                    "embedding vector({}) NOT NULL, PRIMARY KEY (tenant_id, id))"
                ).format(backup_identifier, sql.Literal(dimension))
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
                ).format(sql.Literal(dimension))
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
            execute_values(
                cursor,
                """
                INSERT INTO rag_schema_metadata (key, value, updated_at) VALUES %s
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                """,
                [
                    (key, value, datetime.now(UTC).replace(tzinfo=None))
                    for key, value in target_metadata.items()
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
            "source_embedding_fingerprint": source_fingerprint,
            "target_embedding_fingerprint": target_fingerprint,
            "backup_table": backup_table,
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-fingerprint", required=True)
    parser.add_argument("--confirm-document-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--backup-table",
        default="rag_embedding_backup_" + datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    _load_environment()
    args = parse_args()
    result = rebuild(
        source_fingerprint=args.from_fingerprint,
        expected_count=args.confirm_document_count,
        batch_size=args.batch_size,
        backup_table=args.backup_table,
        apply=args.apply,
    )
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
