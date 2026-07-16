"""Real pgvector integration coverage, enabled explicitly in CI."""
import asyncio
import json
import os

import pytest

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
    }


def test_real_postgres_schema_health_and_citation_pairing(monkeypatch):
    monkeypatch.setattr(postgres_store, "_postgres_config", _integration_config)
    monkeypatch.setattr(postgres_store, "_embedding_dimension", lambda: 1024)

    async def run():
        await postgres_store.close_postgres_store()
        await postgres_store.init_postgres_store()
        assert await postgres_store.check_postgres_health() is True

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
