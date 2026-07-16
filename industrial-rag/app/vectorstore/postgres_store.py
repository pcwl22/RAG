"""PostgreSQL + pgvector storage with hybrid retrieval."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:
    psycopg2 = None

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_pool: Any | None = None
_executor: ThreadPoolExecutor | None = None
SCHEMA_VERSION = "1"


def _postgres_config() -> dict:
    return get_settings().get("postgres", {})


def _embedding_dimension() -> int:
    return int(get_settings().get("embedding", {}).get("dimension", 1024))


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
        await _execute_sync(_create_schema_sync)
    except Exception:
        _dispose_postgres_runtime()
        raise
    logger.info("PostgreSQL store initialized")


def _execute_sync(func, *args, **kwargs):
    if _executor is None:
        raise RuntimeError("PostgreSQL store is not initialized")
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_executor, func, *args, **kwargs)


def _connection():
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


def _create_schema_sync() -> None:
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor()
        dimension = _embedding_dimension()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector({dimension}),
                metadata JSONB DEFAULT '{{}}',
                partition TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_content_trgm
            ON documents USING gin (content gin_trgm_ops);
            """
        )
        # Match the exact expression used by keyword retrieval, including metadata.
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_search_trgm
            ON documents USING gin ((content || ' ' || COALESCE(metadata::text, '')) gin_trgm_ops);
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_partition ON documents (partition);")
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_document_id
            ON documents ((metadata->>'document_id'));
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_source_key
            ON documents ((metadata->>'source_key'));
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin (metadata);")
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat
            ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """
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
        cur.executemany(
            """
            INSERT INTO rag_schema_metadata (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value, updated_at = NOW()
            """,
            [
                ("schema_version", SCHEMA_VERSION),
                ("embedding_dimension", str(dimension)),
            ],
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


def _check_postgres_health_sync() -> bool:
    """Verify that the pool can execute a query, not merely that it exists."""
    if _pool is None:
        return False
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                to_regclass('public.documents') IS NOT NULL,
                to_regclass('public.rag_schema_metadata') IS NOT NULL,
                EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')
            """
        )
        documents_ready, metadata_ready, vector_ready = cur.fetchone()
        if not (documents_ready and metadata_ready and vector_ready):
            return False
        cur.execute(
            """
            SELECT
                format_type(attribute.atttypid, attribute.atttypmod),
                (SELECT value FROM rag_schema_metadata WHERE key = 'schema_version'),
                (SELECT value FROM rag_schema_metadata WHERE key = 'embedding_dimension')
            FROM pg_attribute AS attribute
            WHERE attribute.attrelid = 'public.documents'::regclass
              AND attribute.attname = 'embedding'
              AND NOT attribute.attisdropped
            """
        )
        row = cur.fetchone()
        expected_dimension = _embedding_dimension()
        return bool(
            row
            and row[0] == f"vector({expected_dimension})"
            and row[1] == SCHEMA_VERSION
            and row[2] == str(expected_dimension)
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
        metadatas = metadatas or [{} for _ in ids]
        records = []
        for index, doc_id in enumerate(ids):
            embedding = f"[{','.join(str(x) for x in embeddings[index])}]"
            records.append(
                (
                    doc_id,
                    documents[index],
                    embedding,
                    json.dumps(metadatas[index], ensure_ascii=False),
                    partition,
                )
            )

        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO documents (id, content, embedding, metadata, partition)
            VALUES (%s, %s, %s::vector, %s::jsonb, %s)
            ON CONFLICT (id) DO UPDATE SET
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
        cur.execute(
            """
            DELETE FROM documents
            WHERE metadata->>'source_key' = %s
               OR (
                    metadata->>'source_key' IS NULL
                    AND metadata->>'filename' = %s
                    AND partition = %s
               )
            """,
            (source_key, filename, partition),
        )
        records = [
            (
                doc_id,
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
            INSERT INTO documents (id, content, embedding, metadata, partition, created_at)
            VALUES (%s, %s, %s::vector, %s::jsonb, %s, NOW())
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
        embedding = f"[{','.join(str(x) for x in query_embedding)}]"
        sql = """
            SELECT id, content, 1 - (embedding <=> %s::vector) AS score, metadata, partition
            FROM documents
            WHERE embedding IS NOT NULL
        """
        params: list[Any] = [embedding]
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
        sql = """
            SELECT id, content, metadata, partition
            FROM documents
            WHERE (id = ANY(%s) OR metadata->>'semantic_chunk_id' = ANY(%s))
        """
        params: list[Any] = [ids, ids]
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
        pair_conditions = " OR ".join(
            "(metadata->>'law_name' = %s AND metadata->>'article_number' = %s)"
            for _ in citation_pairs
        )
        sql = f"""
            SELECT id, content, metadata, partition
            FROM documents
            WHERE ({pair_conditions})
        """
        params: list[Any] = [value for pair in citation_pairs for value in pair]
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
        keywords = _extract_chinese_keywords(query)
        if not keywords:
            return []

        search_expr = "(content || ' ' || COALESCE(metadata::text, ''))"
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
            WHERE ({where_parts})
        """

        params: list[Any] = [f"%{kw}%" for kw in keywords] + [f"%{kw}%" for kw in keywords]
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
        cur.execute("DELETE FROM documents WHERE metadata->>'document_id' = %s", (document_id,))
        deleted = cur.rowcount > 0
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
        cur.execute(
            """
            SELECT
                metadata->>'document_id' AS document_id,
                metadata->>'filename' AS filename,
                partition,
                COUNT(*) AS chunk_count,
                MAX(created_at) AS updated_at
            FROM documents
            WHERE metadata->>'document_id' IS NOT NULL
            GROUP BY metadata->>'document_id', metadata->>'filename', partition
            ORDER BY MAX(created_at) DESC
            OFFSET %s LIMIT %s
            """,
            (skip, limit),
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
        cur.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT metadata->>'document_id', metadata->>'filename', partition
                FROM documents
                WHERE metadata->>'document_id' IS NOT NULL
                GROUP BY metadata->>'document_id', metadata->>'filename', partition
            ) AS grouped_documents
            """
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
        cur.execute(
            """
            SELECT COUNT(*) AS total
            FROM documents
            WHERE metadata->>'document_id' = %s
            """,
            (document_id,),
        )
        total = int(cur.fetchone()["total"])

        cur.execute(
            """
            SELECT id, content, metadata, partition, created_at
            FROM documents
            WHERE metadata->>'document_id' = %s
            ORDER BY
                CASE
                    WHEN metadata->>'chunk_index' ~ '^[0-9]+$'
                    THEN (metadata->>'chunk_index')::int
                    ELSE 0
                END,
                id
            OFFSET %s LIMIT %s
            """,
            (document_id, skip, limit),
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
