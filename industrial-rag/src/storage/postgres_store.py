"""PostgreSQL + pgvector storage with hybrid retrieval.

The retrieval path runs vector search and keyword search in parallel, then uses
Reciprocal Rank Fusion (RRF) and Dynamic Top-K pruning.
"""
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

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_pool: Any | None = None
_executor: ThreadPoolExecutor | None = None


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
    )
    await _execute_sync(_create_schema_sync)
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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_partition ON documents (partition);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin (metadata);")
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat
            ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """
        )
        conn.commit()
    finally:
        if cur is not None:
            cur.close()
        if _pool is not None:
            _pool.putconn(conn)


async def close_postgres_store() -> None:
    global _pool, _executor
    if _pool is not None:
        _pool.closeall()
        _pool = None
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


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
                partition = EXCLUDED.partition;
            """,
            records,
        )
        conn.commit()
        return len(ids)
    finally:
        if cur is not None:
            cur.close()
        if _pool is not None:
            _pool.putconn(conn)


async def add_documents(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict] | None = None,
    partition: str = "general",
) -> int:
    return await _execute_sync(_add_documents_sync, ids, embeddings, documents, metadatas, partition)


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
        if _pool is not None:
            _pool.putconn(conn)


async def vector_search(
    query_embedding: list[float],
    top_k: int = 10,
    partition: str | None = None,
) -> list[dict]:
    return await _execute_sync(_vector_search_sync, query_embedding, top_k, partition)


def _extract_chinese_keywords(query: str) -> list[str]:
    """从中文查询中提取关键词用于ILIKE匹配"""
    import re
    # 去除标点
    cleaned = re.sub(r'[，。？！、\s?!.,;；：:""''（）()\n]+', ' ', query).strip()
    # 去除常见疑问/虚词短语（长的先替换）
    stop_phrases = ['什么时候', '什么', '怎么样', '怎么', '如何', '是否', '能否', '为什么',
                    '怎样', '哪些', '哪个', '多少', '可以', '能够', '应该', '需要']
    for sp in sorted(stop_phrases, key=len, reverse=True):
        cleaned = cleaned.replace(sp, ' ')
    # 去除单字虚词
    cleaned = re.sub(r'(?<![a-zA-Z])[的了吗呢吧是在有个后前经被把给从到](?![a-zA-Z])', ' ', cleaned)
    # 按空格分割
    parts = [p.strip() for p in cleaned.split() if len(p.strip()) >= 2]

    # 对于长片段(>4字符)，提取2-4字符的子串
    keywords = []
    for part in parts:
        if len(part) <= 4:
            keywords.append(part)
        else:
            # 提取4字符、3字符、2字符的非重叠片段
            i = 0
            while i < len(part):
                if i + 4 <= len(part):
                    keywords.append(part[i:i+4])
                    i += 4
                elif i + 3 <= len(part):
                    keywords.append(part[i:i+3])
                    i += 3
                elif i + 2 <= len(part):
                    keywords.append(part[i:i+2])
                    i += 2
                else:
                    break

    # 去重，保持顺序
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:6]


def _keyword_search_sync(query: str, top_k: int, partition: str | None) -> list[dict]:
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        keywords = _extract_chinese_keywords(query)

        if not keywords:
            return []

        # 构建SQL：每个关键词匹配加0.2分
        score_parts = ' + '.join([
            "CASE WHEN content ILIKE %s THEN 0.2 ELSE 0 END" for _ in keywords
        ])

        where_parts = ' OR '.join([
            "content ILIKE %s" for _ in keywords
        ])

        sql = f"""
            SELECT
                id,
                content,
                ({score_parts}) AS score,
                metadata,
                partition
            FROM documents
            WHERE {where_parts}
        """

        # 参数：score部分的ILIKE + where部分的ILIKE
        params: list[Any] = [f"%{kw}%" for kw in keywords] + [f"%{kw}%" for kw in keywords]

        if partition:
            sql += " AND partition = %s"
            params.append(partition)

        sql += " ORDER BY score DESC LIMIT %s"
        params.append(top_k)

        cur.execute(sql, params)
        return [_row_to_doc(row) for row in cur.fetchall()]
    finally:
        if cur is not None:
            cur.close()
        if _pool is not None:
            _pool.putconn(conn)


async def bm25_search(query: str, top_k: int = 10, partition: str | None = None) -> list[dict]:
    """Keyword retrieval implemented with pg_trgm similarity and ILIKE."""
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


def _delete_document_sync(document_id: str) -> None:
    conn = _connection()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE metadata->>'document_id' = %s", (document_id,))
        conn.commit()
    finally:
        if cur is not None:
            cur.close()
        if _pool is not None:
            _pool.putconn(conn)


async def delete_document(document_id: str) -> None:
    await _execute_sync(_delete_document_sync, document_id)


def _list_documents_sync(skip: int, limit: int) -> list[dict]:
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
        if _pool is not None:
            _pool.putconn(conn)


async def list_documents(skip: int = 0, limit: int = 100) -> list[dict]:
    return await _execute_sync(_list_documents_sync, skip, limit)
