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
    keywords.extend(re.findall(r"第[一二三四五六七八九十百千万零〇两0-9]+条", query))
    keywords.extend(re.findall(r"[\u4e00-\u9fff]{2,12}(?:罪|合同|劳动合同|解除劳动合同|定义|刑罚)", query))

    offense_hints = {
        "盗窃罪": ["第二百六十四条", "盗窃公私财物", "入户盗窃", "扒窃"],
        "职务侵占罪": ["第二百七十一条", "职务上的便利", "本单位财物", "非法占为己有"],
        "胁从犯": ["第二十八条", "被胁迫参加犯罪", "减轻处罚", "免除处罚"],
    }
    for offense, hints in offense_hints.items():
        if offense in query:
            keywords.extend([offense, *hints])

    civil_hints = {
        ("工作人员", "执行工作任务"): ["第一千一百九十一条", "用人单位的工作人员", "执行工作任务", "用人单位承担侵权责任"],
        ("员工", "执行工作任务"): ["第一千一百九十一条", "用人单位的工作人员", "执行工作任务", "用人单位承担侵权责任"],
        ("劳动者", "执行工作任务"): ["第一千一百九十一条", "用人单位的工作人员", "执行工作任务", "用人单位承担侵权责任"],
    }
    for triggers, hints in civil_hints.items():
        if all(trigger in query for trigger in triggers):
            keywords.extend(hints)

    parts = [p.strip() for p in cleaned.split() if len(p.strip()) >= 2]
    for part in parts:
        if len(part) <= 8:
            keywords.append(part)
        else:
            for size in (6, 4, 3, 2):
                for i in range(0, len(part) - size + 1, size):
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

        weights = [_keyword_match_weight(keyword) for keyword in keywords]
        score_parts = " + ".join(
            [f"CASE WHEN content ILIKE %s THEN {weight} ELSE 0 END" for weight in weights]
        )
        where_parts = " OR ".join(["content ILIKE %s" for _ in keywords])

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
        if _pool is not None:
            _pool.putconn(conn)


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
