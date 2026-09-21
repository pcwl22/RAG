"""
重新分块脚本
使用新的Parent-Child策略重新处理现有文档
"""
import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.auth import (  # noqa: E402
    Principal,
    normalize_tenant_id,
    reset_current_principal,
    set_current_principal,
)
from app.embedding.embedder import encode_texts  # noqa: E402
from app.parser.chunk import ParentChildChunker  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402
from app.vectorstore.postgres_store import (  # noqa: E402
    close_postgres_store,
    init_postgres_store,
    replace_document,
)

logger = get_logger(__name__)


def _source_key(filename: str, partition: str, tenant_id: str) -> str:
    """Match the ingest service's stable source identity for legacy rows."""
    material = f"{tenant_id}\0{partition.strip().casefold()}\0{filename.strip().casefold()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


async def rechunk_all_documents(tenant_id: str) -> dict[str, int]:
    """Rechunk documents for one tenant without destroying the old version first."""
    tenant_id = normalize_tenant_id(tenant_id)
    logger.info("开始重新分块任务...")

    # 初始化存储
    await init_postgres_store()

    context_token = set_current_principal(
        Principal(subject="rechunk-script", tenant_id=tenant_id, roles=frozenset({"admin"}), auth_type="local")
    )
    try:
        # 1. 获取所有现有文档
        import psycopg2.extras

        from app.vectorstore.postgres_store import _connection

        conn = _connection()
        cur = None
        documents = []

        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # 获取所有唯一文档
            from app.vectorstore.postgres_store import _set_tenant

            _set_tenant(cur, tenant_id)
            cur.execute("""
                SELECT
                    metadata->>'document_id' AS document_id,
                    metadata->>'filename' AS filename,
                    partition,
                    COALESCE(MAX(NULLIF(metadata->>'source_key', '')), '') AS source_key,
                    STRING_AGG(
                        content,
                        E'\n\n'
                        ORDER BY CASE
                            WHEN metadata->>'chunk_index' ~ '^[0-9]+$'
                            THEN (metadata->>'chunk_index')::int
                            ELSE 0
                        END
                    ) AS full_text
                FROM documents
                WHERE tenant_id = %s::uuid
                  AND metadata->>'document_id' IS NOT NULL
                GROUP BY
                    metadata->>'document_id',
                    metadata->>'filename',
                    partition
            """, (tenant_id,))

            documents = [dict(row) for row in cur.fetchall()]
            logger.info(f"找到 {len(documents)} 个文档需要重新分块")

        finally:
            if cur:
                cur.close()
            from app.vectorstore.postgres_store import _return_connection
            _return_connection(conn)

        if not documents:
            logger.info("没有文档需要处理")
            return {"processed": 0, "failed": 0, "total": 0}

        # 2. 初始化分块器
        chunker = ParentChildChunker(
            parent_min_tokens=1500,
            parent_max_tokens=3000,
            child_min_tokens=300,
            child_max_tokens=800,
        )

        # 3. 逐个处理文档
        total_processed = 0
        total_failed = 0

        for doc in documents:
            document_id = doc['document_id']
            filename = doc['filename']
            partition = doc['partition'] or 'general'
            full_text = doc['full_text']

            if not full_text or len(full_text.strip()) < 50:
                logger.warning(f"跳过文档 {document_id}：内容太短")
                total_failed += 1
                continue

            logger.info(f"处理文档: {filename} ({document_id})")

            try:
                # 3.1 使用新策略分块；旧版本保持不变直到新版本完整写入。
                parent_chunks, child_chunks = chunker.chunk_document(
                    text=full_text,
                    document_id=document_id,
                    filename=filename,
                    partition=partition,
                )

                logger.info(
                    f"生成分块: {len(parent_chunks)} 个父块, {len(child_chunks)} 个子块"
                )

                # 3.3 准备子块数据（用于检索）
                chunk_texts = [chunk.content for chunk in child_chunks]
                chunk_ids = [chunk.id for chunk in child_chunks]
                chunk_metadatas = [chunk.metadata for chunk in child_chunks]
                source_key = doc.get("source_key") or _source_key(filename, partition, tenant_id)
                for metadata in chunk_metadatas:
                    metadata["tenant_id"] = tenant_id
                    metadata["source_key"] = source_key

                # 3.4 生成 Embedding
                logger.info(f"生成 Embedding: {len(chunk_texts)} 个子块")
                embeddings = []
                batch_size = 16

                for start in range(0, len(chunk_texts), batch_size):
                    batch = chunk_texts[start : start + batch_size]
                    embeddings.extend(encode_texts(batch, batch_size=len(batch)))

                # 3.5 在单事务中替换该租户的逻辑来源。
                await replace_document(
                    source_key=source_key,
                    filename=filename,
                    ids=chunk_ids,
                    embeddings=embeddings,
                    documents=chunk_texts,
                    metadatas=chunk_metadatas,
                    partition=partition,
                )

                logger.info(f"✓ 文档处理完成: {filename}")
                total_processed += 1

            except Exception as e:
                logger.error(f"✗ 文档处理失败 {filename}: {e}", exc_info=True)
                total_failed += 1

        logger.info(f"""
重新分块任务完成！
- 成功: {total_processed}
- 失败: {total_failed}
- 总计: {len(documents)}
        """)

    finally:
        reset_current_principal(context_token)
        await close_postgres_store()
    return {"processed": total_processed, "failed": total_failed, "total": len(documents)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True, help="Tenant UUID to rebuild")
    args = parser.parse_args()
    asyncio.run(rechunk_all_documents(args.tenant_id))
