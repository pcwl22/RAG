"""
重新分块脚本
使用新的Parent-Child策略重新处理现有文档
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.parser.chunk import ParentChildChunker
from app.embedding.embedder import encode_texts
from app.vectorstore.postgres_store import init_postgres_store, close_postgres_store, add_documents

logger = get_logger(__name__)


async def rechunk_all_documents():
    """重新分块所有文档"""
    logger.info("开始重新分块任务...")

    # 初始化存储
    await init_postgres_store()

    try:
        # 1. 获取所有现有文档
        from app.vectorstore.postgres_store import _connection
        import psycopg2.extras

        conn = _connection()
        cur = None
        documents = []

        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # 获取所有唯一文档
            cur.execute("""
                SELECT
                    metadata->>'document_id' AS document_id,
                    metadata->>'filename' AS filename,
                    partition,
                    STRING_AGG(content, E'\n\n' ORDER BY (metadata->>'chunk_index')::int) AS full_text
                FROM documents
                WHERE metadata->>'document_id' IS NOT NULL
                GROUP BY
                    metadata->>'document_id',
                    metadata->>'filename',
                    partition
            """)

            documents = [dict(row) for row in cur.fetchall()]
            logger.info(f"找到 {len(documents)} 个文档需要重新分块")

        finally:
            if cur:
                cur.close()
            from app.vectorstore.postgres_store import _pool
            if _pool:
                _pool.putconn(conn)

        if not documents:
            logger.info("没有文档需要处理")
            return

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
                # 3.1 删除旧的分块
                conn = _connection()
                cur = None
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "DELETE FROM documents WHERE metadata->>'document_id' = %s",
                        (document_id,)
                    )
                    conn.commit()
                    logger.info(f"已删除旧分块: {document_id}")
                finally:
                    if cur:
                        cur.close()
                    from app.vectorstore.postgres_store import _pool
                    if _pool:
                        _pool.putconn(conn)

                # 3.2 使用新策略分块
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

                # 3.4 生成 Embedding
                logger.info(f"生成 Embedding: {len(chunk_texts)} 个子块")
                embeddings = []
                batch_size = 16

                for start in range(0, len(chunk_texts), batch_size):
                    batch = chunk_texts[start : start + batch_size]
                    embeddings.extend(encode_texts(batch, batch_size=len(batch)))

                # 3.5 存储到数据库
                await add_documents(
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
        await close_postgres_store()


if __name__ == "__main__":
    asyncio.run(rechunk_all_documents())
