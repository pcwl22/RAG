"""
重新上传并分块脚本
直接使用上传目录中的文档重新处理
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingest.pipeline import parse_document
from src.core.chunking import ParentChildChunker
from src.models.embedding import encode_texts
from src.storage.postgres_store import init_postgres_store, close_postgres_store, add_documents
from src.utils.logger import get_logger
import uuid

logger = get_logger(__name__)


async def reprocess_documents():
    """重新处理上传目录中的所有文档"""
    logger.info("开始重新处理文档...")

    # 初始化存储
    await init_postgres_store()

    try:
        # 查找所有PDF文件
        upload_dir = Path("E:/RAG/industrial-rag/data/uploads")
        pdf_files = list(upload_dir.glob("*.pdf"))

        logger.info(f"找到 {len(pdf_files)} 个PDF文件")

        if not pdf_files:
            logger.info("没有文件需要处理")
            return

        # 初始化分块器
        chunker = ParentChildChunker(
            parent_min_tokens=1500,
            parent_max_tokens=3000,
            child_min_tokens=300,
            child_max_tokens=800,
        )

        total_processed = 0
        total_failed = 0

        for pdf_file in pdf_files:
            filename = pdf_file.name
            # 提取原始文件名（去掉UUID前缀）
            if "_" in filename:
                original_filename = filename.split("_", 1)[1]
            else:
                original_filename = filename

            logger.info(f"处理文件: {original_filename}")

            try:
                # 1. 解析PDF
                text = parse_document(str(pdf_file))

                if not text or len(text.strip()) < 50:
                    logger.warning(f"跳过 {original_filename}：内容太短")
                    total_failed += 1
                    continue

                # 2. 生成新的document_id
                document_id = str(uuid.uuid4())

                # 3. 使用新策略分块
                parent_chunks, child_chunks = chunker.chunk_document(
                    text=text,
                    document_id=document_id,
                    filename=original_filename,
                    partition="general",
                )

                logger.info(
                    f"生成分块: {len(parent_chunks)} 个父块, {len(child_chunks)} 个子块"
                )

                # 4. 准备子块数据（用于检索）
                chunk_texts = [chunk.content for chunk in child_chunks]
                chunk_ids = [chunk.id for chunk in child_chunks]
                chunk_metadatas = [chunk.metadata for chunk in child_chunks]

                # 5. 生成 Embedding
                logger.info(f"生成 Embedding: {len(chunk_texts)} 个子块")
                embeddings = []
                batch_size = 16

                for start in range(0, len(chunk_texts), batch_size):
                    batch = chunk_texts[start : start + batch_size]
                    embeddings.extend(encode_texts(batch, batch_size=len(batch)))

                # 6. 存储到数据库
                await add_documents(
                    ids=chunk_ids,
                    embeddings=embeddings,
                    documents=chunk_texts,
                    metadatas=chunk_metadatas,
                    partition="general",
                )

                logger.info(f"[OK] 文档处理完成: {original_filename}")
                total_processed += 1

            except Exception as e:
                logger.error(f"[FAIL] 文档处理失败 {original_filename}: {e}", exc_info=True)
                total_failed += 1

        logger.info(f"""
重新处理任务完成！
- 成功: {total_processed}
- 失败: {total_failed}
- 总计: {len(pdf_files)}
        """)

    finally:
        await close_postgres_store()


if __name__ == "__main__":
    asyncio.run(reprocess_documents())
