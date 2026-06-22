"""
简化的查询路由 - 绕过 Pydantic 验证问题
修复中文编码问题
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import time
import json

from src.core.retrieval.engine import RetrievalEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)
router_simple = APIRouter()


@router_simple.post("/query_simple")
async def query_simple(request: Request):
    """
    简化的查询端点 - 直接解析 JSON
    修复中文编码问题
    """
    start_time = time.time()

    try:
        # 直接从 request 解析 JSON（确保 UTF-8）
        body_bytes = await request.body()
        body_str = body_bytes.decode('utf-8')
        body = json.loads(body_str)

        query = body.get("query")
        top_k = body.get("top_k", 5)
        similarity_threshold = body.get("similarity_threshold", 0.5)
        enable_rerank = body.get("enable_rerank", True)

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "query field is required"}
            )

        logger.info(f"Simple query: {query}, top_k: {top_k}")

        # 初始化检索引擎
        engine = RetrievalEngine()

        # 执行检索
        results = await engine.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            enable_rerank=enable_rerank,
        )

        # 构造响应
        documents = [
            {
                "id": doc["id"],
                "content": doc["content"],
                "score": float(doc["score"]),
                "metadata": doc.get("metadata", {}),
                "chunk_index": doc.get("chunk_index", 0),
            }
            for doc in results
        ]

        retrieval_time = time.time() - start_time

        return JSONResponse(
            content={
                "query": query,
                "documents": documents,
                "total": len(documents),
                "retrieval_time": retrieval_time,
            },
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        logger.error(f"Simple query failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            media_type="application/json; charset=utf-8"
        )


@router_simple.post("/answer_simple")
async def answer_simple(request: Request):
    """
    简化的问答端点 - 直接解析 JSON
    修复中文编码问题
    """
    start_time = time.time()

    try:
        # 直接从 request 解析 JSON（确保 UTF-8）
        body_bytes = await request.body()
        body_str = body_bytes.decode('utf-8')
        body = json.loads(body_str)

        query = body.get("query")
        top_k = body.get("top_k", 5)
        enable_rerank = body.get("enable_rerank", True)

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "query field is required"}
            )

        logger.info(f"Simple answer: {query}, top_k: {top_k}")

        # 初始化检索引擎
        engine = RetrievalEngine()

        # 检索相关文档
        results = await engine.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=0.5,
            enable_rerank=enable_rerank,
        )

        if not results:
            return JSONResponse(
                content={
                    "query": query,
                    "answer": "抱歉，我在知识库中没有找到相关信息来回答这个问题。",
                    "sources": [],
                    "total_time": time.time() - start_time,
                },
                media_type="application/json; charset=utf-8"
            )

        # 生成答案
        from src.core.generation.generator import Generator
        generator = Generator()

        answer = await generator.generate(query=query, context_docs=results)

        # 转换来源文档
        sources = [
            {
                "id": doc["id"],
                "content": doc["content"],
                "score": float(doc["score"]),
                "metadata": doc.get("metadata", {}),
                "chunk_index": doc.get("chunk_index", 0),
            }
            for doc in results
        ]

        total_time = time.time() - start_time

        logger.info(f"Answer generated in {total_time:.3f}s")

        return JSONResponse(
            content={
                "query": query,
                "answer": answer,
                "sources": sources,
                "total_time": total_time,
            },
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        logger.error(f"Simple answer failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            media_type="application/json; charset=utf-8"
        )
