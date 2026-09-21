"""
兼容旧客户端的简化查询路由。
"""
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.retrieval_params import resolve_retrieval_params
from app.retrieval.domain_signal_map import is_known_out_of_scope
from app.retrieval.factory import build_retrieval_engine
from app.utils.logger import get_logger

logger = get_logger(__name__)
router_simple = APIRouter()


class SimpleQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    enable_rerank: bool | None = None


@router_simple.post("/query_simple")
async def query_simple(request: SimpleQueryRequest) -> JSONResponse:
    """
    保留旧路径的简化查询端点，并复用统一的请求边界校验。
    """
    start_time = time.time()

    try:
        query = request.query

        params = resolve_retrieval_params(
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            enable_rerank=request.enable_rerank,
        )

        logger.info("Simple query received, top_k=%s", params.top_k)

        # Reject known out-of-scope requests before loading models or touching
        # the vector store.  This keeps the compatibility endpoint aligned
        # with the main retrieval engines.
        if is_known_out_of_scope(query):
            results = []
        else:
            engine = build_retrieval_engine()
            results = await engine.retrieve(
                query=query,
                top_k=params.top_k,
                similarity_threshold=params.similarity_threshold,
                enable_rerank=params.enable_rerank,
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
            content={"error": "Query failed"},
            media_type="application/json; charset=utf-8"
        )


@router_simple.post("/answer_simple")
async def answer_simple(request: SimpleQueryRequest) -> JSONResponse:
    """
    保留旧路径的简化问答端点，并复用统一的请求边界校验。
    """
    start_time = time.time()

    try:
        query = request.query

        params = resolve_retrieval_params(
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            enable_rerank=request.enable_rerank,
        )

        logger.info("Simple answer received, top_k=%s", params.top_k)

        if is_known_out_of_scope(query):
            results = []
        else:
            engine = build_retrieval_engine()
            results = await engine.retrieve(
                query=query,
                top_k=params.top_k,
                similarity_threshold=params.similarity_threshold,
                enable_rerank=params.enable_rerank,
            )

        if not results:
            from app.service.chat_service import format_no_context_answer

            return JSONResponse(
                content={
                    "query": query,
                    "answer": format_no_context_answer(),
                    "sources": [],
                    "total_time": time.time() - start_time,
                },
                media_type="application/json; charset=utf-8"
            )

        # 生成答案
        from app.service.chat_service import Generator
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
            content={"error": "Answer generation failed"},
            media_type="application/json; charset=utf-8"
        )
