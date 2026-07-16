"""Query API routes."""
import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.retrieval_params import resolve_retrieval_params
from app.retrieval.domain_signal_map import is_known_out_of_scope
from app.retrieval.result_merge import result_score
from app.service.chat_service import Generator
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    enable_rerank: bool | None = None
    enable_multimodal: bool = False
    partition: str | None = None


class RetrievedDocument(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict[str, Any]
    chunk_index: int = 0


class QueryResponse(BaseModel):
    query: str
    documents: list[RetrievedDocument]
    total: int
    retrieval_time: float


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    enable_rerank: bool | None = None
    enable_multimodal: bool = False
    stream: bool = False
    partition: str | None = None


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[RetrievedDocument]
    total_time: float


def _retrieval_engine():
    retrieval_cfg = get_settings().get("rag", {}).get("retrieval", {})
    if retrieval_cfg.get("enable_hybrid", True):
        from app.retrieval.hybrid import HybridRetrievalEngine

        return HybridRetrievalEngine()

    from app.retrieval.dense import RetrievalEngine

    return RetrievalEngine()


def _to_retrieved_document(doc: dict) -> RetrievedDocument:
    metadata = doc.get("metadata") or {}
    return RetrievedDocument(
        id=str(doc.get("id")),
        content=doc.get("content", ""),
        score=result_score(doc),
        metadata=metadata,
        chunk_index=int(metadata.get("chunk_index", doc.get("chunk_index", 0)) or 0),
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    start = time.time()
    try:
        params = resolve_retrieval_params(
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            enable_rerank=request.enable_rerank,
        )
        if is_known_out_of_scope(request.query):
            results = []
        else:
            engine = _retrieval_engine()
            results = await engine.retrieve(
                query=request.query,
                top_k=params.top_k,
                similarity_threshold=params.similarity_threshold,
                enable_rerank=params.enable_rerank,
                partition=request.partition,
            )
        documents = [_to_retrieved_document(doc) for doc in results]
        return QueryResponse(
            query=request.query,
            documents=documents,
            total=len(documents),
            retrieval_time=time.time() - start,
        )
    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Query failed") from exc


@router.post("/answer", response_model=AnswerResponse)
async def answer_question(request: AnswerRequest) -> AnswerResponse | StreamingResponse:
    start = time.time()
    try:
        params = resolve_retrieval_params(
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            enable_rerank=request.enable_rerank,
        )
        if request.stream:

            async def stream_generator():
                try:
                    yield f"data: {json.dumps({'type': 'status', 'data': '正在检索相关文档...'}, ensure_ascii=False)}\n\n"
                    if is_known_out_of_scope(request.query):
                        results = []
                    else:
                        engine = _retrieval_engine()
                        results = await engine.retrieve(
                            query=request.query,
                            top_k=params.top_k,
                            similarity_threshold=params.similarity_threshold,
                            enable_rerank=params.enable_rerank,
                            partition=request.partition,
                        )

                    sources = [_to_retrieved_document(doc).model_dump() for doc in results]
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"
                    if not results:
                        message = "抱歉，我在知识库中没有找到足够相关的信息来回答这个问题。"
                        yield f"data: {json.dumps({'type': 'chunk', 'data': message}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'status', 'data': '正在生成答案...'}, ensure_ascii=False)}\n\n"
                        generator = Generator()
                        async for chunk in generator.generate_stream(query=request.query, context_docs=results):
                            yield f"data: {json.dumps({'type': 'chunk', 'data': chunk}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'total_time': time.time() - start}, ensure_ascii=False)}\n\n"
                except Exception as exc:
                    logger.error("Answer stream failed: %s", exc, exc_info=True)
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Answer generation failed'}, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )

        if is_known_out_of_scope(request.query):
            results = []
        else:
            engine = _retrieval_engine()
            results = await engine.retrieve(
                query=request.query,
                top_k=params.top_k,
                similarity_threshold=params.similarity_threshold,
                enable_rerank=params.enable_rerank,
                partition=request.partition,
            )

        if not results:
            answer = "抱歉，我在知识库中没有找到足够相关的信息来回答这个问题。"
        else:
            generator = Generator()
            answer = await generator.generate(query=request.query, context_docs=results)

        return AnswerResponse(
            query=request.query,
            answer=answer,
            sources=[_to_retrieved_document(doc) for doc in results],
            total_time=time.time() - start,
        )
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Answer generation failed") from exc
