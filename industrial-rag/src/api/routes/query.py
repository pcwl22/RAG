"""Query API routes."""
import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.generation.generator import Generator
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    enable_rerank: bool = True
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
    top_k: int = Field(default=5, ge=1, le=50)
    enable_rerank: bool = True
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
    if retrieval_cfg.get("use_postgres", True) or retrieval_cfg.get("enable_hybrid", True):
        from src.core.retrieval.hybrid_engine import HybridRetrievalEngine

        return HybridRetrievalEngine()

    from src.core.retrieval.engine import RetrievalEngine

    return RetrievalEngine()


def _to_retrieved_document(doc: dict) -> RetrievedDocument:
    score = float(doc.get("rrf_score", doc.get("score", 0.0)) or 0.0)
    metadata = doc.get("metadata") or {}
    return RetrievedDocument(
        id=str(doc.get("id")),
        content=doc.get("content", ""),
        score=score,
        metadata=metadata,
        chunk_index=int(metadata.get("chunk_index", doc.get("chunk_index", 0)) or 0),
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    start = time.time()
    try:
        engine = _retrieval_engine()
        results = await engine.retrieve(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            enable_rerank=request.enable_rerank,
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
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/answer", response_model=AnswerResponse)
async def answer_question(request: AnswerRequest) -> AnswerResponse | StreamingResponse:
    start = time.time()
    try:
        engine = _retrieval_engine()
        results = await engine.retrieve(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=0.2,
            enable_rerank=request.enable_rerank,
            partition=request.partition,
        )

        generator = Generator()
        if request.stream:

            async def stream_generator():
                sources = [_to_retrieved_document(doc).model_dump() for doc in results]
                yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"
                if not results:
                    message = "抱歉，我在知识库中没有找到足够相关的信息来回答这个问题。"
                    yield f"data: {json.dumps({'type': 'chunk', 'data': message}, ensure_ascii=False)}\n\n"
                else:
                    async for chunk in generator.generate_stream(request.query, results):
                        yield f"data: {json.dumps({'type': 'chunk', 'data': chunk}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'total_time': time.time() - start}, ensure_ascii=False)}\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        if not results:
            answer = "抱歉，我在知识库中没有找到足够相关的信息来回答这个问题。"
        else:
            answer = await generator.generate(query=request.query, context_docs=results)

        return AnswerResponse(
            query=request.query,
            answer=answer,
            sources=[_to_retrieved_document(doc) for doc in results],
            total_time=time.time() - start,
        )
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
