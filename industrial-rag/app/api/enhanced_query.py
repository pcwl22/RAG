"""Enhanced query API integrating query understanding."""
import json
import time
from collections.abc import AsyncGenerator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from app.api.input_validation import validate_message_budget
from app.api.retrieval_params import resolve_retrieval_params
from app.service.enhanced_query_service import EnhancedQueryOptions, EnhancedQueryService
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class EnhancedQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    chat_history: list[ChatHistoryMessage] | None = Field(
        None, max_length=100, description="Conversation history"
    )
    enable_coreference: bool = True
    enable_decomposition: bool = True
    enable_rewrite: bool = True
    stream: bool = False
    top_k: int | None = Field(default=None, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    enable_rerank: bool | None = None
    partition: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_total_input(self) -> "EnhancedQueryRequest":
        history = (message.content for message in (self.chat_history or []))
        validate_message_budget([self.query, *history])
        return self


class EnhancedQueryResponse(BaseModel):
    query: str
    understanding: dict[str, Any]
    results: list[dict[str, Any]]
    answer: str
    sub_answers: list[dict[str, Any]] = Field(default_factory=list)


def _options_from_request(request: EnhancedQueryRequest) -> EnhancedQueryOptions:
    params = resolve_retrieval_params(
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        enable_rerank=request.enable_rerank,
    )
    return EnhancedQueryOptions(
        query=request.query,
        chat_history=(
            [message.model_dump() for message in request.chat_history]
            if request.chat_history
            else None
        ),
        enable_coreference=request.enable_coreference,
        enable_decomposition=request.enable_decomposition,
        enable_rewrite=request.enable_rewrite,
        top_k=params.top_k,
        similarity_threshold=params.similarity_threshold,
        enable_rerank=params.enable_rerank,
        partition=request.partition,
    )


def _sse_payload(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/query/enhanced", response_model=EnhancedQueryResponse)
async def enhanced_query(request: EnhancedQueryRequest) -> EnhancedQueryResponse | StreamingResponse:
    start = time.time()
    options = _options_from_request(request)

    if request.stream:

        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                service = EnhancedQueryService()
                async for event in service.stream_events(options):
                    yield _sse_payload(event)
                yield _sse_payload({"type": "done", "total_time": time.time() - start})
            except Exception as exc:
                logger.error("Enhanced query stream failed: %s", exc, exc_info=True)
                yield _sse_payload({"type": "error", "error": "Enhanced query failed"})
                yield _sse_payload({"type": "done", "error": True})

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    try:
        result = await EnhancedQueryService().run(options)
        return EnhancedQueryResponse(
            query=result.query,
            understanding=result.understanding,
            results=result.results,
            answer=result.answer,
            sub_answers=result.sub_answers,
        )
    except Exception as exc:
        logger.error("Enhanced query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Enhanced query failed") from exc
