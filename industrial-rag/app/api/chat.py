"""
对话API路由
"""
import json
from collections.abc import AsyncGenerator
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from app.api.input_validation import validate_message_budget
from app.api.retrieval_params import resolve_retrieval_params
from app.service.enhanced_query_service import EnhancedQueryOptions, EnhancedQueryService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class Message(BaseModel):
    """消息"""

    role: Literal["user", "assistant"] = Field(..., description="角色")
    content: str = Field(..., min_length=1, max_length=8000, description="消息内容")


class ChatRequest(BaseModel):
    """对话请求"""

    messages: list[Message] = Field(..., min_length=1, max_length=100, description="对话历史")
    use_rag: bool = Field(default=True, description="是否使用RAG")
    top_k: int | None = Field(default=None, description="检索文档数；不传则使用配置默认值", ge=1, le=50)
    similarity_threshold: float | None = Field(
        default=None,
        description="相似度阈值；不传则使用配置默认值",
        ge=0.0,
        le=1.0,
    )
    enable_rerank: bool | None = Field(default=None, description="是否启用重排；不传则使用配置默认值")
    temperature: float = Field(default=0.7, description="非 RAG 对话生成温度", ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, description="非 RAG 对话最大生成 token 数", ge=1, le=4096)

    @model_validator(mode="after")
    def validate_total_input(self) -> "ChatRequest":
        validate_message_budget(message.content for message in self.messages)
        return self


class ChatResponse(BaseModel):
    """对话响应"""

    message: Message
    sources: list[dict] | None = None


def _last_user_message_and_history(
    messages: list[Message],
) -> tuple[Message | None, list[dict[str, str]]]:
    """Select the current user turn and exclude any trailing stale messages."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return messages[index], [message.model_dump() for message in messages[:index]]
    return None, []


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    对话接口（非流式）

    Args:
        request: 对话请求

    Returns:
        对话响应
    """
    user_message, chat_history = _last_user_message_and_history(request.messages)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    try:
        from app.service.chat_service import Generator

        generator = Generator()

        sources = None

        # 如果启用RAG，先检索
        if request.use_rag:
            params = resolve_retrieval_params(
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                enable_rerank=request.enable_rerank,
            )
            result = await EnhancedQueryService().run(EnhancedQueryOptions(
                query=user_message.content,
                chat_history=chat_history,
                top_k=params.top_k,
                similarity_threshold=params.similarity_threshold,
                enable_rerank=params.enable_rerank,
            ))
            sources = result.results
            answer = result.answer
        else:
            # 直接对话，不使用RAG
            answer = await generator.chat(
                messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

        return ChatResponse(message=Message(role="assistant", content=answer), sources=sources)

    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat request failed") from e


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    流式对话接口

    Args:
        request: 对话请求

    Returns:
        SSE流式响应
    """

    async def generate() -> AsyncGenerator[str, None]:
        try:
            from app.service.chat_service import Generator

            generator = Generator()

            # 获取最后一条用户消息
            user_message, chat_history = _last_user_message_and_history(request.messages)
            if not user_message:
                yield f"data: {json.dumps({'type': 'error', 'error': 'No user message found'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'error': True}, ensure_ascii=False)}\n\n"
                return

            # 如果启用RAG，先检索并发送来源
            if request.use_rag:
                params = resolve_retrieval_params(
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    enable_rerank=request.enable_rerank,
                )
                options = EnhancedQueryOptions(
                    query=user_message.content,
                    chat_history=chat_history,
                    top_k=params.top_k,
                    similarity_threshold=params.similarity_threshold,
                    enable_rerank=params.enable_rerank,
                )
                async for event in EnhancedQueryService().stream_events(options):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            else:
                # 直接流式对话
                async for chunk in generator.chat_stream(
                    messages=[
                        {"role": msg.role, "content": msg.content} for msg in request.messages
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ):
                    yield f"data: {json.dumps({'type': 'chunk', 'data': chunk}, ensure_ascii=False)}\n\n"

            # 结束标记
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Stream chat failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': 'Chat stream failed'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'error': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)
