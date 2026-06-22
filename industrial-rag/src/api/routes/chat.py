"""
对话API路由
"""
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class Message(BaseModel):
    """消息"""

    role: str = Field(..., description="角色: user/assistant/system")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """对话请求"""

    messages: List[Message] = Field(..., description="对话历史")
    use_rag: bool = Field(default=True, description="是否使用RAG")
    top_k: int = Field(default=5, description="检索文档数", ge=1, le=20)
    temperature: float = Field(default=0.7, description="生成温度", ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, description="最大生成token数", ge=1, le=4096)


class ChatResponse(BaseModel):
    """对话响应"""

    message: Message
    sources: List[dict] | None = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    对话接口（非流式）

    Args:
        request: 对话请求

    Returns:
        对话响应
    """
    try:
        from src.core.generation.generator import Generator

        generator = Generator()

        # 获取最后一条用户消息
        user_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        sources = None

        # 如果启用RAG，先检索
        if request.use_rag:
            from src.core.retrieval.engine import RetrievalEngine

            engine = RetrievalEngine()
            results = await engine.retrieve(
                query=user_message.content, top_k=request.top_k, similarity_threshold=0.5
            )
            sources = results

            # 生成带上下文的答案
            answer = await generator.generate(
                query=user_message.content,
                context_docs=results,
            )
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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口

    Args:
        request: 对话请求

    Returns:
        SSE流式响应
    """

    async def generate():
        try:
            from src.core.generation.generator import Generator

            generator = Generator()

            # 获取最后一条用户消息
            user_message = next(
                (msg for msg in reversed(request.messages) if msg.role == "user"), None
            )
            if not user_message:
                yield f"data: {{'error': 'No user message found'}}\n\n"
                return

            # 如果启用RAG，先检索并发送来源
            if request.use_rag:
                from src.core.retrieval.engine import RetrievalEngine

                engine = RetrievalEngine()
                results = await engine.retrieve(
                    query=user_message.content, top_k=request.top_k, similarity_threshold=0.5
                )

                # 发送来源信息
                import json

                sources_data = [
                    {
                        "id": doc["id"],
                        "content": doc["content"][:200],
                        "score": doc["score"],
                    }
                    for doc in results
                ]
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_data}, ensure_ascii=False)}\n\n"

                # 流式生成答案
                async for chunk in generator.generate_stream(
                    query=user_message.content,
                    context_docs=results,
                ):
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk}, ensure_ascii=False)}\n\n"
            else:
                # 直接流式对话
                async for chunk in generator.chat_stream(
                    messages=[
                        {"role": msg.role, "content": msg.content} for msg in request.messages
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ):
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk}, ensure_ascii=False)}\n\n"

            # 结束标记
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream chat failed: {e}", exc_info=True)
            import json

            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
