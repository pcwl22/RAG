"""Answer generation for retrieved RAG contexts."""
from typing import AsyncIterator

from src.core.cache import SemanticCache
from src.core.config import get_settings
from src.models.embedding import encode_query
from src.models.llm import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _generation_config() -> dict:
    return get_settings().get("rag", {}).get("generation", {})


class Generator:
    """Build RAG prompts and call the configured LLM."""

    def __init__(self):
        self.config = _generation_config()
        self.llm = get_llm_client()
        self.cache = SemanticCache()

    def _build_context(self, docs: list[dict]) -> str:
        if not docs:
            return ""

        template = self.config.get(
            "context_template",
            "## 相关文档 {index}\n来源: {source}\n分数: {score:.4f}\n\n{content}\n",
        )

        contexts: list[str] = []
        for index, doc in enumerate(docs, 1):
            metadata = doc.get("metadata") or {}
            source = metadata.get("filename") or metadata.get("document_id") or "unknown"
            score = float(doc.get("rrf_score", doc.get("score", 0.0)) or 0.0)
            content = doc.get("content") or metadata.get("parent_content") or doc.get("child_content", "")
            contexts.append(template.format(index=index, source=source, score=score, content=content))
        return "\n".join(contexts)

    def _build_prompt(self, query: str, context: str) -> str:
        if not context:
            return f"""用户问题：
{query}

知识库中没有检索到相关上下文。请直接说明无法基于当前知识库回答，并提示需要补充的资料类型。"""

        max_length = int(self.config.get("max_context_length", 8000))
        if len(context) > max_length:
            context = context[:max_length] + "\n\n[上下文已截断]"

        return f"""以下是从知识库检索到的上下文：

{context}

---

用户问题：
{query}

请基于上述上下文回答。若上下文不足，请明确指出不足之处。"""

    def _get_system_prompt(self) -> str:
        return self.config.get(
            "system_prompt",
            "你是一个专业的知识库问答助手。请严格基于上下文回答，不编造信息，并标注来源。",
        )

    async def generate(self, query: str, context_docs: list[dict], use_cache: bool = True) -> str:
        if use_cache:
            try:
                query_embedding = encode_query(query)
                cached = await self.cache.get(query_embedding)
                if cached:
                    return cached
            except Exception as exc:
                logger.warning("Semantic cache lookup skipped: %s", exc)
                query_embedding = None
        else:
            query_embedding = None

        prompt = self._build_prompt(query, self._build_context(context_docs))
        answer = await self.llm.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=float(self.config.get("temperature", 0.3)),
            max_tokens=int(self.config.get("max_tokens", 2048)),
        )

        if use_cache and query_embedding is not None:
            try:
                await self.cache.set(query_embedding, answer)
            except Exception as exc:
                logger.warning("Semantic cache write skipped: %s", exc)
        return answer

    async def generate_stream(self, query: str, context_docs: list[dict]) -> AsyncIterator[str]:
        prompt = self._build_prompt(query, self._build_context(context_docs))
        async for chunk in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=float(self.config.get("temperature", 0.3)),
            max_tokens=int(self.config.get("max_tokens", 2048)),
        ):
            yield chunk

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = "\n\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in messages
        )
        return await self.llm.generate(
            prompt=f"{prompt}\n\nAssistant:",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        prompt = "\n\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in messages
        )
        async for chunk in self.llm.generate_stream(
            prompt=f"{prompt}\n\nAssistant:",
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk
