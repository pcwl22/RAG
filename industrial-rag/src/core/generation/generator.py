"""Answer generation for retrieved RAG contexts."""
import re
from typing import AsyncIterator

from src.core.cache import SemanticCache
from src.core.config import get_settings
from src.models.embedding import encode_query
from src.models.llm import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

ANSWER_FORMAT_INSTRUCTIONS = """

必须严格按以下格式输出，不要添加 Markdown 标题符号（不要使用 ##），不要添加额外章节：
1. 【回答】里直接给结论，不要使用“根据提供的上下文”“根据上下文”“来源：”等前置或来源表述。
2. 匹配文件、具体位置和命中内容只放在对应章节里。
3. 只要任一上下文包含能直接回答问题的法条或规定，就必须基于该条文作答，不要因为其他上下文无关而回答“无法回答”。

【回答】

在这里直接给出答案正文。

【匹配文件】

1. 文件名或来源
2. 文件名或来源

【文件具体位置和内容】

1. 文件名或来源：章节/条款/页码；
内容：命中的原文片段

2. 文件名或来源：章节/条款/页码；
内容：命中的原文片段
"""


def _generation_config() -> dict:
    return get_settings().get("rag", {}).get("generation", {})


def _doc_debug_summary(docs: list[dict], max_items: int = 5, preview_len: int = 120) -> list[dict]:
    summary: list[dict] = []
    for doc in docs[:max_items]:
        metadata = doc.get("metadata") or {}
        content = doc.get("content") or metadata.get("parent_content") or doc.get("child_content", "")
        preview = " ".join(str(content).split())[:preview_len]
        summary.append(
            {
                "id": doc.get("id"),
                "score": round(float(doc.get("score", 0.0) or 0.0), 6),
                "rrf_score": round(float(doc.get("rrf_score", 0.0) or 0.0), 6),
                "filename": metadata.get("filename"),
                "chunk_index": metadata.get("chunk_index", doc.get("chunk_index")),
                "preview": preview,
            }
        )
    return summary


def _doc_source(doc: dict) -> str:
    metadata = doc.get("metadata") or {}
    source = str(metadata.get("filename") or metadata.get("document_id") or "unknown")
    return _clean_source_name(source)


def _clean_source_name(source: str) -> str:
    return re.sub(r"^[0-9a-fA-F-]{32,}_", "", source)


def _doc_location(doc: dict) -> str:
    metadata = doc.get("metadata") or {}
    content = _doc_snippet(doc)
    parts: list[str] = []

    section_numbers = metadata.get("section_numbers")
    if isinstance(section_numbers, list) and section_numbers:
        parts.append("章节：" + "、".join(str(item) for item in section_numbers[:3]))

    title_paths = metadata.get("title_paths")
    if isinstance(title_paths, list) and title_paths:
        title_path = title_paths[0]
        if isinstance(title_path, list):
            parts.append("路径：" + " > ".join(str(item) for item in title_path))
        else:
            parts.append("路径：" + str(title_path))

    article_match = re.search(r"第[一二三四五六七八九十百千万零〇两0-9]+条", content)
    if article_match:
        parts.append("条款：" + article_match.group(0))

    page = metadata.get("page") or metadata.get("page_number") or metadata.get("page_index")
    if page is not None:
        parts.append(f"页码：{page}")

    location = "；".join(parts) if parts else "未提供章节或页码 metadata"
    return f"{location}；\n内容：{content}"


def _doc_snippet(doc: dict, max_len: int = 180) -> str:
    metadata = doc.get("metadata") or {}
    content = str(doc.get("child_content") or doc.get("content") or metadata.get("parent_content") or "")
    content = re.sub(r"\s+", " ", content).strip()
    if len(content) <= max_len:
        return content
    return content[:max_len].rstrip() + "..."


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
            source = _doc_source(doc)
            score = float(doc.get("rrf_score", doc.get("score", 0.0)) or 0.0)
            content = doc.get("content") or metadata.get("parent_content") or doc.get("child_content", "")
            contexts.append(template.format(index=index, source=source, score=score, content=content))
        return "\n".join(contexts)

    def _build_matched_files(self, docs: list[dict]) -> list[str]:
        matched_files: list[str] = []
        for doc in docs:
            source = _doc_source(doc)
            if source and source not in matched_files:
                matched_files.append(source)
        return matched_files

    def _build_file_locations(self, docs: list[dict]) -> list[str]:
        locations: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            source = _doc_source(doc)
            location = _doc_location(doc)
            item = f"{source}：{location}"
            if item not in seen:
                seen.add(item)
                locations.append(item)
        return locations

    def _format_answer(self, answer: str, context_docs: list[dict]) -> str:
        answer = answer.strip()
        answer_match = re.search(r"(?:##\s*)?【回答】", answer)
        if answer_match:
            answer_body = answer[answer_match.end():]
            answer_body = re.split(
                r"\n\s*---\s*\n|\n(?:##\s*)?【匹配文件】|\n(?:##\s*)?【文件具体位置（文件章节页码）】|\n(?:##\s*)?【文件具体位置和内容（文件章节页码）】|\n(?:##\s*)?【文件具体位置和内容】",
                answer_body,
                maxsplit=1,
            )[0]
        else:
            answer_body = answer
        answer_body = re.sub(r"^根据(提供的)?上下文[，,]?\s*", "", answer_body.strip())
        answer_body = re.sub(r"\n+\s*来源：.*", "", answer_body).strip()

        matched_files = self._build_matched_files(context_docs)
        if matched_files:
            files_text = "\n".join(f"{index}. {source}" for index, source in enumerate(matched_files, 1))
        else:
            files_text = "无"

        file_locations = self._build_file_locations(context_docs)
        if file_locations:
            locations_text = "\n".join(f"{index}. {location}" for index, location in enumerate(file_locations, 1))
        else:
            locations_text = "无"

        return f"""【回答】

{answer_body}

【匹配文件】

{files_text}

【文件具体位置和内容】

{locations_text}"""

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
                    return self._format_answer(cached, context_docs)
            except Exception as exc:
                logger.warning("Semantic cache lookup skipped: %s", exc)
                query_embedding = None
        else:
            query_embedding = None

        logger.info(
            "Generating answer with retrieved contexts",
            extra={
                "query": query,
                "context_doc_count": len(context_docs),
                "context_docs": _doc_debug_summary(context_docs),
            },
        )

        prompt = self._build_prompt(query, self._build_context(context_docs)) + ANSWER_FORMAT_INSTRUCTIONS
        answer = await self.llm.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=float(self.config.get("temperature", 0.3)),
            max_tokens=int(self.config.get("max_tokens", 2048)),
        )
        answer = self._format_answer(answer, context_docs)

        if use_cache and query_embedding is not None:
            try:
                await self.cache.set(query_embedding, answer)
            except Exception as exc:
                logger.warning("Semantic cache write skipped: %s", exc)
        return answer

    async def generate_stream(self, query: str, context_docs: list[dict]) -> AsyncIterator[str]:
        prompt = self._build_prompt(query, self._build_context(context_docs)) + ANSWER_FORMAT_INSTRUCTIONS
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
