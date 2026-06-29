"""Query understanding: coreference resolution and decomposition."""
import re

from src.core.config import get_settings
from src.models.llm import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

COREFERENCE_WORDS = [
    "它",
    "他",
    "她",
    "这个",
    "那个",
    "这些",
    "那些",
    "这里",
    "那里",
    "该",
    "上述",
    "前者",
    "后者",
]

COMPOUND_HINTS = [
    "并且",
    "同时",
    "以及",
    "还有",
    "另外",
    "分别",
    "对比",
    "比较",
    "和",
    "或",
    "或者",
    "第一",
    "第二",
    "第三",
    "1.",
    "2.",
    "3.",
]


def _query_understanding_config() -> dict:
    return (
        get_settings()
        .get("rag", {})
        .get("retrieval", {})
        .get("query_understanding", {})
    )


def _clean_numbered_line(text: str) -> str:
    return re.sub(r"^\s*(?:[-*]|\d+[.)、]|[一二三四五六七八九十]+[、.])\s*", "", text).strip()


class QueryUnderstanding:
    """LLM-backed query understanding module."""

    def __init__(self):
        self.llm = get_llm_client()
        self.config = _query_understanding_config()

    async def resolve_coreference(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Rewrite pronouns such as "它/那个" into explicit entities."""
        if not chat_history or not any(word in query for word in COREFERENCE_WORDS):
            return query

        recent_history = chat_history[-6:]
        history_context = "\n".join(
            f"{'用户' if msg.get('role') == 'user' else '助手'}: {msg.get('content', '')}"
            for msg in recent_history
        )
        prompt = f"""请根据对话历史改写当前问题，把“它、这个、那个、上述、前者、后者”等指代词还原为明确实体。

对话历史：
{history_context}

当前问题：
{query}

要求：
1. 只输出改写后的问题。
2. 不要解释。
3. 如果没有足够上下文，请保持原问题。
"""
        try:
            resolved = (await self.llm.generate(prompt=prompt, max_tokens=160)).strip()
            return resolved or query
        except Exception as exc:
            logger.warning("Coreference resolution failed: %s", exc)
            return query

    async def rewrite_query(self, query: str) -> str:
        """把口语化/场景化提问改写成贴近法律条文表述的检索式。

        场景化提问（如"小明被瓷砖砸伤谁赔"）用的是生活语言，而法条用的是
        抽象术语（"建筑物搁置物脱落致害责任"），向量和重排都对不上。改写成
        法言法语能显著提升召回。失败时回退原查询。
        """
        prompt = f"""你是法律检索助手。请把用户的口语化或场景化问题改写成贴近法律条文表述的检索查询，便于在法律法规库中检索。

要求：
1. 提取问题中的法律关系和核心事实，转换为规范的法律概念与术语。
2. 去掉具体人名、地名等与检索无关的细节（如"小明""某小区"），只保留法律要件。
3. 只输出改写后的检索查询，一行，不要解释。
4. 如果原问题已经是规范的法律表述，原样输出。

示例：
原问题：小明在小区里被业主共有的外墙脱落瓷砖砸伤，找不到具体是谁，谁来赔？
改写：建筑物搁置物悬挂物脱落坠落致人损害的责任承担 无法确定具体侵权人

原问题：{query}
改写："""
        try:
            rewritten = (await self.llm.generate(prompt=prompt, max_tokens=160)).strip()
            return rewritten or query
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
            return query

    async def decompose_query(self, query: str, max_subqueries: int | None = None) -> list[str]:
        """Split a compound question into independent subquestions."""
        max_subqueries = max_subqueries or int(self.config.get("max_subqueries", 3))
        hint_count = sum(1 for hint in COMPOUND_HINTS if hint in query)
        question_count = query.count("?") + query.count("？")
        if hint_count < 1 and question_count < 2:
            return [query]

        prompt = f"""请判断下面的问题是否包含多个可独立检索的子问题。如果包含，请拆成 2-{max_subqueries} 个独立子问题；如果不包含，请原样输出。

原问题：
{query}

输出要求：
1. 每行一个子问题。
2. 不要编号。
3. 不要解释。
4. 每个子问题都必须保留必要实体，能够单独检索。
"""
        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=320)
            subqueries = [_clean_numbered_line(line) for line in response.splitlines()]
            subqueries = [line for line in subqueries if line]
            if not subqueries:
                return [query]
            if len(subqueries) == 1:
                return [query]
            return subqueries[:max_subqueries]
        except Exception as exc:
            logger.warning("Query decomposition failed: %s", exc)
            return [query]

    async def understand_query(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        enable_coreference: bool = True,
        enable_decomposition: bool = True,
        enable_rewrite: bool = True,
    ) -> dict:
        resolved_query = query
        if enable_coreference and chat_history:
            resolved_query = await self.resolve_coreference(query, chat_history)

        # 查询改写：口语/场景化 -> 法言法语。改写结果只用于检索，不替换原查询。
        rewritten_query = resolved_query
        if enable_rewrite:
            rewritten_query = await self.rewrite_query(resolved_query)

        # 问题拆分基于改写后的查询（复合问题改写后仍是复合的）。
        subqueries = [rewritten_query]
        if enable_decomposition:
            subqueries = await self.decompose_query(rewritten_query)
        is_decomposed = len(subqueries) > 1

        # 检索查询集 = 原查询 + 改写/拆分查询，合并去重。
        # 与 is_decomposed 区分开：原+改写不是"子问题"，不应触发分别生成答案再聚合。
        retrieval_queries: list[str] = []
        for q in [resolved_query, *subqueries]:
            if q and q not in retrieval_queries:
                retrieval_queries.append(q)

        return {
            "original_query": query,
            "resolved_query": resolved_query,
            "rewritten_query": rewritten_query,
            "subqueries": subqueries,
            "retrieval_queries": retrieval_queries,
            "is_decomposed": is_decomposed,
        }
