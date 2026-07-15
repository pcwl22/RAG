"""Query understanding: coreference resolution and decomposition."""
import json
import re

from app.llm.model import get_llm_client
from app.retrieval.domain_signal_map import build_domain_signal_queries
from app.retrieval.legal_concept_map import (
    build_concept_article_queries,
    match_legal_concept_articles,
)
from app.utils.config import get_settings
from app.utils.logger import get_logger

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

FORMAL_LEGAL_MARKERS = [
    "罪",
    "刑法",
    "民法典",
    "公司法",
    "劳动合同法",
    "法条",
    "条款",
    "核心区别",
    "区别",
    "构成要件",
    "犯罪主体",
    "犯罪客体",
    "量刑",
    "刑罚",
]

FACT_TRIGGERED_SIGNAL_TERMS = {
    "共同犯罪": ["共同", "共犯", "同伙", "合谋", "通谋", "事前", "结伙", "多人"],
    "连续犯": ["连续犯"],
}


def _query_understanding_config() -> dict:
    return (
        get_settings()
        .get("rag", {})
        .get("retrieval", {})
        .get("query_understanding", {})
    )


def _clean_numbered_line(text: str) -> str:
    return re.sub(r"^\s*(?:[-*]|\d+[.)、]|[一二三四五六七八九十]+[、.])\s*", "", text).strip()


def _clean_rewritten_query(text: str) -> str:
    """Normalize LLM rewrite output to a single query line."""
    lines = [line.strip() for line in text.strip().strip("`").splitlines() if line.strip()]
    if not lines:
        return ""

    selected = lines[0]
    for line in reversed(lines):
        if re.match(r"^(?:改写|检索查询|查询)[:：]", line):
            selected = re.sub(r"^(?:改写|检索查询|查询)[:：]\s*", "", line).strip()
            break

    selected = re.sub(r"^(?:改写|检索查询|查询)[:：]\s*", "", selected).strip()
    return selected.strip("\"'“”‘’")


def _extract_json_object(text: str) -> dict:
    text = text.strip().strip("`")
    if text.startswith("json"):
        text = text[4:].strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_signal_values(value) -> list[str]:
    if isinstance(value, str):
        raw_values = re.split(r"[，,、;/；\n]+", value)
    elif isinstance(value, list):
        raw_values = value
    else:
        return []

    values: list[str] = []
    for item in raw_values:
        text = str(item).strip().strip("\"'“”‘’")
        if text and text not in values:
            values.append(text)
    return values[:6]


def _drop_unsupported_article_terms(value: str, allowed_articles: set[str]) -> str:
    article_terms = _article_terms(value)
    if not article_terms:
        return value
    if set(article_terms).issubset(allowed_articles):
        return value

    cleaned = value
    for article in article_terms:
        if article not in allowed_articles:
            cleaned = cleaned.replace(article, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _drop_untriggered_signal_terms(value: str, query: str) -> str:
    cleaned = value
    for term, triggers in FACT_TRIGGERED_SIGNAL_TERMS.items():
        if term not in cleaned:
            continue
        if any(trigger in query for trigger in triggers):
            continue
        cleaned = cleaned.replace(term, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_signal_values_for_query(value, query: str) -> list[str]:
    allowed_articles = set(_article_terms(query))
    values: list[str] = []
    for item in _normalize_signal_values(value):
        cleaned = _drop_unsupported_article_terms(item, allowed_articles)
        cleaned = _drop_untriggered_signal_terms(cleaned, query)
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return values[:6]


def _build_signal_query(signals: dict) -> str:
    ordered_fields = ["核心法律概念", "行为", "主体", "结果", "争议点"]
    terms: list[str] = []
    for field in ordered_fields:
        for value in _normalize_signal_values(signals.get(field)):
            if value not in terms:
                terms.append(value)
    return " ".join(terms)


def _article_terms(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"第[一二三四五六七八九十百千万零〇两0-9]+条", text)))


def _crime_terms(text: str) -> list[str]:
    terms: list[str] = []
    parts = re.split(r"(?:以及|或者|还是|比较|对比|区别|[，,。？！?、\s与和及或])+", text)
    for part in parts:
        for match in re.findall(r"[\u4e00-\u9fff]{2,12}罪", part):
            terms.append(match.lstrip("与和及或"))
    return list(dict.fromkeys(terms))


def _chinese_bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"[^\u4e00-\u9fff]", "", text)
    return {cleaned[index:index + 2] for index in range(max(len(cleaned) - 1, 0))}


def _looks_like_formal_legal_query(query: str) -> bool:
    return any(marker in query for marker in FORMAL_LEGAL_MARKERS) or bool(_article_terms(query))


def _is_related_rewrite(original: str, rewritten: str) -> bool:
    """Reject LLM rewrites that drift to an unrelated legal topic."""
    if not rewritten or rewritten == original:
        return True

    for term in _crime_terms(original):
        root = term[:-1] if term.endswith("罪") else term
        if term not in rewritten and root not in rewritten:
            return False

    for term in _article_terms(original):
        if term not in rewritten:
            return False

    if not _looks_like_formal_legal_query(original):
        return True

    original_bigrams = _chinese_bigrams(original)
    rewritten_bigrams = _chinese_bigrams(rewritten)
    if not original_bigrams or not rewritten_bigrams:
        return True

    overlap = len(original_bigrams & rewritten_bigrams) / min(len(original_bigrams), len(rewritten_bigrams))
    return overlap >= 0.15


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
5. 不得复用示例内容；改写必须保留原问题的核心法律对象（如罪名、法条、法律关系）。
6. 不要推断或补充原问题没有明确提到的具体法条号。

示例：
原问题：小明在小区里被业主共有的外墙脱落瓷砖砸伤，找不到具体是谁，谁来赔？
改写：建筑物搁置物悬挂物脱落坠落致人损害的责任承担 无法确定具体侵权人

原问题：{query}
改写："""
        try:
            rewritten = _clean_rewritten_query(await self.llm.generate(prompt=prompt, max_tokens=160))
            if not _is_related_rewrite(query, rewritten):
                logger.warning(
                    "Rejected unrelated query rewrite",
                    extra={"original_query": query, "rewritten_query": rewritten},
                )
                return query
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

    async def extract_retrieval_signals(self, query: str) -> dict:
        """Extract generic legal retrieval facets.

        The LLM is still forbidden to invent article numbers here. Controlled
        concept-to-article mapping is applied separately from a local table.
        """
        prompt = f"""你是法律检索助手。请从用户问题中提取用于检索的通用法律语义信号，不要推断或补充具体法条号，不要做“关键词到法条”的映射。

用户问题：
{query}

只输出 JSON 对象，字段固定为：
{{
  "核心法律概念": [],
  "行为": [],
  "主体": [],
  "结果": [],
  "争议点": []
}}

要求：
1. 只抽取问题本身能支持的词语或短语。
2. 可以做同义的法言法语概括，但不得添加具体条号。
3. 每个数组最多 5 项。
4. 不要解释。"""
        try:
            data = _extract_json_object(await self.llm.generate(prompt=prompt, max_tokens=260))
            allowed = ["核心法律概念", "行为", "主体", "结果", "争议点"]
            return {key: _normalize_signal_values_for_query(data.get(key), query) for key in allowed}
        except Exception as exc:
            logger.warning("Retrieval signal extraction failed: %s", exc)
            return {}

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

        retrieval_signals = await self.extract_retrieval_signals(resolved_query)
        signal_query = _build_signal_query(retrieval_signals)
        domain_signal_queries = build_domain_signal_queries(
            resolved_query,
            rewritten_query,
            signal_query,
        )
        concept_article_mappings = match_legal_concept_articles(
            resolved_query,
            rewritten_query,
            retrieval_signals,
        )
        concept_article_queries = build_concept_article_queries(concept_article_mappings)

        # 检索查询集 = 原查询 + 改写/拆分查询 + 结构化语义信号，合并去重。
        # 与 is_decomposed 区分开：原+改写不是"子问题"，不应触发分别生成答案再聚合。
        retrieval_queries: list[str] = []
        for q in [
            resolved_query,
            *subqueries,
            signal_query,
            *domain_signal_queries,
            *concept_article_queries,
        ]:
            if q and q not in retrieval_queries:
                retrieval_queries.append(q)

        return {
            "original_query": query,
            "resolved_query": resolved_query,
            "rewritten_query": rewritten_query,
            "retrieval_signals": retrieval_signals,
            "concept_article_mappings": concept_article_mappings,
            "subqueries": subqueries,
            "retrieval_queries": retrieval_queries,
            "is_decomposed": is_decomposed,
        }
