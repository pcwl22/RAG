"""Business service for enhanced RAG queries."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.llm.model import get_llm_client
from app.retrieval.hybrid import HybridRetrievalEngine
from app.retrieval.query_understanding import QueryUnderstanding
from app.retrieval.reranker import rerank_documents
from app.retrieval.result_merge import merge_retrieval_results
from app.service.chat_service import (
    Generator,
    format_missing_law_answer,
    missing_requested_laws,
)
from app.utils.config import get_settings
from app.utils.inference import run_inference

FACT_TRIGGERED_CONTEXT_RULES = [
    {
        "terms": ["第三百一十条", "共同犯罪", "事前通谋"],
        "triggers": [
            "共同",
            "共犯",
            "同伙",
            "合谋",
            "通谋",
            "事前",
            "结伙",
            "多人",
            "胁从犯",
            "胁迫",
        ],
    },
    {
        "terms": ["第一百九十一条"],
        "triggers": [
            "洗钱",
            "资金账户",
            "转账",
            "支付结算",
            "跨境",
            "毒品犯罪",
            "黑社会",
            "恐怖",
            "走私",
            "贪污",
            "金融",
        ],
    },
    {
        "terms": ["第二百九十三条之一"],
        "triggers": ["催收", "债务", "高利贷", "非法债务"],
    },
    {
        "terms": ["第三百一十三条"],
        "triggers": ["判决", "裁定", "执行", "拒不执行"],
    },
    {
        "terms": ["第三百零七条"],
        "triggers": ["证人", "作证", "伪证", "证据", "阻止", "毁灭", "伪造"],
    },
    {
        "terms": ["第三十八条"],
        "triggers": [
            "劳动者解除",
            "劳动者可以解除",
            "未缴",
            "社保",
            "未支付",
            "拖欠",
            "劳动保护",
            "劳动条件",
            "规章制度",
        ],
    },
    {
        "terms": ["第四十六条"],
        "triggers": [
            "经济补偿",
            "补偿",
            "第三十八条",
            "第四十条",
            "第四十一条",
            "协商一致",
            "合同期满",
            "终止",
        ],
    }
]


@dataclass(frozen=True)
class EnhancedQueryOptions:
    """Runtime options for one enhanced query."""

    query: str
    chat_history: list[dict[str, str]] | None = None
    enable_coreference: bool = True
    enable_decomposition: bool = True
    enable_rewrite: bool = True
    top_k: int = 5
    similarity_threshold: float = 0.05
    enable_rerank: bool = True
    partition: str | None = None


@dataclass(frozen=True)
class EnhancedQueryResult:
    """Completed enhanced query result."""

    query: str
    understanding: dict[str, Any]
    results: list[dict[str, Any]]
    answer: str
    sub_answers: list[dict[str, Any]] = field(default_factory=list)


def build_aggregate_prompt(query: str, sub_answers: list[dict[str, Any]]) -> str:
    content = "\n\n".join(
        f"子问题 {index}: {item['query']}\n子答案: {item['answer']}"
        for index, item in enumerate(sub_answers, 1)
    )
    return f"""请把以下多个子问题答案聚合成一个最终回答。

原始问题：
{query}

子问题答案：
{content}

要求：
1. 保留对子问题的关键结论。
2. 消除重复信息。
3. 如果子答案指出信息不足，最终答案也要说明。
4. 直接输出最终答案。

补充要求：
1. 只输出最终结论，不要输出“具体分析如下”“子问题”“信息补充说明”等过程性内容。
2. 不要添加来源章节，来源会由系统统一补充。
3. 直接回答原始问题。"""


def _doc_search_text(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    return "\n".join(
        str(part)
        for part in [
            doc.get("id"),
            doc.get("content"),
            doc.get("child_content"),
            metadata.get("article_text"),
            metadata.get("semantic_chunk_id"),
            metadata.get("article_number"),
            metadata.get("legal_citation"),
        ]
        if part
    )


def filter_untriggered_context_docs(
    query: str,
    docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop contexts whose legal path requires facts absent from the user question."""
    filtered: list[dict[str, Any]] = []
    for doc in docs:
        if doc.get("explicit_citation_exact_match"):
            filtered.append(doc)
            continue
        doc_text = _doc_search_text(doc)
        should_drop = False
        for rule in FACT_TRIGGERED_CONTEXT_RULES:
            if not any(term in doc_text for term in rule["terms"]):
                continue
            if any(trigger in query for trigger in rule["triggers"]):
                continue
            should_drop = True
            break
        if not should_drop:
            filtered.append(doc)
    return filtered


def mapped_semantic_chunk_ids(understanding: dict[str, Any]) -> list[str]:
    """Extract article ids from controlled concept-to-article mappings."""
    ids: list[str] = []
    for mapping in understanding.get("concept_article_mappings") or []:
        for article in mapping.get("articles") or []:
            semantic_id = str(article.get("semantic_chunk_id") or "").strip()
            if semantic_id and semantic_id not in ids:
                ids.append(semantic_id)
    return ids


def mapped_article_coverage(
    priority_ids: list[str], docs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Report whether controlled recall targets actually exist in the corpus result."""
    retrieved = {_doc_identity(doc) for doc in docs}
    matched = [identity for identity in priority_ids if identity in retrieved]
    return {
        "expected_ids": priority_ids,
        "retrieved_ids": matched,
        "missing_ids": [identity for identity in priority_ids if identity not in retrieved],
        "coverage": len(matched) / len(priority_ids) if priority_ids else 1.0,
    }


def retrieval_top_k_for_context(options: EnhancedQueryOptions, priority_ids: list[str]) -> int:
    """Use a wider candidate window when controlled article mappings exist."""
    if not priority_ids:
        return options.top_k
    return max(options.top_k, 10, options.top_k + len(priority_ids))


def _doc_identity(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    return str(
        metadata.get("semantic_chunk_id")
        or metadata.get("chunk_id")
        or doc.get("id")
        or ""
    )


def select_dynamic_context_docs(
    docs: list[dict[str, Any]],
    *,
    top_k: int,
    priority_ids: list[str] | None = None,
    enabled: bool = True,
    flat_score_top_n: int = 2,
    min_score_spread: float = 0.05,
    min_score_gap: float = 0.08,
) -> list[dict[str, Any]]:
    """Select a compact final context after scoring against the original query.

    Cross-query reranker scores are not directly comparable. This function is
    called only after the merged candidates have been scored once more against
    the resolved user question. Controlled article mappings are retained even
    when the final score distribution is flat.
    """
    if not docs or top_k <= 0:
        return []
    if not enabled:
        return docs[:top_k]
    ranked = sorted(docs, key=lambda doc: float(doc.get("score") or 0.0), reverse=True)

    priority_order = {
        identity: len(priority_ids or []) - index
        for index, identity in enumerate(priority_ids or [])
    }
    priority_docs = [doc for doc in ranked if _doc_identity(doc) in priority_order]
    priority_docs.sort(key=lambda doc: priority_order[_doc_identity(doc)], reverse=True)

    limit = min(top_k, len(ranked))
    scores = [float(doc.get("score") or 0.0) for doc in ranked[:limit]]
    spread = scores[0] - scores[-1] if len(scores) > 1 else 0.0
    if spread < min_score_spread:
        dynamic_count = min(limit, max(1, flat_score_top_n))
        if priority_docs:
            dynamic_count = 0
    else:
        gaps = [scores[index] - scores[index + 1] for index in range(len(scores) - 1)]
        best_gap = max(gaps, default=0.0)
        dynamic_count = gaps.index(best_gap) + 1 if best_gap >= min_score_gap else limit

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in [*priority_docs, *ranked[:dynamic_count]]:
        identity = _doc_identity(doc) or str(id(doc))
        if identity in seen or len(selected) >= top_k:
            continue
        seen.add(identity)
        copied = doc.copy()
        copied["context_selection"] = (
            "mapped_article" if doc in priority_docs else "dynamic_score"
        )
        selected.append(copied)
    return selected


async def aggregate_sub_answers(query: str, sub_answers: list[dict[str, Any]]) -> str:
    llm = get_llm_client()
    prompt = build_aggregate_prompt(query, sub_answers)
    return await llm.generate(prompt=prompt, max_tokens=1200)


async def aggregate_sub_answers_stream(
    query: str,
    sub_answers: list[dict[str, Any]],
) -> AsyncIterator[str]:
    llm = get_llm_client()
    prompt = build_aggregate_prompt(query, sub_answers)
    async for chunk in llm.generate_stream(prompt=prompt, max_tokens=1200):
        yield chunk


class EnhancedQueryService:
    """Coordinates query understanding, retrieval, and answer generation."""

    def __init__(
        self,
        *,
        query_understanding: QueryUnderstanding | None = None,
        retrieval_engine: HybridRetrievalEngine | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.query_understanding = query_understanding or QueryUnderstanding()
        self.retrieval_engine = retrieval_engine or HybridRetrievalEngine()
        self.generator = generator or Generator()

    async def understand(self, options: EnhancedQueryOptions) -> dict[str, Any]:
        return await self.query_understanding.understand_query(
            query=options.query,
            chat_history=options.chat_history,
            enable_coreference=options.enable_coreference,
            enable_decomposition=options.enable_decomposition,
            enable_rewrite=options.enable_rewrite,
        )

    async def retrieve(
        self,
        understanding: dict[str, Any],
        options: EnhancedQueryOptions,
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        retrieval_queries = understanding["retrieval_queries"]
        priority_ids = mapped_semantic_chunk_ids(understanding)
        retrieval_top_k = retrieval_top_k_for_context(options, priority_ids)
        retrieval_tasks = [
            self.retrieval_engine.retrieve(
                query=query,
                top_k=retrieval_top_k,
                similarity_threshold=options.similarity_threshold,
                enable_rerank=options.enable_rerank,
                partition=options.partition,
            )
            for query in retrieval_queries
        ]
        # Mapped statutes are recall safeguards, so fetch each exact semantic
        # id without the reranker threshold that can otherwise discard a
        # legally mandatory but lexically short provision (for example art. 28).
        result_groups = await asyncio.gather(*retrieval_tasks)
        safeguard_queries: list[str] = []
        if priority_ids:
            exact_retriever = getattr(self.retrieval_engine, "retrieve_by_ids", None)
            if callable(exact_retriever):
                exact_docs = await exact_retriever(priority_ids, options.partition)
                if exact_docs:
                    result_groups.append(exact_docs)
                    safeguard_queries.append("controlled-article-exact-lookup")
            else:
                # Compatibility fallback for custom retrieval engines.
                safeguard_queries = list(priority_ids)
                result_groups.extend(
                    await asyncio.gather(
                        *(
                            self.retrieval_engine.retrieve(
                                query=semantic_chunk_id,
                                top_k=1,
                                similarity_threshold=0.0,
                                enable_rerank=False,
                                partition=options.partition,
                            )
                            for semantic_chunk_id in safeguard_queries
                        )
                    )
                )
        result_groups = [
            filter_untriggered_context_docs(understanding["resolved_query"], group)
            for group in result_groups
        ]
        for group in result_groups:
            for doc in group:
                if doc.get("explicit_citation_exact_match"):
                    identity = _doc_identity(doc)
                    if identity and identity not in priority_ids:
                        priority_ids.append(identity)
        explicit_priority_ids = {
            _doc_identity(doc)
            for group in result_groups
            for doc in group
            if doc.get("explicit_citation_exact_match")
        }
        merge_k = max(options.top_k * 3, options.top_k + len(priority_ids))
        merged_candidates = merge_retrieval_results(
            result_groups,
            merge_k,
            [*retrieval_queries, *safeguard_queries],
            priority_ids=priority_ids,
        )
        understanding["mapped_article_coverage"] = mapped_article_coverage(
            priority_ids, merged_candidates
        )
        retrieval_cfg = get_settings().get("rag", {}).get("retrieval", {})
        has_rerank_scores = any(
            (doc.get("metadata") or {}).get("rerank_prob") is not None
            for doc in merged_candidates
        )
        if options.enable_rerank and has_rerank_scores:
            # Score every merged candidate against one common query so the
            # values used for final selection are comparable.
            final_ranked = await run_inference(
                rerank_documents,
                understanding["resolved_query"],
                merged_candidates,
                top_n=None,
            )
            final_ids = {_doc_identity(doc) for doc in final_ranked}
            # A controlled mapping is a recall safeguard. Keep mapped articles
            # even if the absolute reranker threshold removed them.
            final_ranked.extend(
                doc
                for doc in merged_candidates
                if _doc_identity(doc) in priority_ids and _doc_identity(doc) not in final_ids
            )
        else:
            final_ranked = merged_candidates

        merged_results = select_dynamic_context_docs(
            final_ranked,
            top_k=options.top_k,
            priority_ids=priority_ids,
            enabled=(
                has_rerank_scores
                and bool(retrieval_cfg.get("dynamic_context_selection", True))
            ),
            flat_score_top_n=int(retrieval_cfg.get("flat_score_top_n", 2)),
            min_score_spread=float(retrieval_cfg.get("min_final_rerank_spread", 0.05)),
            min_score_gap=float(retrieval_cfg.get("min_final_rerank_gap", 0.08)),
        )
        if explicit_priority_ids:
            # Explicitly named provisions are the complete requested context;
            # fuzzy neighbors only reduce precision for this query shape.
            merged_results = [
                doc for doc in merged_results if _doc_identity(doc) in explicit_priority_ids
            ]
        query_to_docs = dict(
            zip(retrieval_queries, result_groups[: len(retrieval_queries)], strict=False)
        )
        return merged_results, query_to_docs

    async def generate_sub_answers(
        self,
        subqueries: list[str],
        query_to_docs: dict[str, list[dict[str, Any]]],
        fallback_docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate answers for decomposed subqueries using their own retrieval context."""
        sub_docs = [query_to_docs.get(subquery, fallback_docs) for subquery in subqueries]
        answer_tasks = [
            self.generator.generate(query=subquery, context_docs=docs, use_cache=False)
            for subquery, docs in zip(subqueries, sub_docs, strict=False)
        ]
        answers = await asyncio.gather(*answer_tasks)
        return [
            {"query": subquery, "answer": answer, "result_count": len(docs)}
            for subquery, answer, docs in zip(subqueries, answers, sub_docs, strict=False)
        ]

    async def generate_answer(
        self,
        understanding: dict[str, Any],
        merged_results: list[dict[str, Any]],
        query_to_docs: dict[str, list[dict[str, Any]]],
    ) -> tuple[str, list[dict[str, Any]]]:
        missing_laws = missing_requested_laws(understanding["resolved_query"], merged_results)
        if missing_laws:
            return format_missing_law_answer(missing_laws), []

        if not understanding["is_decomposed"]:
            answer = await self.generator.generate(
                query=understanding["resolved_query"],
                context_docs=merged_results,
                use_cache=False,
            )
            return answer, []

        subqueries = understanding["subqueries"]
        sub_answers = await self.generate_sub_answers(subqueries, query_to_docs, merged_results)
        answer = await aggregate_sub_answers(understanding["resolved_query"], sub_answers)
        return (
            self.generator.format_answer(answer, merged_results, understanding["resolved_query"]),
            sub_answers,
        )

    async def run(self, options: EnhancedQueryOptions) -> EnhancedQueryResult:
        understanding = await self.understand(options)
        merged_results, query_to_docs = await self.retrieve(understanding, options)
        answer, sub_answers = await self.generate_answer(
            understanding,
            merged_results,
            query_to_docs,
        )
        return EnhancedQueryResult(
            query=options.query,
            understanding=understanding,
            results=merged_results,
            answer=answer,
            sub_answers=sub_answers,
        )

    async def stream_events(self, options: EnhancedQueryOptions) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "status", "data": "正在理解问题..."}
        understanding = await self.understand(options)
        yield {"type": "understanding", "data": understanding}

        yield {"type": "status", "data": "正在检索相关文档..."}
        merged_results, query_to_docs = await self.retrieve(understanding, options)
        yield {"type": "sources", "data": merged_results}

        missing_laws = missing_requested_laws(understanding["resolved_query"], merged_results)
        if missing_laws:
            yield {"type": "chunk", "data": format_missing_law_answer(missing_laws)}
            return

        if not understanding["is_decomposed"]:
            yield {"type": "status", "data": "正在生成答案..."}
            async for chunk in self.generator.generate_stream(
                query=understanding["resolved_query"],
                context_docs=merged_results,
            ):
                yield {"type": "chunk", "data": chunk}
            return

        yield {"type": "status", "data": "正在生成子问题答案..."}
        subqueries = understanding["subqueries"]
        sub_answers = await self.generate_sub_answers(subqueries, query_to_docs, merged_results)

        yield {"type": "status", "data": "正在汇总最终答案..."}
        yield {"type": "chunk", "data": "【回答】\n结论：\n"}
        aggregate_chunks: list[str] = []
        async for chunk in aggregate_sub_answers_stream(
            understanding["resolved_query"],
            sub_answers,
        ):
            aggregate_chunks.append(chunk)
            yield {"type": "chunk", "data": chunk}
        yield {
            "type": "chunk",
            "data": self.generator.format_basis_section(
                merged_results,
                answer="".join(aggregate_chunks),
                query=understanding["resolved_query"],
            ),
        }
