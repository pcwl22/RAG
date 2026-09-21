"""Business service for enhanced RAG queries."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.llm.answer_contract import AnswerContractError, contract_instructions
from app.llm.model import get_llm_client
from app.llm.prompt import build_system_prompt
from app.retrieval.domain_signal_map import is_known_out_of_scope
from app.retrieval.factory import build_retrieval_engine
from app.retrieval.hybrid import HybridRetrievalEngine
from app.retrieval.query_understanding import QueryUnderstanding
from app.retrieval.reranker import rerank_documents
from app.retrieval.result_merge import merge_retrieval_results
from app.service.chat_service import (
    Generator,
    format_missing_citation_answer,
    format_missing_law_answer,
    format_no_context_answer,
    missing_requested_citations,
    missing_requested_laws,
)
from app.utils.config import get_config_section, get_settings
from app.utils.inference import run_inference
from app.utils.logger import get_logger

logger = get_logger(__name__)

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
            "教唆",
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


def build_aggregate_prompt(
    query: str,
    sub_answers: list[dict[str, Any]],
    context: str = "",
    *,
    structured: bool = True,
) -> str:
    content = "\n\n".join(
        f"子问题 {index}: {item['query']}\n子答案: {item['answer']}"
        for index, item in enumerate(sub_answers, 1)
    )
    evidence = context.strip() or "（无可核验上下文）"
    output_protocol = (
        contract_instructions()
        if structured
        else (
            "直接输出最终答案正文，不要输出 JSON、代码围栏、来源章节或伪造的文件位置。"
            "系统会在答案后追加依据。"
        )
    )
    return f"""请把以下多个子问题答案聚合成一个最终回答。

原始问题：
{query}

可核验上下文（唯一事实来源）：
{evidence}

子问题答案：
{content}

要求：
1. 子答案只是候选草稿，不是事实来源；只能使用原始问题和“可核验上下文”中的信息。
2. 保留有上下文直接支持的关键结论，消除重复信息，并保持不同子问题/场景的适用前提不混淆。
3. 每一项事实、法条号、法律后果和适用条件都必须能在原始问题或可核验上下文中直接找到依据。
4. 如果子答案与上下文冲突，以可核验上下文为准；没有直接依据的内容必须删除，并说明当前资料不足。
5. 如果子答案指出信息不足，且上下文确实不能补足，最终答案也要说明。
6. 不得引用上下文没有出现的具体法条号，不得补充训练知识、常识或推测。
7. 直接输出最终答案。

补充要求：
1. 只输出最终结论，不要输出“具体分析如下”“子问题”“信息补充说明”等过程性内容。
2. 不要添加来源章节，来源会由系统统一补充。
3. 直接回答原始问题。

{output_protocol}"""


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
        if doc.get("explicit_citation_exact_match") or doc.get("mapped_article_exact_match"):
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


def final_rerank_query(understanding: dict[str, Any]) -> str:
    """Build a stable common reranker query from model-supported signals."""
    values: list[str] = []
    for value in [
        understanding.get("resolved_query"),
        understanding.get("rewritten_query"),
        *(understanding.get("subqueries") or []),
    ]:
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)

    signals = understanding.get("retrieval_signals") or {}
    if isinstance(signals, dict):
        for signal_values in signals.values():
            if not isinstance(signal_values, list):
                continue
            for value in signal_values:
                text = str(value or "").strip()
                if text and text not in values:
                    values.append(text)
    return "\n".join(values)


def context_coverage_queries(understanding: dict[str, Any]) -> list[str]:
    """Return user-supported query variants that deserve one context slot.

    Structured signal and concept-article queries are recall aids, not
    independent questions. A plain model rewrite is synthetic as well and may
    drift, so it must not displace evidence for the user's resolved question.
    Only an actual compound query reserves additional subquery coverage.
    """
    values: list[str] = []
    for value in [
        understanding.get("resolved_query"),
        *(
            understanding.get("subqueries") or []
            if understanding.get("is_decomposed")
            else []
        ),
    ]:
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)
    return values


def _context_selection_score(doc: dict[str, Any]) -> float:
    """Blend independent-query evidence with the common-query rerank.

    Independent rewrites can give an unrelated provision a saturated score,
    while the common-query pass can under-score a provision that is specific
    to one valid subquery.  A source-heavy blend retains the latter without
    letting one generated rewrite completely overrule the original question.
    """
    scores: dict[str, float] = {}
    for key in ("source_relevance_score", "score"):
        try:
            score = doc.get(key)
            if score is not None:
                scores[key] = float(score)
        except (TypeError, ValueError):
            continue
    if "source_relevance_score" in scores and "score" in scores:
        return 0.8 * scores["source_relevance_score"] + 0.2 * scores["score"]
    if "source_relevance_score" in scores:
        return scores["source_relevance_score"]
    if "score" in scores:
        return scores["score"]
    return 0.0


def _query_consensus_score(doc: dict[str, Any]) -> float:
    """Return reciprocal-rank evidence when multiple queries found a document."""
    ranks = doc.get("query_retrieval_ranks") or {}
    if not isinstance(ranks, dict):
        return 0.0
    reciprocal_ranks: list[float] = []
    for value in ranks.values():
        try:
            rank = int(value)
        except (TypeError, ValueError):
            continue
        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
    return sum(reciprocal_ranks) if len(reciprocal_ranks) >= 2 else 0.0


def select_dynamic_context_docs(
    docs: list[dict[str, Any]],
    *,
    top_k: int,
    priority_ids: list[str] | None = None,
    coverage_queries: list[str] | None = None,
    enabled: bool = True,
    flat_score_top_n: int = 2,
    min_score_spread: float = 0.05,
    min_score_gap: float = 0.08,
) -> list[dict[str, Any]]:
    """Select a compact final context after scoring against a common query.

    Cross-query reranker scores are not directly comparable. This function is
    called only after the merged candidates have been scored once more against
    a common query. Controlled article mappings are retained even when the
    final score distribution is flat. Decomposed queries may also reserve one
    best candidate per subquery so a compound answer does not silently lose a
    part. One reciprocal-rank consensus candidate is retained when independent
    query variants agree, preventing a query-local score scale from erasing it.
    """
    if not docs or top_k <= 0:
        return []
    if not enabled:
        return docs[:top_k]
    ranked = sorted(docs, key=_context_selection_score, reverse=True)

    priority_order = {
        identity: len(priority_ids or []) - index
        for index, identity in enumerate(priority_ids or [])
    }
    priority_docs = [doc for doc in ranked if _doc_identity(doc) in priority_order]
    priority_docs.sort(key=lambda doc: priority_order[_doc_identity(doc)], reverse=True)
    coverage_docs: list[dict[str, Any]] = []
    for query in dict.fromkeys(coverage_queries or []):
        candidates = [
            doc for doc in ranked if query in (doc.get("matched_queries") or [])
        ]
        match = min(
            candidates,
            key=lambda doc: (
                int((doc.get("query_retrieval_ranks") or {}).get(query, 2**31 - 1)),
                -_context_selection_score(doc),
            ),
            default=None,
        )
        if match is not None:
            coverage_docs.append(match)
    consensus_docs = sorted(
        (doc for doc in ranked if _query_consensus_score(doc) > 0.0),
        key=lambda doc: (
            _query_consensus_score(doc),
            _context_selection_score(doc),
        ),
        reverse=True,
    )[:1]

    limit = min(top_k, len(ranked))
    scores = [_context_selection_score(doc) for doc in ranked[:limit]]
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
    for doc in [
        *priority_docs,
        *coverage_docs,
        *consensus_docs,
        *ranked[:dynamic_count],
    ]:
        identity = _doc_identity(doc) or str(id(doc))
        if identity in seen or len(selected) >= top_k:
            continue
        seen.add(identity)
        copied = doc.copy()
        copied["context_selection_score"] = _context_selection_score(doc)
        if doc in priority_docs:
            copied["context_selection"] = "mapped_article"
        elif doc in coverage_docs:
            copied["context_selection"] = "query_coverage"
        elif doc in consensus_docs:
            copied["context_selection"] = "multi_query_consensus"
        else:
            copied["context_selection"] = "dynamic_score"
        selected.append(copied)
    return selected


def _aggregate_generation_kwargs() -> dict[str, Any]:
    config = get_config_section("rag", "generation")
    return {
        "temperature": float(config.get("temperature", 0.0)),
        "max_tokens": int(config.get("max_tokens", 1024)),
    }


async def aggregate_sub_answers(
    query: str,
    sub_answers: list[dict[str, Any]],
    context: str = "",
) -> str:
    llm = get_llm_client()
    generation = get_config_section("rag", "generation")
    contract = generation.get("answer_contract", {})
    structured = isinstance(contract, dict) and bool(contract.get("enabled", False))
    prompt = build_aggregate_prompt(query, sub_answers, context, structured=structured)
    return await llm.generate(
        prompt=prompt,
        system_prompt=build_system_prompt(),
        **_aggregate_generation_kwargs(),
        structured_output=structured,
    )


async def aggregate_sub_answers_stream(
    query: str,
    sub_answers: list[dict[str, Any]],
    context: str = "",
) -> AsyncIterator[str]:
    llm = get_llm_client()
    generation = get_config_section("rag", "generation")
    contract = generation.get("answer_contract", {})
    structured = isinstance(contract, dict) and bool(contract.get("enabled", False))
    prompt = build_aggregate_prompt(query, sub_answers, context, structured=structured)
    async for chunk in llm.generate_stream(
        prompt=prompt,
        system_prompt=build_system_prompt(),
        **_aggregate_generation_kwargs(),
        structured_output=structured,
    ):
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
        self.retrieval_engine = retrieval_engine or build_retrieval_engine()
        self.generator = generator or Generator()

    def _build_aggregate_context(
        self,
        merged_results: list[dict[str, Any]],
        query_to_docs: dict[str, list[dict[str, Any]]],
    ) -> str:
        """Build a bounded, de-duplicated context for the final aggregation.

        The final merged results are authoritative because they are also the
        sources returned to the caller. Include a small per-query recall tail
        so the aggregator can verify each decomposed sub-answer, while using
        the generator's formatter and overall length limit when available.
        """
        ordered_docs = self._build_aggregate_docs(merged_results, query_to_docs)

        builder = getattr(self.generator, "build_context", None)
        if not callable(builder):
            builder = getattr(self.generator, "_build_context", None)
        if callable(builder):
            context = str(builder(ordered_docs) or "")
            # Production Generator already applies both the per-document and
            # global limits while constructing its context entries.  Returning
            # that exact string keeps the aggregate validation map identical
            # to what the model saw.
            return context
        else:
            # Keep custom test/demonstration generators compatible without
            # weakening the production Generator path above.
            context = "\n\n".join(
                str(doc.get("content") or doc.get("child_content") or "")
                for doc in ordered_docs
                if doc.get("content") or doc.get("child_content")
            )

        config = getattr(self.generator, "config", {})
        try:
            max_length = int(config.get("max_context_length", 8000))
        except (AttributeError, TypeError, ValueError):
            max_length = 8000
        if max_length > 0 and len(context) > max_length:
            context = context[:max_length] + "\n\n[上下文已截断]"
        return context

    def _build_aggregate_docs(
        self,
        merged_results: list[dict[str, Any]],
        query_to_docs: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Return the exact ordered documents represented by aggregate context."""
        ordered_docs: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(doc: dict[str, Any]) -> None:
            identity = _doc_identity(doc) or f"object:{id(doc)}"
            if identity in seen:
                return
            seen.add(identity)
            ordered_docs.append(doc)

        for doc in merged_results:
            add(doc)
        for docs in query_to_docs.values():
            for doc in docs[:3]:
                add(doc)
        return ordered_docs

    def _contract_enabled(self) -> bool:
        checker = getattr(self.generator, "_answer_contract_enabled", None)
        return bool(checker()) if callable(checker) else False

    def _bounded_generation_docs(
        self,
        docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        resolver = getattr(self.generator, "context_docs_for_generation", None)
        return list(resolver(docs)) if callable(resolver) else docs

    async def _aggregate_contract_answer(
        self,
        query: str,
        sub_answers: list[dict[str, Any]],
        context: str,
        context_docs: list[dict[str, Any]],
        initial_raw_answer: str | None = None,
        max_attempts: int = 2,
    ) -> str:
        """Validate aggregate output and perform one corrective retry."""
        raw_answer = initial_raw_answer
        if raw_answer is None:
            try:
                raw_answer = await aggregate_sub_answers(query, sub_answers, context=context)
            except Exception as exc:
                logger.warning(
                    "Structured aggregate generation failed",
                    extra={"error_type": type(exc).__name__},
                )
                return format_no_context_answer()
        parser = getattr(self.generator, "parse_contract_answer", None)
        recoverer = getattr(self.generator, "recover_contract_answer", None)
        formatter = getattr(self.generator, "format_verified_answer", None)
        if not callable(parser) or not callable(formatter):
            return self.generator.format_answer(raw_answer, context_docs, query)

        attempts = max(1, min(max_attempts, 2))
        for attempt in range(attempts):
            try:
                validated = parser(raw_answer, context_docs, query)
                return str(formatter(validated, context_docs, query))
            except (AnswerContractError, TypeError, ValueError):
                if attempt >= attempts - 1:
                    if callable(recoverer):
                        try:
                            recovered = recoverer(raw_answer, context_docs, query)
                            return str(formatter(recovered, context_docs, query))
                        except (AnswerContractError, TypeError, ValueError):
                            pass
                    break
                try:
                    raw_answer = await aggregate_sub_answers(
                        query,
                        sub_answers,
                        context=(
                            context
                            + "\n\n上一次聚合输出未通过服务端校验。请只重新输出合法 JSON，"
                            "并逐字引用正确 context_id 对应的连续上下文原文。"
                        ),
                    )
                except Exception as exc:
                    logger.warning(
                        "Structured aggregate retry failed",
                        extra={"error_type": type(exc).__name__},
                    )
                    break
        return format_no_context_answer()

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
        if is_known_out_of_scope(understanding["resolved_query"]):
            # Do not let an LLM rewrite turn a scope rejection into a positive
            # retrieval result.  The original, user-supported query controls
            # corpus admission for the complete enhanced pipeline.
            understanding["out_of_scope"] = True
            return [], {query: [] for query in dict.fromkeys(retrieval_queries)}

        priority_ids = mapped_semantic_chunk_ids(understanding)
        retrieval_top_k = retrieval_top_k_for_context(options, priority_ids)
        retrieval_cfg = get_settings().get("rag", {}).get("retrieval", {})
        merge_multiplier = max(1, int(retrieval_cfg.get("merge_candidate_multiplier", 8)))
        merge_k = max(
            options.top_k * merge_multiplier,
            options.top_k + len(priority_ids),
        )
        retrieval_tasks = [
            self.retrieval_engine.retrieve(
                query=query,
                top_k=retrieval_top_k,
                rerank_top_k=merge_k,
                # This first pass only builds a recall set.  The common-query
                # pass below owns the final threshold so one decomposed query
                # cannot discard a candidate before it can be compared fairly.
                rerank_apply_threshold=False,
                similarity_threshold=options.similarity_threshold,
                enable_rerank=options.enable_rerank,
                partition=options.partition,
                # Only the user-supported resolved query may make cited
                # provisions exclusive. Rewrites and controlled recall queries
                # can contain generated article numbers and are recall aids.
                enable_exact_citations=query == understanding["resolved_query"],
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
                    for doc in exact_docs:
                        doc["mapped_article_exact_match"] = True
                    result_groups.append(exact_docs)
                    safeguard_queries.append("controlled-article-exact-lookup")
            else:
                # Compatibility fallback for custom retrieval engines.
                safeguard_queries = list(priority_ids)
                fallback_groups = await asyncio.gather(
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
                for group in fallback_groups:
                    for doc in group:
                        doc["mapped_article_exact_match"] = True
                result_groups.extend(fallback_groups)
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
        merged_candidates = merge_retrieval_results(
            result_groups,
            merge_k,
            [*retrieval_queries, *safeguard_queries],
            priority_ids=priority_ids,
        )
        understanding["mapped_article_coverage"] = mapped_article_coverage(
            priority_ids, merged_candidates
        )
        has_rerank_scores = any(
            (doc.get("metadata") or {}).get("rerank_prob") is not None
            for doc in merged_candidates
        )
        if options.enable_rerank and has_rerank_scores:
            # Score every merged candidate against one common query so the
            # values used for final selection are comparable.  Thresholding
            # is deferred: a document may have been strongly supported by one
            # decomposed query while looking weak against the broad aggregate.
            final_ranked = await run_inference(
                rerank_documents,
                final_rerank_query(understanding),
                merged_candidates,
                top_n=None,
                apply_threshold=False,
            )
            reranker_cfg = get_settings().get("reranker", {})
            threshold_value = reranker_cfg.get("score_threshold", 0.5)
            threshold = (
                float(threshold_value) if threshold_value is not None else None
            )
            admitted: list[dict[str, Any]] = []
            for doc in final_ranked:
                final_score = _context_selection_score(
                    {"score": doc.get("score")}
                )
                source_score = _context_selection_score(
                    {"source_relevance_score": doc.get("source_relevance_score")}
                )
                if threshold is not None and max(final_score, source_score) < threshold:
                    continue
                metadata = doc.setdefault("metadata", {})
                metadata["final_rerank_score"] = final_score
                admitted.append(doc)
            final_ranked = admitted
            final_ids = {_doc_identity(doc) for doc in final_ranked}
            # A controlled mapping is a recall safeguard. Keep mapped articles
            # even if the final common-query admission policy removed them.
            final_ranked.extend(
                doc
                for doc in merged_candidates
                if _doc_identity(doc) in priority_ids and _doc_identity(doc) not in final_ids
            )
        else:
            final_ranked = merged_candidates

        merged_results = select_dynamic_context_docs(
            final_ranked,
            # Named/mapped provisions are mandatory coverage, not optional
            # ranking candidates. Expand the final context when there are more
            # mandatory provisions than the caller's fuzzy-result budget.
            top_k=max(options.top_k, len(priority_ids)),
            priority_ids=priority_ids,
            # Multi-query retrieval is useful only if a strong candidate from
            # the original question cannot be erased by a generated rewrite
            # (or vice versa) whose score is calibrated on a different query.
            coverage_queries=context_coverage_queries(understanding),
            enabled=(
                has_rerank_scores
                and bool(retrieval_cfg.get("dynamic_context_selection", True))
            ),
            flat_score_top_n=int(retrieval_cfg.get("flat_score_top_n", 2)),
            min_score_spread=float(retrieval_cfg.get("min_final_rerank_spread", 0.05)),
            min_score_gap=float(retrieval_cfg.get("min_final_rerank_gap", 0.08)),
        )
        explicit_priority_ids = {
            _doc_identity(doc)
            for group in result_groups
            for doc in group
            if doc.get("explicit_citation_exact_match")
        }
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
        priority_docs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate answers for decomposed subqueries using their own retrieval context."""
        priority_docs = priority_docs or []
        priority_ids = {_doc_identity(doc) for doc in priority_docs}
        sub_docs: list[list[dict[str, Any]]] = []
        for subquery in subqueries:
            docs = query_to_docs.get(subquery, fallback_docs)
            if priority_docs:
                docs = [
                    *priority_docs,
                    *(doc for doc in docs if _doc_identity(doc) not in priority_ids),
                ]
            sub_docs.append(docs)
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
        coverage_input = (
            self._build_aggregate_docs(merged_results, query_to_docs)
            if understanding["is_decomposed"]
            else merged_results
        )
        coverage_docs = self._bounded_generation_docs(coverage_input)
        missing_citations = missing_requested_citations(
            understanding["resolved_query"], coverage_docs
        )
        if missing_citations:
            return format_missing_citation_answer(missing_citations), []
        missing_laws = missing_requested_laws(understanding["resolved_query"], coverage_docs)
        if missing_laws:
            return format_missing_law_answer(missing_laws), []
        if not merged_results:
            return format_no_context_answer(), []

        if not understanding["is_decomposed"]:
            answer = await self.generator.generate(
                query=understanding["resolved_query"],
                context_docs=merged_results,
                use_cache=False,
            )
            return answer, []

        subqueries = understanding["subqueries"]
        priority_docs = [
            doc
            for doc in merged_results
            if doc.get("mapped_article_priority")
            or doc.get("mapped_article_exact_match")
            or doc.get("explicit_citation_exact_match")
        ]
        sub_answers = await self.generate_sub_answers(
            subqueries,
            query_to_docs,
            merged_results,
            priority_docs=priority_docs,
        )
        aggregate_context = self._build_aggregate_context(merged_results, query_to_docs)
        aggregate_docs = self._build_aggregate_docs(merged_results, query_to_docs)
        if self._contract_enabled():
            answer = await self._aggregate_contract_answer(
                understanding["resolved_query"],
                sub_answers,
                aggregate_context,
                aggregate_docs,
            )
        else:
            raw_answer = await aggregate_sub_answers(
                understanding["resolved_query"],
                sub_answers,
                context=aggregate_context,
            )
            answer = self.generator.format_answer(
                raw_answer,
                merged_results,
                understanding["resolved_query"],
            )
        return answer, sub_answers

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

        coverage_input = (
            self._build_aggregate_docs(merged_results, query_to_docs)
            if understanding["is_decomposed"]
            else merged_results
        )
        coverage_docs = self._bounded_generation_docs(coverage_input)
        missing_citations = missing_requested_citations(
            understanding["resolved_query"], coverage_docs
        )
        if missing_citations:
            yield {"type": "chunk", "data": format_missing_citation_answer(missing_citations)}
            return
        missing_laws = missing_requested_laws(understanding["resolved_query"], coverage_docs)
        if missing_laws:
            yield {"type": "chunk", "data": format_missing_law_answer(missing_laws)}
            return
        if not merged_results:
            yield {"type": "chunk", "data": format_no_context_answer()}
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
        priority_docs = [
            doc
            for doc in merged_results
            if doc.get("mapped_article_priority")
            or doc.get("mapped_article_exact_match")
            or doc.get("explicit_citation_exact_match")
        ]
        sub_answers = await self.generate_sub_answers(
            subqueries,
            query_to_docs,
            merged_results,
            priority_docs=priority_docs,
        )

        yield {"type": "status", "data": "正在汇总最终答案..."}
        aggregate_chunks: list[str] = []
        aggregate_context = self._build_aggregate_context(merged_results, query_to_docs)
        try:
            async for chunk in aggregate_sub_answers_stream(
                understanding["resolved_query"],
                sub_answers,
                context=aggregate_context,
            ):
                aggregate_chunks.append(chunk)
        except Exception as exc:
            if not self._contract_enabled():
                raise
            logger.warning(
                "Structured aggregate stream failed",
                extra={"error_type": type(exc).__name__},
            )
            aggregate_docs = self._build_aggregate_docs(merged_results, query_to_docs)
            answer = await self._aggregate_contract_answer(
                understanding["resolved_query"],
                sub_answers,
                aggregate_context,
                aggregate_docs,
                max_attempts=1,
            )
            for chunk in self.generator._chunk_answer(answer):
                yield {"type": "chunk", "data": chunk}
            return
        raw_aggregate = "".join(aggregate_chunks)
        if self._contract_enabled():
            aggregate_docs = self._build_aggregate_docs(merged_results, query_to_docs)
            # The aggregate stream is buffered until the structured contract
            # is validated, so clients never observe an unverified draft.
            answer = await self._aggregate_contract_answer(
                understanding["resolved_query"],
                sub_answers,
                aggregate_context,
                aggregate_docs,
                initial_raw_answer=raw_aggregate,
            )
            for chunk in self.generator._chunk_answer(answer):
                yield {"type": "chunk", "data": chunk}
            return

        yield {"type": "chunk", "data": "【回答】\n结论：\n"}
        yield {"type": "chunk", "data": raw_aggregate}
        yield {
            "type": "chunk",
            "data": self.generator.format_basis_section(
                merged_results,
                answer=raw_aggregate,
                query=understanding["resolved_query"],
            ),
        }
