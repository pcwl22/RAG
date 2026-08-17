"""Hybrid retrieval engine: keyword + vector + RRF."""

import re
import time
from typing import Any

from app.embedding.embedder import encode_query
from app.retrieval.domain_signal_map import is_known_out_of_scope
from app.retrieval.reranker import rerank_documents
from app.utils.config import get_config_section
from app.utils.inference import run_inference
from app.utils.logger import get_logger
from app.utils.metrics import RETRIEVAL_DURATION, RETRIEVAL_REQUESTS
from app.vectorstore import storage_adapter

logger = get_logger(__name__)

_LAW_ALIASES = {
    "中华人民共和国劳动合同法实施条例": (
        "中华人民共和国劳动合同法实施条例",
        "劳动合同法实施条例",
        "实施条例",
    ),
    "中华人民共和国劳动合同法": (
        "中华人民共和国劳动合同法",
        "劳动合同法",
    ),
    "中华人民共和国民法典": ("中华人民共和国民法典", "民法典"),
    "中华人民共和国刑法": ("中华人民共和国刑法", "刑法"),
}

def _explicit_legal_citations(query: str) -> list[tuple[str, str]]:
    """Preserve the law-to-article relationship expressed in query order."""
    law_spans: list[tuple[int, int, str]] = []
    for canonical_law, aliases in _LAW_ALIASES.items():
        for alias in aliases:
            law_spans.extend(
                (match.start(), match.end(), canonical_law)
                for match in re.finditer(re.escape(alias), query)
            )
    selected_laws: list[tuple[int, int, str]] = []
    for span in sorted(law_spans, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(span[0] < kept[1] and kept[0] < span[1] for kept in selected_laws):
            continue
        selected_laws.append(span)

    article_pattern = r"第[一二三四五六七八九十百千万零〇两0-9]+条(?:之[一二三四五六七八九十0-9]+)?"
    tokens: list[tuple[int, str, str]] = [
        (start, "law", law) for start, _end, law in selected_laws
    ]
    tokens.extend((match.start(), "article", match.group()) for match in re.finditer(article_pattern, query))
    tokens.sort(key=lambda item: item[0])

    current_law: str | None = None
    pairs: list[tuple[str, str]] = []
    for _position, token_type, value in tokens:
        if token_type == "law":
            current_law = value
        elif current_law is not None and (current_law, value) not in pairs:
            pairs.append((current_law, value))
    return pairs


def _retrieval_config() -> dict[str, Any]:
    return get_config_section("rag", "retrieval")


def _candidate_count(top_k: int, enable_rerank: bool, config: dict) -> int:
    if not enable_rerank:
        return top_k
    multiplier = int(config.get("rerank_candidate_multiplier", 4))
    minimum = int(config.get("rerank_min_candidates", 20))
    maximum = int(config.get("rerank_max_candidates", 80))
    return max(top_k, min(max(top_k * multiplier, minimum), maximum))


def _deduplicate_results(results: list[dict], max_per_document: int) -> list[dict]:
    if max_per_document <= 0:
        return results

    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []

    for doc in results:
        metadata = doc.get("metadata") or {}
        document_id = str(metadata.get("document_id") or metadata.get("filename") or doc.get("id"))
        parent_id = str(metadata.get("parent_id", metadata.get("chunk_index", doc.get("id"))))
        key = (document_id, parent_id)
        if key in seen:
            continue
        if counts.get(document_id, 0) >= max_per_document:
            continue
        seen.add(key)
        counts[document_id] = counts.get(document_id, 0) + 1
        deduped.append(doc)
    return deduped


class HybridRetrievalEngine:
    """PostgreSQL hybrid retriever."""

    def __init__(self) -> None:
        self.config = _retrieval_config()

    async def retrieve_by_ids(
        self, ids: list[str], partition: str | None = None
    ) -> list[dict]:
        """Fetch controlled mapping targets without fuzzy ranking."""
        return await storage_adapter.get_documents_by_ids(ids, partition)

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        enable_rerank: bool | None = None,
        partition: str | None = None,
        enable_rrf: bool | None = None,
        enable_dynamic_topk: bool | None = None,
        enable_exact_citations: bool = True,
    ) -> list[dict]:
        started = time.perf_counter()
        outcome = "success"
        try:
            return await self._retrieve_impl(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                enable_rerank=enable_rerank,
                partition=partition,
                enable_rrf=enable_rrf,
                enable_dynamic_topk=enable_dynamic_topk,
                enable_exact_citations=enable_exact_citations,
            )
        except Exception:
            outcome = "error"
            raise
        finally:
            RETRIEVAL_REQUESTS.labels(outcome=outcome).inc()
            RETRIEVAL_DURATION.observe(time.perf_counter() - started)

    async def _retrieve_impl(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        enable_rerank: bool | None = None,
        partition: str | None = None,
        enable_rrf: bool | None = None,
        enable_dynamic_topk: bool | None = None,
        enable_exact_citations: bool = True,
    ) -> list[dict]:
        if is_known_out_of_scope(query):
            logger.info("Query is outside the configured corpus scope")
            return []
        if top_k is None:
            top_k = int(self.config.get("top_k", 5))
        if similarity_threshold is None:
            similarity_threshold = float(self.config.get("similarity_threshold", 0.05))
        if enable_rerank is None:
            enable_rerank = bool(self.config.get("enable_rerank", True))
        if enable_rrf is None:
            enable_rrf = bool(self.config.get("enable_rrf", True))
        if enable_dynamic_topk is None:
            enable_dynamic_topk = bool(self.config.get("enable_dynamic_topk", True))

        query_embedding = await run_inference(encode_query, query)
        candidate_k = _candidate_count(top_k, enable_rerank, self.config)

        results = await storage_adapter.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=candidate_k,
            partition=partition,
        )

        # RRF scores are intentionally small, so threshold on original retrieval
        # scores when RRF metadata is available.
        filtered: list[dict] = []
        for doc in results:
            if doc.get("rrf_score") is not None:
                source_score = max(float(doc.get("vector_score") or 0.0), float(doc.get("bm25_score") or 0.0))
            else:
                source_score = float(doc.get("score") or 0.0)
            if source_score >= similarity_threshold:
                filtered.append(doc)

        max_per_document = int(self.config.get("max_chunks_per_document", 6))
        filtered = _deduplicate_results(filtered, max_per_document)

        citation_pairs = _explicit_legal_citations(query) if enable_exact_citations else []
        exact_docs = (
            await storage_adapter.get_documents_by_citations(citation_pairs, partition)
            if citation_pairs
            else []
        )

        if not filtered and not exact_docs:
            logger.info("Hybrid retrieval returned no documents after filtering")
            return []

        if enable_rerank and filtered:
            filtered = await run_inference(rerank_documents, query, filtered, top_n=top_k)
        else:
            filtered = filtered[:top_k]

        if exact_docs:
            exact_ids = {str(doc.get("id")) for doc in exact_docs}
            filtered = [
                *exact_docs,
                *(doc for doc in filtered if str(doc.get("id")) not in exact_ids),
            ][:top_k]

        logger.info("Hybrid retrieval returned %s documents", len(filtered))
        return filtered
