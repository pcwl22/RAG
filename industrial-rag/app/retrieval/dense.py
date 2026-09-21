"""Retrieval engine for vector search and optional reranking."""
from typing import Any

from app.embedding.embedder import encode_query
from app.retrieval.domain_signal_map import is_known_out_of_scope
from app.retrieval.hybrid import explicit_legal_citations
from app.retrieval.reranker import rerank_documents
from app.utils.config import get_config_section
from app.utils.inference import run_inference
from app.utils.logger import get_logger, text_log_metadata
from app.vectorstore import storage_adapter

logger = get_logger(__name__)


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
    deduped: list[dict] = []
    seen_chunks: set[tuple[str, int]] = set()

    for doc in results:
        metadata = doc.get("metadata", {}) or {}
        document_id = str(metadata.get("document_id") or metadata.get("filename") or doc.get("id"))
        chunk_index = int(metadata.get("chunk_index", doc.get("chunk_index", 0)) or 0)
        key = (document_id, chunk_index)

        if key in seen_chunks:
            continue
        if counts.get(document_id, 0) >= max_per_document:
            continue

        seen_chunks.add(key)
        counts[document_id] = counts.get(document_id, 0) + 1
        deduped.append(doc)

    return deduped


class RetrievalEngine:
    """Vector retrieval with optional cross-encoder reranking."""

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
        rerank_top_k: int | None = None,
        rerank_apply_threshold: bool | None = None,
        similarity_threshold: float | None = None,
        enable_rerank: bool | None = None,
        partition: str | None = None,
        enable_exact_citations: bool = True,
    ) -> list[dict]:
        if is_known_out_of_scope(query):
            logger.info("Query is outside the configured corpus scope")
            return []
        if top_k is None:
            top_k = self.config.get("top_k", 5)
        if similarity_threshold is None:
            similarity_threshold = self.config.get("similarity_threshold", 0.05)
        if enable_rerank is None:
            enable_rerank = self.config.get("enable_rerank", True)
        if rerank_apply_threshold is None:
            rerank_apply_threshold = True

        # Enhanced retrieval merges several independent query results.  Keep
        # a wider reranked candidate set for that merger while preserving the
        # ordinary ``top_k`` contract for callers that do not opt in.
        result_top_k = top_k
        if enable_rerank and rerank_top_k is not None:
            result_top_k = max(top_k, int(rerank_top_k))

        logger.info("Encoding query", extra=text_log_metadata(query, "query"))
        query_embedding = await run_inference(encode_query, query)

        # Check if hybrid search is enabled
        enable_hybrid = self.config.get("enable_hybrid", False)

        if enable_hybrid:
            # Use hybrid search (vector + BM25 with RRF fusion)
            retrieve_k = _candidate_count(result_top_k, enable_rerank, self.config)
            results = await storage_adapter.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                top_k=retrieve_k,
                partition=partition,
                enable_rrf=bool(self.config.get("enable_rrf", True)),
                enable_dynamic_topk=bool(self.config.get("enable_dynamic_topk", True)),
            )
            logger.info(f"Hybrid search returned {len(results)} docs")
        else:
            # Use vector-only search
            retrieve_k = _candidate_count(result_top_k, enable_rerank, self.config)
            results = await storage_adapter.search(
                query_embedding=query_embedding,
                top_k=retrieve_k,
                partition=partition,
            )
            logger.info(f"Vector search returned {len(results)} docs")

        filtered = [doc for doc in results if doc["score"] >= similarity_threshold]
        max_per_document = int(self.config.get("max_chunks_per_document", 6))
        filtered = _deduplicate_results(filtered, max_per_document)
        citation_pairs = explicit_legal_citations(query) if enable_exact_citations else []
        exact_docs = (
            await storage_adapter.get_documents_by_citations(citation_pairs, partition)
            if citation_pairs
            else []
        )

        logger.info(
            f"Retrieved {len(results)} docs, {len(filtered)} kept "
            f"(threshold={similarity_threshold}, candidates={retrieve_k})"
        )

        if not filtered and not exact_docs:
            return []

        if enable_rerank:
            filtered = await run_inference(
                rerank_documents,
                query,
                filtered,
                top_n=result_top_k,
                apply_threshold=rerank_apply_threshold,
            )
        else:
            filtered = filtered[:top_k]

        if exact_docs:
            exact_ids = {str(doc.get("id")) for doc in exact_docs}
            result_limit = max(result_top_k, len(exact_docs))
            filtered = [
                *exact_docs,
                *(doc for doc in filtered if str(doc.get("id")) not in exact_ids),
            ][:result_limit]

        logger.info(f"Final results: {len(filtered)} docs")
        return filtered
