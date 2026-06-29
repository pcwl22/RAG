"""Hybrid retrieval engine: keyword + vector + RRF."""

from app.utils.config import get_settings
from app.embedding.embedder import encode_query
from app.retrieval.reranker import rerank_documents
from app.vectorstore import qdrant_client as storage_adapter
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _retrieval_config() -> dict:
    return get_settings().get("rag", {}).get("retrieval", {})


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

    def __init__(self):
        self.config = _retrieval_config()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        enable_rerank: bool | None = None,
        partition: str | None = None,
        enable_rrf: bool | None = None,
        enable_dynamic_topk: bool | None = None,
    ) -> list[dict]:
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

        query_embedding = encode_query(query)
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

        if not filtered:
            logger.info("Hybrid retrieval returned no documents after filtering")
            return []

        if enable_rerank:
            filtered = rerank_documents(query, filtered, top_n=top_k)
        else:
            filtered = filtered[:top_k]

        logger.info("Hybrid retrieval returned %s documents", len(filtered))
        return filtered
