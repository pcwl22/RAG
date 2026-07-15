"""Retrieval engine for vector search and optional reranking."""

from app.embedding.embedder import encode_query
from app.retrieval.reranker import rerank_documents
from app.utils.config import get_settings
from app.utils.inference import run_inference
from app.utils.logger import get_logger
from app.vectorstore import storage_adapter

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

    def __init__(self):
        self.config = _retrieval_config()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        enable_rerank: bool | None = None,
        enable_multimodal: bool = False,
        partition: str | None = None,
    ) -> list[dict]:
        if top_k is None:
            top_k = self.config.get("top_k", 5)
        if similarity_threshold is None:
            similarity_threshold = self.config.get("similarity_threshold", 0.05)
        if enable_rerank is None:
            enable_rerank = self.config.get("enable_rerank", True)

        logger.info(f"Encoding query: {query[:50]}...")
        query_embedding = await run_inference(encode_query, query)

        # Check if hybrid search is enabled
        enable_hybrid = self.config.get("enable_hybrid", False)

        if enable_hybrid:
            # Use hybrid search (vector + BM25 with RRF fusion)
            retrieve_k = _candidate_count(top_k, enable_rerank, self.config)
            results = await storage_adapter.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                top_k=retrieve_k,
                partition=partition,
            )
            logger.info(f"Hybrid search returned {len(results)} docs")
        else:
            # Use vector-only search
            retrieve_k = _candidate_count(top_k, enable_rerank, self.config)
            results = await storage_adapter.search(
                query_embedding=query_embedding,
                top_k=retrieve_k,
                partition=partition,
            )
            logger.info(f"Vector search returned {len(results)} docs")

        filtered = [doc for doc in results if doc["score"] >= similarity_threshold]
        max_per_document = int(self.config.get("max_chunks_per_document", 6))
        filtered = _deduplicate_results(filtered, max_per_document)

        logger.info(
            f"Retrieved {len(results)} docs, {len(filtered)} kept "
            f"(threshold={similarity_threshold}, candidates={retrieve_k})"
        )

        if not filtered:
            return []

        if enable_rerank:
            filtered = await run_inference(rerank_documents, query, filtered, top_n=top_k)
        else:
            filtered = filtered[:top_k]

        logger.info(f"Final results: {len(filtered)} docs")
        return filtered
