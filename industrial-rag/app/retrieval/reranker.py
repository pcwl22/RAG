"""BGE reranker wrapper."""
from typing import Any

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_reranker_model: Any = None


def _reranker_config() -> dict:
    return get_settings().get("reranker", {})


def load_reranker() -> Any:
    """Load the cross-encoder reranker lazily."""
    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    cfg = _reranker_config()
    if not cfg.get("enabled", False):
        logger.info("Reranker disabled in config")
        return None

    logger.info("Loading reranker model...")
    try:
        from sentence_transformers import CrossEncoder

        model_path = cfg.get("model_path", "E:/RAG/models/bge-reranker-v2-m3")
        device = cfg.get("device", "cuda")
        max_length = cfg.get("max_length")
        _reranker_model = CrossEncoder(model_path, device=device, max_length=max_length)
        logger.info(f"Reranker loaded successfully from {model_path}")
        return _reranker_model
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        return None


def rerank_documents(
    query: str,
    documents: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """Rerank documents and preserve vector/reranker diagnostics in metadata."""
    if not documents:
        return []

    reranker = load_reranker()
    if not reranker:
        logger.warning("Reranker not available, returning original order")
        return documents[:top_n] if top_n else documents

    cfg = _reranker_config()
    batch_size = cfg.get("batch_size", 8)
    # Absolute relevance threshold on the sigmoid-calibrated rerank probability.
    # null/None disables threshold filtering (only top_n is applied).
    threshold = cfg.get("score_threshold")
    if threshold is not None:
        threshold = float(threshold)

    pairs = [[query, doc["content"]] for doc in documents]

    try:
        scores = reranker.predict(pairs, batch_size=batch_size)
        raw_scores = [float(score) for score in scores]

        for doc, raw_score in zip(documents, raw_scores):
            # BGE-reranker's CrossEncoder has default_activation_function=Sigmoid,
            # so predict() ALREADY returns a calibrated relevance probability in
            # [0, 1] that is comparable across queries. Use it directly — applying
            # sigmoid a second time would squash [0,1] into [0.5, 0.73] and destroy
            # the separation between relevant and irrelevant documents.
            rerank_prob = raw_score

            metadata = doc.setdefault("metadata", {})
            # Keep retrieval scores for diagnostics, but do NOT fuse them into
            # the final ordering: on legal text embedding/BM25 scores are noisy
            # and would only dilute the cross-encoder's sharper judgement.
            metadata["vector_score"] = float(doc.get("score", 0.0))
            metadata["rerank_raw_score"] = raw_score
            metadata["rerank_prob"] = rerank_prob
            doc["score"] = rerank_prob

        # Rank purely by cross-encoder probability.
        reranked = sorted(documents, key=lambda x: x["score"], reverse=True)

        # Absolute threshold filtering. Allowed to return 0 docs — "no relevant
        # law found" is a valid and important answer in the legal domain.
        if threshold is not None:
            kept = [doc for doc in reranked if doc["score"] >= threshold]
        else:
            kept = reranked

        # top_n is an UPPER bound only, never a lower bound.
        if top_n:
            kept = kept[:top_n]

        top_score = kept[0]["score"] if kept else 0.0
        logger.info(
            f"Reranked {len(documents)} docs -> {len(kept)} kept "
            f"(threshold={threshold}, top prob={top_score:.3f})"
        )
        return kept
    except Exception as e:
        logger.error(f"Reranking failed: {e}, falling back to original order")
        return documents[:top_n] if top_n else documents
