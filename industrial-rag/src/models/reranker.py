"""BGE reranker wrapper."""
from typing import Any

from src.core.config import get_settings
from src.utils.logger import get_logger

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
        _reranker_model = CrossEncoder(model_path, device=device)
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
    score_weight = float(cfg.get("score_weight", 0.85))
    score_weight = min(1.0, max(0.0, score_weight))

    pairs = [[query, doc["content"]] for doc in documents]

    try:
        scores = reranker.predict(pairs, batch_size=batch_size)
        raw_scores = [float(score) for score in scores]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score

        for doc, raw_score in zip(documents, raw_scores):
            vector_score = float(doc.get("score", 0.0))
            rerank_score = 0.5 if score_range == 0 else (raw_score - min_score) / score_range
            fused_score = score_weight * rerank_score + (1.0 - score_weight) * vector_score

            metadata = doc.setdefault("metadata", {})
            metadata["vector_score"] = vector_score
            metadata["rerank_raw_score"] = raw_score
            metadata["rerank_score"] = rerank_score
            metadata["rerank_score_weight"] = score_weight
            doc["score"] = fused_score

        reranked = sorted(documents, key=lambda x: x["score"], reverse=True)
        if top_n:
            reranked = reranked[:top_n]

        logger.info(
            f"Reranked {len(documents)} -> {len(reranked)} docs, "
            f"top score: {reranked[0]['score']:.3f}"
        )
        return reranked
    except Exception as e:
        logger.error(f"Reranking failed: {e}, falling back to original order")
        return documents[:top_n] if top_n else documents
