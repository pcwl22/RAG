"""BGE reranker wrapper."""
import os
from typing import Any, cast

from app.embedding.embedder import resolve_torch_device
from app.embedding.model_bundle import prepare_runtime_model
from app.utils.config import get_config_section, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_reranker_model: Any = None


def _reranker_config() -> dict[str, Any]:
    return cast(dict[str, Any], get_config_section("reranker"))


def _fallback(documents: list[dict], top_n: int | None) -> list[dict]:
    mode = str(_reranker_config().get("failure_mode", "closed")).strip().lower()
    if mode == "open":
        return documents[:top_n] if top_n else documents
    return []


def _reranker_document_text(doc: dict[str, Any]) -> str:
    """Use the retrieved child passage when parent context is also present."""
    metadata = doc.get("metadata") or {}
    return str(
        doc.get("child_content")
        or metadata.get("child_content")
        or doc.get("content")
        or ""
    )


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

        prepare_runtime_model(get_settings(), "bge-reranker-v2-m3")
        model_path = cfg.get("model_path", "E:/RAG/models/bge-reranker-v2-m3")
        device = resolve_torch_device(
            os.getenv("RERANKER_DEVICE") or cfg.get("device", "cuda"),
            allow_cpu_fallback=bool(cfg.get("allow_cpu_fallback", True)),
        )
        raw_max_length = cfg.get("max_length")
        max_length: int | None = (
            int(raw_max_length) if raw_max_length is not None else None
        )
        reranker_kwargs: dict[str, Any] = {"device": device}
        if max_length is not None:
            reranker_kwargs["max_length"] = max_length
        _reranker_model = CrossEncoder(model_path, **reranker_kwargs)
        logger.info(f"Reranker loaded successfully from {model_path}")
        return _reranker_model
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        return None


def rerank_documents(
    query: str,
    documents: list[dict],
    top_n: int | None = None,
    *,
    apply_threshold: bool = True,
) -> list[dict]:
    """Rerank documents and preserve vector/reranker diagnostics in metadata.

    ``apply_threshold=False`` is intended for a second pass over candidates
    that were already admitted by one or more independent retrieval queries.
    The caller remains responsible for applying its final admission policy in
    that mode; this keeps a broad common query from deleting a document that
    was highly relevant to one decomposed sub-question.
    """
    if not documents:
        return []

    reranker = load_reranker()
    if not reranker:
        logger.error("Reranker unavailable; applying configured failure mode")
        return _fallback(documents, top_n)

    cfg = _reranker_config()
    batch_size = cfg.get("batch_size", 8)
    # Absolute relevance threshold on the sigmoid-calibrated rerank probability.
    # null/None disables threshold filtering (only top_n is applied).
    threshold = cfg.get("score_threshold")
    if threshold is not None:
        threshold = float(threshold)

    pairs = [[query, _reranker_document_text(doc)] for doc in documents]

    try:
        scores = reranker.predict(pairs, batch_size=batch_size)
        raw_scores = [float(score) for score in scores]

        for doc, raw_score in zip(documents, raw_scores, strict=True):
            # CrossEncoder predict() already applies its configured activation.
            # Keep the value unchanged: applying sigmoid again would squash the
            # score range. Scores from differently worded expansion queries still
            # express relevance to different questions, so the enhanced service
            # performs a final common-query rerank before comparing them.
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
        effective_threshold = threshold if apply_threshold else None
        if effective_threshold is not None:
            kept = [doc for doc in reranked if doc["score"] >= effective_threshold]
        else:
            kept = reranked

        # top_n is an UPPER bound only, never a lower bound.
        if top_n:
            kept = kept[:top_n]

        top_score = kept[0]["score"] if kept else 0.0
        logger.info(
            f"Reranked {len(documents)} docs -> {len(kept)} kept "
            f"(threshold={effective_threshold}, top prob={top_score:.3f})"
        )
        return kept
    except Exception as e:
        logger.error("Reranking failed: %s", e, exc_info=True)
        return _fallback(documents, top_n)
