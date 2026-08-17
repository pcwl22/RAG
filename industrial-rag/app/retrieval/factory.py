"""Create retrieval engines from the active runtime configuration."""

from typing import Any

from app.utils.config import get_settings


def build_retrieval_engine() -> Any:
    """Return the configured retriever for every API and service entrypoint."""
    retrieval = get_settings().get("rag", {}).get("retrieval", {})
    if bool(retrieval.get("enable_hybrid", True)):
        from app.retrieval.hybrid import HybridRetrievalEngine

        return HybridRetrievalEngine()

    from app.retrieval.dense import RetrievalEngine

    return RetrievalEngine()


__all__ = ["build_retrieval_engine"]
