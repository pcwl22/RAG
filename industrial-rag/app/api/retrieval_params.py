"""Shared retrieval parameter defaults for API routes."""
from dataclasses import dataclass
from typing import Any

from app.utils.config import get_config_section


@dataclass(frozen=True)
class RetrievalParams:
    top_k: int
    similarity_threshold: float
    enable_rerank: bool


def _retrieval_config() -> dict[str, Any]:
    return get_config_section("rag", "retrieval")


def resolve_retrieval_params(
    *,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    enable_rerank: bool | None = None,
) -> RetrievalParams:
    """Resolve request overrides against the configured retrieval defaults."""
    cfg = _retrieval_config()
    return RetrievalParams(
        top_k=int(top_k if top_k is not None else cfg.get("top_k", 5)),
        similarity_threshold=float(
            similarity_threshold
            if similarity_threshold is not None
            else cfg.get("similarity_threshold", 0.05)
        ),
        enable_rerank=bool(
            enable_rerank
            if enable_rerank is not None
            else cfg.get("enable_rerank", True)
        ),
    )
