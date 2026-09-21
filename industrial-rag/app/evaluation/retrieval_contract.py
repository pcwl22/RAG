"""Validation helpers for deterministic retrieval evaluation contracts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_RUNTIME_IDENTITY_KEYS = frozenset({"provider", "model_name", "endpoint_sha256"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RETRIEVAL_RUNTIME_CONTRACT_SCHEMA_VERSION = 1
_RETRIEVAL_ARTIFACT_NAMES = (
    "domain_signal_queries.json",
    "legal_concept_mappings.json",
    "out_of_scope_signals.json",
)


def build_llm_runtime_identity(
    *,
    provider: str,
    model_name: str,
    base_url: str | None,
) -> dict[str, Any]:
    """Build the canonical, non-secret identity used across release gates."""
    endpoint = str(base_url or "").strip().rstrip("/")
    identity = normalize_llm_runtime_identity(
        {
            "provider": provider,
            "model_name": model_name,
            "endpoint_sha256": (
                hashlib.sha256(endpoint.encode("utf-8")).hexdigest() if endpoint else None
            ),
        }
    )
    if identity is None:
        raise ValueError("LLM runtime identity is invalid")
    return identity


def llm_runtime_identity_from_client(llm: Any) -> dict[str, Any]:
    """Extract the canonical identity from the application's configured LLM client."""
    provider = str(getattr(llm, "provider", "") or "").strip().lower()
    config = getattr(llm, "config", None)
    if not provider or not isinstance(config, dict):
        raise ValueError("LLM runtime identity is unavailable")
    model_name = str(config.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("LLM model identity is unavailable")
    return build_llm_runtime_identity(
        provider=provider,
        model_name=model_name,
        base_url=str(config.get("base_url") or ""),
    )


def normalize_sha256(value: Any) -> str | None:
    """Normalize a bare SHA-256 digest used in evaluation evidence."""
    digest = str(value or "").strip().lower()
    return digest if _SHA256_RE.fullmatch(digest) else None


def normalize_llm_runtime_identity(value: Any) -> dict[str, Any] | None:
    """Return a bounded, non-secret identity or ``None`` for an invalid value."""
    if not isinstance(value, dict) or set(value) != _RUNTIME_IDENTITY_KEYS:
        return None

    provider = str(value.get("provider") or "").strip().lower()
    model_name = str(value.get("model_name") or "").strip()
    if not provider or not model_name or len(provider) > 64 or len(model_name) > 256:
        return None
    if any(ord(char) < 32 for char in provider + model_name):
        return None

    endpoint = value.get("endpoint_sha256")
    if endpoint is not None:
        endpoint = normalize_sha256(endpoint)
        if endpoint is None:
            return None

    return {
        "provider": provider,
        "model_name": model_name,
        "endpoint_sha256": endpoint,
    }


def _config_section(settings: dict[str, Any], *path: str) -> dict[str, Any]:
    section: Any = settings
    for key in path:
        if not isinstance(section, dict):
            return {}
        section = section.get(key, {})
    return section if isinstance(section, dict) else {}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_retrieval_runtime_contract(
    settings: dict[str, Any] | None = None,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Build a non-secret contract for every setting that can change retrieval.

    Release evaluations use this object in their checkpoint fingerprints.  It
    prevents a partial run from being resumed after a configuration or
    controlled-recall rule change, even when the dataset and LLM identity are
    unchanged.
    """
    if settings is None:
        from app.utils.config import get_settings

        settings = get_settings()
    if not isinstance(settings, dict):
        raise TypeError("retrieval runtime settings must be a mapping")

    retrieval = _config_section(settings, "rag", "retrieval")
    understanding = _config_section(settings, "rag", "retrieval", "query_understanding")
    reranker = _config_section(settings, "reranker")
    embedding = _config_section(settings, "embedding")
    config_path = str(_config_section(settings, "_meta").get("config_path") or "")
    profile = Path(config_path).stem if config_path else "unknown"

    resolved_artifact_root = artifact_root or Path(__file__).resolve().parents[1] / "retrieval"
    artifact_hashes: dict[str, str] = {}
    for name in _RETRIEVAL_ARTIFACT_NAMES:
        path = resolved_artifact_root / name
        if not path.is_file():
            raise FileNotFoundError(f"retrieval contract artifact is missing: {name}")
        artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    snapshot = {
        "schema_version": RETRIEVAL_RUNTIME_CONTRACT_SCHEMA_VERSION,
        "config_profile": profile,
        "embedding": {
            "model_name": str(embedding.get("model_name") or ""),
            "model_revision": str(embedding.get("model_revision") or ""),
            "device": str(embedding.get("device") or "cpu"),
            "allow_cpu_fallback": bool(embedding.get("allow_cpu_fallback", True)),
            "max_length": (
                int(embedding["max_length"])
                if embedding.get("max_length") is not None
                else None
            ),
            "normalize_embeddings": bool(embedding.get("normalize_embeddings", True)),
            "use_dense": bool(embedding.get("use_dense", True)),
            "use_sparse": bool(embedding.get("use_sparse", False)),
            "use_colbert": bool(embedding.get("use_colbert", False)),
        },
        "reranker": {
            "enabled": bool(reranker.get("enabled", False)),
            "failure_mode": str(reranker.get("failure_mode", "closed")).strip().lower(),
            "model_name": str(reranker.get("model_name") or ""),
            "model_revision": str(reranker.get("model_revision") or ""),
            "device": str(reranker.get("device") or "cpu"),
            "allow_cpu_fallback": bool(reranker.get("allow_cpu_fallback", True)),
            "batch_size": int(reranker.get("batch_size", 8)),
            "max_length": (
                int(reranker["max_length"])
                if reranker.get("max_length") is not None
                else None
            ),
            "top_n": int(reranker.get("top_n", 5)),
            "score_threshold": (
                float(reranker["score_threshold"])
                if reranker.get("score_threshold") is not None
                else None
            ),
        },
        "retrieval": {
            "top_k": int(retrieval.get("top_k", 5)),
            "similarity_threshold": float(retrieval.get("similarity_threshold", 0.05)),
            "enable_rerank": bool(retrieval.get("enable_rerank", True)),
            "rerank_candidate_multiplier": int(
                retrieval.get("rerank_candidate_multiplier", 4)
            ),
            "rerank_min_candidates": int(retrieval.get("rerank_min_candidates", 20)),
            "rerank_max_candidates": int(retrieval.get("rerank_max_candidates", 80)),
            "max_chunks_per_document": int(retrieval.get("max_chunks_per_document", 6)),
            "enable_hybrid": bool(retrieval.get("enable_hybrid", True)),
            "alpha": float(retrieval.get("alpha", 0.5)),
            "enable_rrf": bool(retrieval.get("enable_rrf", True)),
            "rrf_k": int(retrieval.get("rrf_k", 60)),
            "enable_dynamic_topk": bool(retrieval.get("enable_dynamic_topk", True)),
            "dynamic_topk_threshold": float(
                retrieval.get("dynamic_topk_threshold", 0.5)
            ),
            "merge_candidate_multiplier": int(
                retrieval.get("merge_candidate_multiplier", 8)
            ),
            "dynamic_context_selection": bool(
                retrieval.get("dynamic_context_selection", True)
            ),
            "flat_score_top_n": int(retrieval.get("flat_score_top_n", 2)),
            "min_final_rerank_spread": float(
                retrieval.get("min_final_rerank_spread", 0.05)
            ),
            "min_final_rerank_gap": float(
                retrieval.get("min_final_rerank_gap", 0.08)
            ),
            "query_understanding": {
                "enable_coreference": bool(understanding.get("enable_coreference", True)),
                "enable_decomposition": bool(understanding.get("enable_decomposition", True)),
                "enable_rewrite": bool(understanding.get("enable_rewrite", True)),
                "max_subqueries": int(understanding.get("max_subqueries", 3)),
            },
        },
        "artifacts": artifact_hashes,
    }
    return {"sha256": _canonical_sha256(snapshot), "snapshot": snapshot}


def normalize_retrieval_runtime_contract(value: Any) -> dict[str, Any] | None:
    """Validate a self-hashing retrieval contract without accepting extra fields."""
    if not isinstance(value, dict) or set(value) != {"sha256", "snapshot"}:
        return None
    digest = normalize_sha256(value.get("sha256"))
    snapshot = value.get("snapshot")
    if digest is None or not isinstance(snapshot, dict):
        return None
    if snapshot.get("schema_version") != RETRIEVAL_RUNTIME_CONTRACT_SCHEMA_VERSION:
        return None
    try:
        computed = _canonical_sha256(snapshot)
    except (TypeError, ValueError):
        return None
    if computed != digest:
        return None
    return {"sha256": digest, "snapshot": snapshot}
