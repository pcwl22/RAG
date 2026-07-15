"""Utilities for merging multi-query retrieval results."""
from __future__ import annotations

from typing import Any


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def result_score(doc: dict[str, Any]) -> float:
    """Return the score that should drive final merged ordering."""
    metadata = doc.get("metadata") or {}
    for value in (
        metadata.get("rerank_prob"),
        doc.get("rerank_prob"),
        doc.get("score"),
        doc.get("rrf_score"),
    ):
        score = _coerce_score(value)
        if score is not None:
            return score
    return 0.0


def _doc_identity_values(doc: dict[str, Any]) -> set[str]:
    metadata = doc.get("metadata") or {}
    values = [
        doc.get("id"),
        metadata.get("semantic_chunk_id"),
        metadata.get("chunk_id"),
    ]
    return {str(value) for value in values if value}


def _priority_value(doc: dict[str, Any], priority_ids: list[str]) -> int:
    if not priority_ids:
        return 0

    identities = _doc_identity_values(doc)
    for index, priority_id in enumerate(priority_ids):
        if priority_id in identities:
            return len(priority_ids) - index
    return 0


def merge_retrieval_results(
    result_groups: list[list[dict[str, Any]]],
    top_k: int,
    queries: list[str] | None = None,
    priority_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Merge multi-query retrieval results by chunk identity and best score."""
    chunks: dict[str, dict[str, Any]] = {}
    priority_ids = list(dict.fromkeys(priority_ids or []))

    for group_index, group in enumerate(result_groups):
        query = queries[group_index] if queries and group_index < len(queries) else None
        for rank, doc in enumerate(group):
            doc_id = str(doc.get("id") or "")
            if not doc_id:
                continue

            score = result_score(doc)
            if doc_id not in chunks:
                chunks[doc_id] = {
                    "doc": doc,
                    "best_score": score,
                    "best_rank": rank,
                    "appearances": 1,
                    "matched_queries": [query] if query else [],
                    "priority": _priority_value(doc, priority_ids),
                }
                continue

            item = chunks[doc_id]
            item["appearances"] += 1
            if query and query not in item["matched_queries"]:
                item["matched_queries"].append(query)
            item["priority"] = max(item["priority"], _priority_value(doc, priority_ids))
            if (score, -rank) > (item["best_score"], -item["best_rank"]):
                item["doc"] = doc
                item["best_score"] = score
                item["best_rank"] = rank

    ranked = sorted(
        chunks.values(),
        key=lambda item: (
            item["priority"],
            item["best_score"],
            item["appearances"],
            -item["best_rank"],
        ),
        reverse=True,
    )

    merged: list[dict[str, Any]] = []
    for item in ranked[:top_k]:
        doc = item["doc"].copy()
        doc["score"] = item["best_score"]
        doc["merge_score"] = item["best_score"]
        doc["multi_query_appearances"] = item["appearances"]
        doc["matched_queries"] = item["matched_queries"]
        if item["priority"]:
            doc["mapped_article_priority"] = item["priority"]
        merged.append(doc)

    return merged
