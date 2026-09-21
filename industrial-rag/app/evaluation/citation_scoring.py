"""Shared exact citation matching for release evaluation and diagnostics."""
from __future__ import annotations

from typing import Any


def document_identity_values(document: dict[str, Any]) -> set[str]:
    metadata = document.get("metadata") or {}
    return {
        str(value)
        for value in (
            document.get("id"),
            metadata.get("semantic_chunk_id"),
            metadata.get("chunk_id"),
        )
        if value
    }


def document_search_text(document: dict[str, Any]) -> str:
    metadata = document.get("metadata") or {}
    return " ".join(
        str(value)
        for value in (
            document.get("id"),
            metadata.get("law_name"),
            metadata.get("article_number"),
            metadata.get("legal_citation"),
            document.get("content"),
        )
        if value
    )


def matches_expected_citation(
    case: dict[str, Any], expected_index: int, document: dict[str, Any]
) -> bool:
    citations = [str(value) for value in case.get("expected_citations") or []]
    if expected_index < 0 or expected_index >= len(citations):
        raise IndexError("expected citation index out of range")

    metadata = case.get("metadata") or {}
    article_ids = [str(value).strip() for value in metadata.get("article_ids") or []]
    expected_id = article_ids[expected_index] if expected_index < len(article_ids) else ""
    if expected_id:
        return expected_id in document_identity_values(document)

    text = document_search_text(document)
    if citations[expected_index] not in text:
        return False
    sources = [
        str(value).strip()
        for value in case.get("expected_sources") or []
        if str(value).strip()
    ]
    if not sources:
        return True
    if len(sources) == len(citations):
        return sources[expected_index] in text
    return any(source in text for source in sources)


def expected_ranks(
    case: dict[str, Any], documents: list[dict[str, Any]], top_k: int | None = None
) -> list[int | None]:
    limit = len(documents) if top_k is None else max(0, top_k)
    candidates = documents[:limit]
    return [
        next(
            (
                rank
                for rank, document in enumerate(candidates, 1)
                if matches_expected_citation(case, expected_index, document)
            ),
            None,
        )
        for expected_index, _citation in enumerate(case.get("expected_citations") or [])
    ]


def expected_rank_map(
    case: dict[str, Any], documents: list[dict[str, Any]], top_k: int | None = None
) -> dict[str, int | None]:
    citations = [str(value) for value in case.get("expected_citations") or []]
    article_ids = [
        str(value).strip()
        for value in (case.get("metadata") or {}).get("article_ids") or []
    ]
    ranks = expected_ranks(case, documents, top_k)
    return {
        (
            article_ids[index]
            if index < len(article_ids) and article_ids[index]
            else citation
        ): rank
        for index, (citation, rank) in enumerate(zip(citations, ranks, strict=True))
    }
