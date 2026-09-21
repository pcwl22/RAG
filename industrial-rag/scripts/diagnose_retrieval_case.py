"""Inspect production retrieval stages for selected evaluation cases.

The command is intentionally read-only: it runs query understanding and
retrieval, then writes bounded document metadata and scores without document
content or provider credentials.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.citation_scoring import (  # noqa: E402
    expected_rank_map,
    matches_expected_citation,
)
from app.retrieval.legal_concept_map import (  # noqa: E402
    build_concept_article_queries,
    match_legal_concept_articles,
)
from app.retrieval.query_understanding import (  # noqa: E402
    _split_labeled_scenarios,
)
from app.service.enhanced_query_service import (  # noqa: E402
    EnhancedQueryOptions,
    EnhancedQueryService,
)
from app.vectorstore.storage_adapter import (  # noqa: E402
    close_vector_store,
    init_vector_store,
)

DIAGNOSTIC_SCHEMA_VERSION = 1
DIAGNOSTIC_IMPLEMENTATION_VERSION = "8"


class _FailClosedLLM:
    """Surface provider failures that query understanding normally degrades."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.failure_count = 0

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await self.delegate.generate(*args, **kwargs)
        except Exception:
            self.failure_count += 1
            raise RuntimeError("query-understanding provider failed") from None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _document_summary(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = doc.get("metadata") or {}
    return {
        "id": doc.get("id"),
        "semantic_chunk_id": metadata.get("semantic_chunk_id"),
        "law_name": metadata.get("law_name"),
        "article_number": metadata.get("article_number"),
        "legal_citation": metadata.get("legal_citation"),
        "score": _number(doc.get("score")),
        "source_relevance_score": _number(doc.get("source_relevance_score")),
        "source_retrieval_rank": doc.get("source_retrieval_rank"),
        "query_retrieval_ranks": doc.get("query_retrieval_ranks") or {},
        "final_rerank_score": _number(metadata.get("final_rerank_score")),
        "mapped_article_priority": doc.get("mapped_article_priority"),
        "context_selection": doc.get("context_selection"),
        "context_selection_score": _number(doc.get("context_selection_score")),
        "matched_query_count": len(doc.get("matched_queries") or []),
    }


def _load_cases(input_path: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed = {str(case.get("id")): case for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in indexed]
    if missing:
        raise ValueError(f"unknown case ids: {', '.join(missing)}")
    return [indexed[case_id] for case_id in case_ids]


def _checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial.json")


def _diagnostic_fingerprint(
    input_path: Path,
    case_ids: list[str],
    top_k: int,
    *,
    deterministic_understanding: bool,
) -> str:
    context = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "implementation_version": DIAGNOSTIC_IMPLEMENTATION_VERSION,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "case_ids": case_ids,
        "top_k": top_k,
        "deterministic_understanding": deterministic_understanding,
    }
    return hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _deterministic_understanding(query: str) -> dict[str, Any]:
    """Build the production retrieval contract without external model calls."""
    subqueries = _split_labeled_scenarios(query, 3) or [query]
    is_decomposed = len(subqueries) > 1
    mapping_inputs = subqueries if is_decomposed else [query]
    mappings: list[dict[str, Any]] = []
    seen_concepts: set[str] = set()
    for mapping_input in mapping_inputs:
        for mapping in match_legal_concept_articles(mapping_input):
            concept_id = str(mapping.get("concept_id") or "")
            if concept_id and concept_id in seen_concepts:
                continue
            if concept_id:
                seen_concepts.add(concept_id)
            mappings.append(mapping)

    retrieval_queries: list[str] = []
    for value in [query, *subqueries, *build_concept_article_queries(mappings)]:
        normalized = str(value or "").strip()
        if normalized and normalized not in retrieval_queries:
            retrieval_queries.append(normalized)
    return {
        "original_query": query,
        "resolved_query": query,
        "rewritten_query": query,
        "retrieval_signals": {},
        "concept_article_mappings": mappings,
        "subqueries": subqueries,
        "retrieval_queries": retrieval_queries,
        "is_decomposed": is_decomposed,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    case_ids: list[str],
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid diagnostic checkpoint: {path}") from exc
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        raise ValueError("diagnostic checkpoint does not match input or options")
    results = payload.get("cases")
    if not isinstance(results, list) or len(results) > len(case_ids):
        raise ValueError(f"invalid diagnostic checkpoint cases: {path}")
    for index, result in enumerate(results):
        if not isinstance(result, dict) or result.get("id") != case_ids[index]:
            raise ValueError(f"diagnostic checkpoint is not an input prefix: {path}")
    return results


async def diagnose(
    input_path: Path,
    output_path: Path,
    case_ids: list[str],
    *,
    top_k: int,
    resume: bool = False,
    deterministic_understanding: bool = False,
) -> dict[str, Any]:
    cases = _load_cases(input_path, case_ids)
    fingerprint = _diagnostic_fingerprint(
        input_path,
        case_ids,
        top_k,
        deterministic_understanding=deterministic_understanding,
    )
    checkpoint_path = _checkpoint_path(output_path)
    if resume:
        results = _load_checkpoint(
            checkpoint_path, fingerprint=fingerprint, case_ids=case_ids
        )
    else:
        checkpoint_path.unlink(missing_ok=True)
        results = []
    await init_vector_store()
    service = EnhancedQueryService()
    tracked_llm = _FailClosedLLM(service.query_understanding.llm)
    if not deterministic_understanding:
        cast(Any, service.query_understanding).llm = tracked_llm
    try:
        for case in cases[len(results) :]:
            options = EnhancedQueryOptions(
                query=str(case["query"]),
                enable_coreference=False,
                enable_decomposition=True,
                enable_rewrite=True,
                top_k=top_k,
                similarity_threshold=0.05,
                enable_rerank=True,
            )
            if deterministic_understanding:
                understanding = _deterministic_understanding(str(case["query"]))
            else:
                previous_failures = tracked_llm.failure_count
                understanding = await service.understand(options)
                if tracked_llm.failure_count != previous_failures:
                    raise RuntimeError(
                        f"query understanding failed for evaluation case {case['id']}"
                    )
            final_docs, query_to_docs = await service.retrieve(understanding, options)
            expected = [str(value) for value in case.get("expected_citations") or []]
            query_groups = []
            for query, docs in query_to_docs.items():
                ranks = expected_rank_map(case, docs)
                if any(rank is not None for rank in ranks.values()):
                    matching_docs = [
                        _document_summary(doc)
                        for doc in docs
                        if any(
                            matches_expected_citation(case, index, doc)
                            for index, _citation in enumerate(expected)
                        )
                    ]
                else:
                    matching_docs = []
                query_groups.append(
                    {
                        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest()[
                            :16
                        ],
                        "query_length": len(query),
                        "document_count": len(docs),
                        "expected_ranks": ranks,
                        "matching_documents": matching_docs,
                    }
                )
            results.append(
                {
                    "id": case["id"],
                    "category": (case.get("metadata") or {}).get("category"),
                    "domain": (case.get("metadata") or {}).get("domain"),
                    "expected_citations": expected,
                    "retrieval_query_count": len(understanding["retrieval_queries"]),
                    "is_decomposed": bool(understanding.get("is_decomposed")),
                    "concept_mappings": [
                        {
                            "concept_id": mapping.get("concept_id"),
                            "article_ids": [
                                article.get("semantic_chunk_id")
                                for article in mapping.get("articles") or []
                            ],
                        }
                        for mapping in understanding.get("concept_article_mappings") or []
                    ],
                    "mapped_article_coverage": understanding.get(
                        "mapped_article_coverage"
                    ),
                    "final_expected_ranks": expected_rank_map(case, final_docs),
                    "final_documents": [
                        _document_summary(doc) for doc in final_docs
                    ],
                    "query_groups": query_groups,
                }
            )
            _write_json_atomic(
                checkpoint_path,
                {
                    "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                    "fingerprint": fingerprint,
                    "complete": False,
                    "input": str(input_path.resolve()),
                    "top_k": top_k,
                    "deterministic_understanding": deterministic_understanding,
                    "case_count": len(results),
                    "cases": results,
                },
            )
    finally:
        await close_vector_store()

    report = {
        "input": str(input_path.resolve()),
        "top_k": top_k,
        "deterministic_understanding": deterministic_understanding,
        "case_count": len(results),
        "complete": True,
        "cases": results,
    }
    _write_json_atomic(output_path, report)
    checkpoint_path.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--deterministic-understanding",
        action="store_true",
        help="Use only local labeled-scenario splitting and controlled concept mappings",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(
        diagnose(
            args.input,
            args.output,
            args.case_id,
            top_k=args.top_k,
            resume=args.resume,
            deterministic_understanding=args.deterministic_understanding,
        )
    )
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "cases": [
                    {
                        "id": case["id"],
                        "final_expected_ranks": case["final_expected_ranks"],
                    }
                    for case in report["cases"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
