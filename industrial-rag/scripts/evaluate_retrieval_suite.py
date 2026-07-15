"""Run a stratified, local retrieval evaluation without an LLM judge."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.hybrid import HybridRetrievalEngine  # noqa: E402
from app.vectorstore.storage_adapter import (  # noqa: E402
    close_vector_store,
    init_vector_store,
)


def _doc_text(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    return " ".join(str(value) for value in (
        doc.get("id"), metadata.get("law_name"), metadata.get("article_number"),
        metadata.get("legal_citation"), doc.get("content"),
    ) if value)


def _score_case(case: dict, docs: list[dict], top_k: int) -> dict[str, Any]:
    expected = case.get("expected_citations") or []
    if not expected:
        return {"abstained": not docs, "retrieved": len(docs)}
    ranks = []
    for citation in expected:
        rank = next((i for i, doc in enumerate(docs[:top_k], 1) if citation in _doc_text(doc)), None)
        ranks.append(rank)
    return {
        "expected_citations": len(expected), "citation_hits": sum(rank is not None for rank in ranks),
        "reciprocal_rank_sum": sum(1.0 / rank for rank in ranks if rank), "ranks": ranks,
        "retrieved": len(docs),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    citations = sum(row.get("expected_citations", 0) for row in rows)
    if citations:
        return {
            "cases": len(rows), "expected_citations": citations,
            "citation_recall": sum(row.get("citation_hits", 0) for row in rows) / citations,
            "citation_mrr": sum(row.get("reciprocal_rank_sum", 0.0) for row in rows) / citations,
            "zero_hit_cases": sum(not any(row.get("ranks", [])) for row in rows),
        }
    return {
        "cases": len(rows),
        "abstention_rate": sum(bool(row.get("abstained")) for row in rows) / len(rows) if rows else 0.0,
        "false_positive_cases": sum(not bool(row.get("abstained")) for row in rows),
    }


async def evaluate(input_path: Path, output_path: Path, top_k: int) -> dict[str, Any]:
    cases = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    await init_vector_store()
    engine = HybridRetrievalEngine()
    rows = []
    try:
        for index, case in enumerate(cases, 1):
            docs = await engine.retrieve(case["query"], top_k=top_k, enable_rerank=True)
            score = _score_case(case, docs, top_k)
            rows.append({"id": case["id"], **case["metadata"], **score})
            print(f"[{index}/{len(cases)}] {case['id']}: {score}")
    finally:
        await close_vector_store()

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"category:{row['category']}"] .append(row)
        groups[f"domain:{row['domain']}"] .append(row)
    in_domain = [row for row in rows if row["category"] != "no_answer"]
    no_answer = [row for row in rows if row["category"] == "no_answer"]
    report = {
        "input": str(input_path.resolve()), "sample_count": len(rows), "top_k": top_k,
        "in_domain": _aggregate(in_domain), "no_answer": _aggregate(no_answer),
        "groups": {name: _aggregate(group) for name, group in sorted(groups.items())},
        "samples": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.input, args.output, args.top_k))
    print(json.dumps({key: report[key] for key in ("sample_count", "in_domain", "no_answer", "groups")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
