"""Export Industrial RAG query runs into LRAGE-friendly JSONL/YAML artifacts.

Input JSONL format:
  {"id": "case-1", "query": "问题", "expected_answer": "可选参考答案", "rubric": "可选评分规则"}

Example:
  python scripts/export_lrage.py --input cases.jsonl --output-dir exports/lrage
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.retrieval_params import resolve_retrieval_params  # noqa: E402
from app.evaluation.lrage_adapter import (  # noqa: E402
    build_lrage_sample,
    load_cases_jsonl,
    write_lrage_export,
)
from app.service.enhanced_query_service import (  # noqa: E402
    EnhancedQueryOptions,
    EnhancedQueryService,
)
from app.vectorstore.storage_adapter import close_vector_store, init_vector_store  # noqa: E402


def _case_expected_answer(case: dict[str, Any]) -> str:
    return str(
        case.get("expected_answer")
        or case.get("reference_answer")
        or case.get("target")
        or ""
    )


async def _run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    params = resolve_retrieval_params(
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        enable_rerank=args.enable_rerank,
    )
    options = EnhancedQueryOptions(
        query=case["query"],
        chat_history=case.get("chat_history"),
        enable_coreference=not args.disable_coreference,
        enable_decomposition=not args.disable_decomposition,
        enable_rewrite=not args.disable_rewrite,
        top_k=params.top_k,
        similarity_threshold=params.similarity_threshold,
        enable_rerank=params.enable_rerank,
        partition=args.partition,
    )

    response = await EnhancedQueryService().run(options)

    return build_lrage_sample(
        query=case["query"],
        answer=response.answer,
        results=response.results,
        understanding=response.understanding,
        sub_answers=response.sub_answers,
        case_id=case.get("id"),
        expected_answer=_case_expected_answer(case),
        rubric=case.get("rubric"),
        metadata={
            "case_metadata": case.get("metadata", {}),
            "expected_citations": case.get("expected_citations", []),
            "expected_sources": case.get("expected_sources", []),
            "partition": args.partition,
            "top_k": args.top_k,
            "similarity_threshold": args.similarity_threshold,
            "enable_rerank": args.enable_rerank,
        },
        max_context_chars=args.max_context_chars,
        max_document_chars=args.max_document_chars,
    )


async def _run_export(args: argparse.Namespace) -> None:
    await init_vector_store()
    try:
        cases = load_cases_jsonl(args.input)
        if args.limit:
            cases = cases[: args.limit]

        samples: list[dict[str, Any]] = []
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case['query']}")
            samples.append(await _run_case(case, args))

        paths = write_lrage_export(
            output_dir=args.output_dir,
            task_name=args.task_name,
            samples=samples,
            max_score=args.max_score,
        )

        print(
            json.dumps(
                {
                    "dataset_jsonl": str(paths.dataset_jsonl.resolve()),
                    "task_yaml": str(paths.task_yaml.resolve()),
                    "manifest_json": str(paths.manifest_json.resolve()),
                    "sample_count": len(samples),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        await close_vector_store()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL cases")
    parser.add_argument("--output-dir", type=Path, default=Path("exports/lrage"))
    parser.add_argument("--task-name", default="industrial_rag_lrage")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--similarity-threshold", type=float, default=None)
    parser.add_argument("--enable-rerank", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--disable-coreference", action="store_true")
    parser.add_argument("--disable-decomposition", action="store_true")
    parser.add_argument("--disable-rewrite", action="store_true")
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--max-document-chars", type=int, default=16000)
    parser.add_argument("--max-score", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    asyncio.run(_run_export(parse_args()))


if __name__ == "__main__":
    main()
