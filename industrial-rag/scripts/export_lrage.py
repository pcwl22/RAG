"""Export Industrial RAG query runs into LRAGE-friendly JSONL/YAML artifacts.

Input JSONL format:
  {"id": "case-1", "query": "问题", "expected_answer": "可选参考答案", "rubric": "可选评分规则"}

Example:
  python scripts/export_lrage.py --input cases.jsonl --output-dir exports/lrage
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from app.evaluation.retrieval_contract import (  # noqa: E402
    build_retrieval_runtime_contract,
    llm_runtime_identity_from_client,
    normalize_retrieval_runtime_contract,
)
from app.llm.request_policy import build_structured_output_policy  # noqa: E402
from app.service.enhanced_query_service import (  # noqa: E402
    EnhancedQueryOptions,
    EnhancedQueryService,
)
from app.vectorstore.storage_adapter import close_vector_store, init_vector_store  # noqa: E402

CHECKPOINT_SCHEMA_VERSION = 3
EXPORT_IMPLEMENTATION_VERSION = "6"


def _case_expected_answer(case: dict[str, Any]) -> str:
    return str(
        case.get("expected_answer")
        or case.get("reference_answer")
        or case.get("target")
        or ""
    )


def _case_key(case: dict[str, Any]) -> tuple[str, str]:
    return (str(case.get("id") or ""), str(case["query"]))


def _export_fingerprint(
    args: argparse.Namespace,
    *,
    source_dataset_sha256: str,
    llm_runtime_identity: dict[str, Any],
    retrieval_runtime_contract: dict[str, Any],
    answer_generation_policy: dict[str, Any],
    sample_count: int,
) -> tuple[str, dict[str, Any]]:
    normalized_retrieval_contract = normalize_retrieval_runtime_contract(
        retrieval_runtime_contract
    )
    if normalized_retrieval_contract is None:
        raise ValueError("LRAGE export requires a valid retrieval runtime contract")
    context = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "export_implementation_version": EXPORT_IMPLEMENTATION_VERSION,
        "source_dataset_sha256": source_dataset_sha256,
        "llm_runtime_identity": llm_runtime_identity,
        "retrieval_runtime_contract": normalized_retrieval_contract,
        "answer_generation_policy": answer_generation_policy,
        "sample_count": sample_count,
        "limit": args.limit,
        "top_k": args.top_k,
        "similarity_threshold": args.similarity_threshold,
        "enable_rerank": args.enable_rerank,
        "partition": args.partition,
        "disable_coreference": args.disable_coreference,
        "disable_decomposition": args.disable_decomposition,
        "disable_rewrite": args.disable_rewrite,
        "max_context_chars": args.max_context_chars,
        "max_document_chars": args.max_document_chars,
    }
    encoded = json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), context


def _load_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid export checkpoint: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported export checkpoint schema: {path}")
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("export checkpoint does not match the input or run configuration")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) > len(cases):
        raise ValueError(f"invalid export checkpoint records: {path}")
    if payload.get("completed_count") != len(records):
        raise ValueError(f"invalid export checkpoint completed_count: {path}")
    if any(not isinstance(record, dict) or "query" not in record for record in records):
        raise ValueError(f"invalid export checkpoint records: {path}")
    actual_keys = [
        (str(record.get("case_id") or record.get("id") or ""), str(record["query"]))
        for record in records
    ]
    expected_keys = [_case_key(case) for case in cases[: len(records)]]
    if actual_keys != expected_keys:
        raise ValueError("export checkpoint records are not an input prefix")
    return records


def _write_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    context: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "context": context,
        "completed_count": len(records),
        "records": records,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        temporary_path.write_text(serialized, encoding="utf-8", newline="\n")
        temporary_path.replace(path)
    except PermissionError:
        # Some managed workspaces allow updating an existing artifact but do
        # not allow creating a sibling file. Preserve resumability there while
        # retaining atomic replacement on ordinary filesystems.
        path.write_text(serialized, encoding="utf-8", newline="\n")
        temporary_path.unlink(missing_ok=True)


def _is_retryable_case_error(exc: BaseException) -> bool:
    """Retry only provider/network failures during a resumable export."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (
        status_code in {408, 409, 429} or status_code >= 500
    ):
        return True
    return exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "TimeoutError",
    }


async def _run_case(
    case: dict[str, Any],
    args: argparse.Namespace,
    service: EnhancedQueryService,
) -> dict[str, Any]:
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

    response = await service.run(options)

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


async def _run_case_with_retries(
    case: dict[str, Any],
    args: argparse.Namespace,
    service: EnhancedQueryService,
) -> dict[str, Any]:
    """Retry transient provider failures without exposing request contents."""
    for attempt in range(args.case_retries + 1):
        try:
            return await _run_case(case, args, service)
        except Exception as exc:
            if attempt >= args.case_retries or not _is_retryable_case_error(exc):
                raise
            print(
                f"retrying case={case.get('id') or 'unknown'} "
                f"attempt={attempt + 1}/{args.case_retries} "
                f"error={exc.__class__.__name__}"
            )
            await asyncio.sleep(args.case_retry_backoff_seconds * (2**attempt))
    raise RuntimeError("unreachable")


async def _run_export(args: argparse.Namespace) -> None:
    await init_vector_store()
    try:
        cases = load_cases_jsonl(args.input)
        if args.limit:
            cases = cases[: args.limit]

        service = EnhancedQueryService()
        llm_runtime_identity = llm_runtime_identity_from_client(
            service.query_understanding.llm
        )
        answer_llm = service.generator.llm
        answer_generation_policy = build_structured_output_policy(
            provider=str(answer_llm.provider),
            model_name=str(answer_llm.config.get("model_name") or ""),
            base_url=str(answer_llm.config.get("base_url") or ""),
        )
        retrieval_runtime_contract = build_retrieval_runtime_contract()
        source_dataset_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
        fingerprint, checkpoint_context = _export_fingerprint(
            args,
            source_dataset_sha256=source_dataset_sha256,
            llm_runtime_identity=llm_runtime_identity,
            retrieval_runtime_contract=retrieval_runtime_contract,
            answer_generation_policy=answer_generation_policy,
            sample_count=len(cases),
        )
        checkpoint_path = args.output_dir / f"{args.task_name}.partial.json"
        completed = (
            _load_checkpoint(checkpoint_path, fingerprint=fingerprint, cases=cases)
            if args.resume
            else []
        )
        if not args.resume:
            checkpoint_path.unlink(missing_ok=True)
        completed_by_key = {
            (str(sample.get("id") or ""), str(sample["query"])): sample for sample in completed
        }
        samples: list[dict[str, Any]] = []
        for index, case in enumerate(cases, 1):
            key = _case_key(case)
            if key in completed_by_key:
                print(f"[{index}/{len(cases)}] resumed case={case.get('id') or index}")
                samples.append(completed_by_key.pop(key))
                continue

            print(f"[{index}/{len(cases)}] case={case.get('id') or index}")
            sample = await _run_case_with_retries(case, args, service)
            sample.setdefault("metadata", {})["evaluation_contract"] = {
                "source_dataset_sha256": source_dataset_sha256,
                "llm_runtime_identity": llm_runtime_identity,
                "retrieval_runtime_contract": retrieval_runtime_contract,
                "answer_generation_policy": answer_generation_policy,
            }
            samples.append(sample)
            _write_checkpoint(
                checkpoint_path,
                fingerprint=fingerprint,
                context=checkpoint_context,
                records=samples,
            )

        paths = write_lrage_export(
            output_dir=args.output_dir,
            task_name=args.task_name,
            samples=samples,
            max_score=args.max_score,
            evaluation_contract={
                "source_dataset_sha256": source_dataset_sha256,
                "llm_runtime_identity": llm_runtime_identity,
                "retrieval_runtime_contract": retrieval_runtime_contract,
                "answer_generation_policy": answer_generation_policy,
                "checkpoint_fingerprint": fingerprint,
            },
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
        checkpoint_path.unlink(missing_ok=True)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed cases from <task-name>.partial.json in the output directory",
    )
    parser.add_argument(
        "--case-retries",
        type=int,
        default=2,
        help="Retries for transient provider failures per case (default: 2)",
    )
    parser.add_argument(
        "--case-retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Initial delay between transient case retries (default: 2 seconds)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.case_retries <= 3:
        raise SystemExit("--case-retries must be between 0 and 3")
    if not 0 <= args.case_retry_backoff_seconds <= 120:
        raise SystemExit("--case-retry-backoff-seconds must be between 0 and 120")
    asyncio.run(_run_export(args))


if __name__ == "__main__":
    main()
