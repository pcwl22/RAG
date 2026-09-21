"""Run a stratified, local retrieval evaluation without an LLM judge."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.citation_scoring import expected_ranks  # noqa: E402
from app.evaluation.retrieval_contract import (  # noqa: E402
    build_retrieval_runtime_contract,
    llm_runtime_identity_from_client,
    normalize_retrieval_runtime_contract,
    normalize_sha256,
)
from app.retrieval.factory import build_retrieval_engine  # noqa: E402
from app.retrieval.result_merge import merge_retrieval_results  # noqa: E402
from app.vectorstore.storage_adapter import (  # noqa: E402
    close_vector_store,
    init_vector_store,
)
from scripts.validate_evaluation_assets import citation_leak_ids  # noqa: E402

CHECKPOINT_SCHEMA_VERSION = 2
EVALUATION_IMPLEMENTATION_VERSION = "14"


class _EvaluationLLMFailureTracker:
    """Expose swallowed production-understanding failures to the release gate."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.failure_count = 0

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await self.delegate.generate(*args, **kwargs)
        except Exception:
            self.failure_count += 1
            # QueryUnderstanding intentionally logs and falls back on provider
            # failures.  Replace the provider exception before it reaches that
            # log boundary so response bodies, URLs, or credentials cannot be
            # copied into release logs.
            raise RuntimeError("evaluation LLM request failed") from None


def _checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial.json")


def _llm_runtime_identity(llm: Any) -> dict[str, Any]:
    """Return the non-secret LLM identity that can affect retrieval results."""
    return llm_runtime_identity_from_client(llm)


def _require_expected_llm_identity(
    identity: dict[str, Any] | None,
    *,
    expected_provider: str | None,
    expected_model: str | None,
    expected_endpoint_sha256: str | None,
) -> None:
    """Fail before the first request when operator-selected LLM identity drifted."""
    if not any((expected_provider, expected_model, expected_endpoint_sha256)):
        return
    if identity is None:
        raise ValueError("expected LLM identity requires an LLM-backed evaluation")

    if expected_provider is not None:
        provider = expected_provider.strip().lower()
        if not provider or identity.get("provider") != provider:
            raise ValueError("configured LLM provider does not match the expected identity")
    if expected_model is not None:
        model = expected_model.strip()
        if not model or identity.get("model_name") != model:
            raise ValueError("configured LLM model does not match the expected identity")
    if expected_endpoint_sha256 is not None:
        digest = normalize_sha256(
            expected_endpoint_sha256.strip().lower().removeprefix("sha256:")
        )
        if digest is None:
            raise ValueError("expected LLM endpoint SHA-256 must contain 64 hex characters")
        if identity.get("endpoint_sha256") != digest:
            raise ValueError("configured LLM endpoint does not match the expected identity")


def _maximum_allowed_misses(
    total_expected_citations: int,
    minimum_citation_recall: float,
) -> int:
    required_hits = math.ceil(minimum_citation_recall * total_expected_citations - 1e-12)
    return total_expected_citations - required_hits


def _require_minimum_citation_recall_attainable(
    rows: list[dict[str, Any]],
    *,
    total_expected_citations: int,
    minimum_citation_recall: float | None,
) -> None:
    if minimum_citation_recall is None or total_expected_citations <= 0:
        return
    misses = sum(
        int(row.get("expected_citations", 0)) - int(row.get("citation_hits", 0)) for row in rows
    )
    maximum_misses = _maximum_allowed_misses(
        total_expected_citations,
        minimum_citation_recall,
    )
    if misses > maximum_misses:
        raise RuntimeError(
            "minimum citation recall is no longer attainable: "
            f"misses={misses}, maximum_misses={maximum_misses}, "
            f"minimum={minimum_citation_recall:.6f}"
        )


def _evaluation_fingerprint(
    input_path: Path,
    *,
    sample_count: int,
    top_k: int,
    limit: int,
    enable_rerank: bool | None,
    enable_rrf: bool | None,
    enable_dynamic_topk: bool | None,
    similarity_threshold: float | None,
    enable_query_rewrite: bool,
    rewrite_concurrency: int,
    use_production_pipeline: bool,
    production_decomposition: bool,
    understanding_concurrency: int,
    llm_runtime_identity: dict[str, Any] | None,
    retrieval_runtime_contract: dict[str, Any],
    minimum_citation_recall: float | None,
) -> tuple[str, dict[str, Any]]:
    uses_llm = use_production_pipeline or (enable_query_rewrite and sample_count > 0)
    if uses_llm and not llm_runtime_identity:
        raise ValueError("LLM-backed evaluation requires a runtime identity")
    normalized_retrieval_contract = normalize_retrieval_runtime_contract(
        retrieval_runtime_contract
    )
    if normalized_retrieval_contract is None:
        raise ValueError("retrieval evaluation requires a valid runtime contract")
    context = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "evaluation_implementation_version": EVALUATION_IMPLEMENTATION_VERSION,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "sample_count": sample_count,
        "top_k": top_k,
        "limit": limit,
        "enable_rerank": enable_rerank,
        "enable_rrf": enable_rrf,
        "enable_dynamic_topk": enable_dynamic_topk,
        "similarity_threshold": similarity_threshold,
        "enable_query_rewrite": enable_query_rewrite,
        "rewrite_concurrency": rewrite_concurrency,
        "use_production_pipeline": use_production_pipeline,
        "production_decomposition": production_decomposition,
        "understanding_concurrency": (
            understanding_concurrency if use_production_pipeline else None
        ),
        "llm_runtime_identity": llm_runtime_identity if uses_llm else None,
        "retrieval_runtime_contract": normalized_retrieval_contract,
        "minimum_citation_recall": minimum_citation_recall,
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
        raise ValueError(f"invalid retrieval evaluation checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid retrieval evaluation checkpoint: {path}")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported retrieval evaluation checkpoint schema: {path}")
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(
            "retrieval evaluation checkpoint does not match the input or run configuration"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) > len(cases):
        raise ValueError(f"invalid retrieval evaluation checkpoint rows: {path}")
    for index, row in enumerate(rows):
        case = cases[index]
        metadata = case.get("metadata") or {}
        if not isinstance(row, dict) or row.get("id") != case.get("id"):
            raise ValueError(f"retrieval evaluation checkpoint is not an input prefix: {path}")
        if row.get("category") != metadata.get("category") or row.get("domain") != metadata.get(
            "domain"
        ):
            raise ValueError(f"retrieval evaluation checkpoint metadata mismatch: {path}")
    return rows


def _write_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    context: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "context": context,
        "completed_count": len(rows),
        "rows": rows,
    }
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _require_no_new_llm_failures(
    tracker: _EvaluationLLMFailureTracker,
    previous_count: int,
    case_id: str,
) -> None:
    if tracker.failure_count != previous_count:
        raise RuntimeError(
            f"production query-understanding LLM failed for evaluation case {case_id}"
        )


def _score_case(case: dict, docs: list[dict], top_k: int) -> dict[str, Any]:
    expected = case.get("expected_citations") or []
    if not expected:
        return {"abstained": not docs, "retrieved": len(docs)}
    ranks = expected_ranks(case, docs, top_k)
    return {
        "expected_citations": len(expected),
        "citation_hits": sum(rank is not None for rank in ranks),
        "reciprocal_rank_sum": sum(1.0 / rank for rank in ranks if rank),
        "ranks": ranks,
        "retrieved": len(docs),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    citations = sum(row.get("expected_citations", 0) for row in rows)
    if citations:
        return {
            "cases": len(rows),
            "expected_citations": citations,
            "citation_recall": sum(row.get("citation_hits", 0) for row in rows) / citations,
            "citation_mrr": sum(row.get("reciprocal_rank_sum", 0.0) for row in rows) / citations,
            "zero_hit_cases": sum(not any(row.get("ranks", [])) for row in rows),
        }
    return {
        "cases": len(rows),
        "abstention_rate": sum(bool(row.get("abstained")) for row in rows) / len(rows)
        if rows
        else 0.0,
        "false_positive_cases": sum(not bool(row.get("abstained")) for row in rows),
    }


async def evaluate(
    input_path: Path,
    output_path: Path,
    top_k: int,
    *,
    limit: int = 0,
    enable_rerank: bool | None = None,
    enable_rrf: bool | None = None,
    enable_dynamic_topk: bool | None = None,
    similarity_threshold: float | None = None,
    enable_query_rewrite: bool = False,
    rewrite_concurrency: int = 4,
    use_production_pipeline: bool = False,
    production_decomposition: bool = True,
    understanding_concurrency: int = 4,
    minimum_citation_recall: float | None = None,
    expected_llm_provider: str | None = None,
    expected_llm_model: str | None = None,
    expected_llm_endpoint_sha256: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    cases = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit:
        cases = cases[:limit]
    leaked = citation_leak_ids(cases)
    if leaked:
        raise ValueError(
            f"{input_path}: citation leakage detected in {len(leaked)} cases; "
            f"examples: {', '.join(leaked[:5])}"
        )
    if minimum_citation_recall is not None and not 0.0 <= minimum_citation_recall <= 1.0:
        raise ValueError("minimum_citation_recall must be between 0 and 1")

    production_service: Any = None
    production_llm_tracker: _EvaluationLLMFailureTracker | None = None
    rewrite_understanding: Any = None
    llm_identity: dict[str, Any] | None = None
    if use_production_pipeline:
        if understanding_concurrency < 1:
            raise ValueError("understanding_concurrency must be at least 1")
        from app.service.enhanced_query_service import EnhancedQueryService

        production_service = EnhancedQueryService()
        llm_identity = _llm_runtime_identity(production_service.query_understanding.llm)
    elif enable_query_rewrite and cases:
        from app.retrieval.query_understanding import QueryUnderstanding

        if rewrite_concurrency < 1:
            raise ValueError("rewrite_concurrency must be at least 1")
        rewrite_understanding = QueryUnderstanding()
        llm_identity = _llm_runtime_identity(rewrite_understanding.llm)

    _require_expected_llm_identity(
        llm_identity,
        expected_provider=expected_llm_provider,
        expected_model=expected_llm_model,
        expected_endpoint_sha256=expected_llm_endpoint_sha256,
    )
    retrieval_runtime_contract = build_retrieval_runtime_contract()

    fingerprint, checkpoint_context = _evaluation_fingerprint(
        input_path,
        sample_count=len(cases),
        top_k=top_k,
        limit=limit,
        enable_rerank=enable_rerank,
        enable_rrf=enable_rrf,
        enable_dynamic_topk=enable_dynamic_topk,
        similarity_threshold=similarity_threshold,
        enable_query_rewrite=enable_query_rewrite,
        rewrite_concurrency=rewrite_concurrency,
        use_production_pipeline=use_production_pipeline,
        production_decomposition=production_decomposition,
        understanding_concurrency=understanding_concurrency,
        llm_runtime_identity=llm_identity,
        retrieval_runtime_contract=retrieval_runtime_contract,
        minimum_citation_recall=minimum_citation_recall,
    )
    checkpoint_path = _checkpoint_path(output_path)
    if resume:
        rows = _load_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            cases=cases,
        )
    else:
        checkpoint_path.unlink(missing_ok=True)
        rows = []
    total_expected_citations = sum(len(case.get("expected_citations") or []) for case in cases)
    _require_minimum_citation_recall_attainable(
        rows,
        total_expected_citations=total_expected_citations,
        minimum_citation_recall=minimum_citation_recall,
    )
    query_variants: dict[str, list[str]] = {str(case["id"]): [case["query"]] for case in cases}
    rewrite_count = 0
    if production_service is not None:
        production_llm_tracker = _EvaluationLLMFailureTracker(
            production_service.query_understanding.llm
        )
        production_service.query_understanding.llm = production_llm_tracker
    elif rewrite_understanding is not None:
        semaphore = asyncio.Semaphore(rewrite_concurrency)

        async def rewrite_case(case: dict[str, Any]) -> tuple[str, str | None]:
            async with semaphore:
                rewritten = (await rewrite_understanding.rewrite_query(str(case["query"]))).strip()
            if not rewritten or rewritten == case["query"]:
                return str(case["id"]), None
            # A rewrite is a recall aid, never a way to smuggle the answer into
            # the query.  Discard a model output that names the expected source
            # citation or statute, and keep the original user question.
            rewrite_case_row = {
                "id": case["id"],
                "query": rewritten,
                "expected_citations": case.get("expected_citations") or [],
            }
            if citation_leak_ids([rewrite_case_row]):
                return str(case["id"]), None
            if any(
                str(source).strip() and str(source).strip() in rewritten
                for source in case.get("expected_sources") or []
            ):
                return str(case["id"]), None
            return str(case["id"]), rewritten

        rewritten_cases = await asyncio.gather(*(rewrite_case(case) for case in cases))
        for case_id, rewritten in rewritten_cases:
            if rewritten:
                query_variants[case_id].append(rewritten)
                rewrite_count += 1

    await init_vector_store()
    # Use the same factory as the API.  Hard-coding HybridRetrievalEngine here
    # made the release metric depend on a different retrieval mode than the
    # configured production service.
    engine = build_retrieval_engine()

    def record_result(index: int, case: dict[str, Any], docs: list[dict[str, Any]]) -> None:
        score = _score_case(case, docs, top_k)
        rows.append({"id": case["id"], **case["metadata"], **score})
        _write_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            context=checkpoint_context,
            rows=rows,
        )
        _require_minimum_citation_recall_attainable(
            rows,
            total_expected_citations=total_expected_citations,
            minimum_citation_recall=minimum_citation_recall,
        )
        print(f"[{index}/{len(cases)}] {case['id']}: {score}")

    try:
        if production_service is not None:
            from app.service.enhanced_query_service import EnhancedQueryOptions

            assert production_llm_tracker is not None
            remaining_cases = cases[len(rows) :]
            for batch_start in range(
                0,
                len(remaining_cases),
                understanding_concurrency,
            ):
                batch_cases = remaining_cases[batch_start : batch_start + understanding_concurrency]
                batch_options = [
                    EnhancedQueryOptions(
                        query=str(case["query"]),
                        enable_coreference=False,
                        enable_decomposition=production_decomposition,
                        enable_rewrite=True,
                        top_k=top_k,
                        similarity_threshold=(
                            similarity_threshold if similarity_threshold is not None else 0.05
                        ),
                        enable_rerank=(enable_rerank if enable_rerank is not None else True),
                    )
                    for case in batch_cases
                ]
                previous_failure_count = production_llm_tracker.failure_count
                understanding_results = await asyncio.gather(
                    *(production_service.understand(options) for options in batch_options)
                )
                _require_no_new_llm_failures(
                    production_llm_tracker,
                    previous_failure_count,
                    str(batch_cases[0]["id"]),
                )
                for case, options, understanding_result in zip(
                    batch_cases,
                    batch_options,
                    understanding_results,
                    strict=True,
                ):
                    docs, _query_to_docs = await production_service.retrieve(
                        understanding_result,
                        options,
                    )
                    record_result(len(rows) + 1, case, docs)
        else:
            for index, case in enumerate(cases[len(rows) :], len(rows) + 1):
                retrieve_kwargs: dict[str, Any] = {"top_k": top_k}
                if enable_rerank is not None:
                    retrieve_kwargs["enable_rerank"] = enable_rerank
                if similarity_threshold is not None:
                    retrieve_kwargs["similarity_threshold"] = similarity_threshold
                if enable_rrf is not None:
                    retrieve_kwargs["enable_rrf"] = enable_rrf
                if enable_dynamic_topk is not None:
                    retrieve_kwargs["enable_dynamic_topk"] = enable_dynamic_topk
                variants = query_variants[str(case["id"])]
                result_groups = [
                    await engine.retrieve(query, **retrieve_kwargs) for query in variants
                ]
                docs = merge_retrieval_results(result_groups, top_k, variants)
                record_result(index, case, docs)
    finally:
        await close_vector_store()

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"category:{row['category']}"].append(row)
        groups[f"domain:{row['domain']}"].append(row)
    in_domain = [row for row in rows if row["category"] != "no_answer"]
    no_answer = [row for row in rows if row["category"] == "no_answer"]
    report = {
        "input": str(input_path.resolve()),
        "sample_count": len(rows),
        "top_k": top_k,
        "retrieval_engine": type(engine).__name__,
        "evaluation_contract": {
            "input_sha256": checkpoint_context["input_sha256"],
            "citation_leakage_checked": True,
            "enable_rerank": enable_rerank,
            "enable_rrf": enable_rrf,
            "enable_dynamic_topk": enable_dynamic_topk,
            "similarity_threshold": similarity_threshold,
            "enable_query_rewrite": enable_query_rewrite,
            "rewrite_concurrency": rewrite_concurrency if enable_query_rewrite else None,
            "rewritten_case_count": rewrite_count,
            "use_production_pipeline": use_production_pipeline,
            "production_decomposition": production_decomposition
            if use_production_pipeline
            else None,
            "fail_closed_query_understanding": use_production_pipeline,
            "understanding_concurrency": (
                understanding_concurrency if use_production_pipeline else None
            ),
            "llm_runtime_identity": llm_identity,
            "retrieval_runtime_contract": retrieval_runtime_contract,
            "minimum_citation_recall": minimum_citation_recall,
            "checkpoint_fingerprint": fingerprint,
        },
        "in_domain": _aggregate(in_domain),
        "no_answer": _aggregate(no_answer),
        "groups": {name: _aggregate(group) for name, group in sorted(groups.items())},
        "samples": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--enable-rerank", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-rrf", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--enable-dynamic-topk", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--similarity-threshold", type=float, default=None)
    parser.add_argument(
        "--enable-query-rewrite", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--rewrite-concurrency", type=int, default=4)
    parser.add_argument(
        "--use-production-pipeline", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--production-decomposition", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--understanding-concurrency",
        type=int,
        default=4,
        help="Maximum concurrent production query-understanding cases",
    )
    parser.add_argument(
        "--minimum-citation-recall",
        type=float,
        default=None,
        help="Fail once completed misses make this full-suite recall unattainable",
    )
    parser.add_argument(
        "--expected-llm-provider",
        help="fail before the first request unless the configured provider matches",
    )
    parser.add_argument(
        "--expected-llm-model",
        help="fail before the first request unless the configured model matches",
    )
    parser.add_argument(
        "--expected-llm-endpoint-sha256",
        help="fail before the first request unless the non-secret endpoint digest matches",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an input/configuration-bound atomic checkpoint for this output",
    )
    args = parser.parse_args()
    report = asyncio.run(
        evaluate(
            args.input,
            args.output,
            args.top_k,
            limit=args.limit,
            enable_rerank=args.enable_rerank,
            enable_rrf=args.enable_rrf,
            enable_dynamic_topk=args.enable_dynamic_topk,
            similarity_threshold=args.similarity_threshold,
            enable_query_rewrite=args.enable_query_rewrite,
            rewrite_concurrency=args.rewrite_concurrency,
            use_production_pipeline=args.use_production_pipeline,
            production_decomposition=args.production_decomposition,
            understanding_concurrency=args.understanding_concurrency,
            minimum_citation_recall=args.minimum_citation_recall,
            expected_llm_provider=args.expected_llm_provider,
            expected_llm_model=args.expected_llm_model,
            expected_llm_endpoint_sha256=args.expected_llm_endpoint_sha256,
            resume=args.resume,
        )
    )
    print(
        json.dumps(
            {key: report[key] for key in ("sample_count", "in_domain", "no_answer", "groups")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
