"""Check deterministic citation quality on the locked 150-case holdout suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.retrieval_contract import (  # noqa: E402
    normalize_llm_runtime_identity,
    normalize_retrieval_runtime_contract,
    normalize_sha256,
)

EXPECTED_SAMPLE_COUNT = 150
DEFAULT_MINIMUMS = {
    "citation_recall": 0.95,
    "citation_mrr": 0.70,
}


def build_holdout_gate(
    retrieval: dict[str, Any],
    *,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
    minimums: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed holdout report without changing the source metrics."""
    limits = {**DEFAULT_MINIMUMS, **(minimums or {})}
    checks: dict[str, dict[str, Any]] = {}
    sample_count = int(retrieval.get("sample_count") or 0)
    checks["sample_count"] = {
        "value": sample_count,
        "minimum": expected_sample_count,
        "status": "passed" if sample_count == expected_sample_count else "failed",
    }

    in_domain = retrieval.get("in_domain") or {}
    for name, minimum in limits.items():
        value = in_domain.get(name)
        status = "passed" if value is not None and float(value) >= minimum else "failed"
        checks[name] = {"value": value, "minimum": minimum, "status": status}

    contract = retrieval.get("evaluation_contract") or {}
    runtime_identity = normalize_llm_runtime_identity(contract.get("llm_runtime_identity"))
    retrieval_runtime_contract = normalize_retrieval_runtime_contract(
        contract.get("retrieval_runtime_contract")
    )
    input_sha256 = normalize_sha256(contract.get("input_sha256"))
    contract_valid = bool(
        contract.get("citation_leakage_checked")
        and contract.get("use_production_pipeline")
        and contract.get("production_decomposition")
        and contract.get("fail_closed_query_understanding")
        and contract.get("checkpoint_fingerprint")
        and contract.get("minimum_citation_recall") == limits["citation_recall"]
        and runtime_identity is not None
        and retrieval_runtime_contract is not None
        and input_sha256 is not None
    )
    checks["retrieval_dataset_contract"] = {
        "value": contract_valid,
        "minimum": True,
        "status": "passed" if contract_valid else "failed",
    }
    return {
        "passed": all(check["status"] == "passed" for check in checks.values()),
        "expected_sample_count": expected_sample_count,
        "sample_count": sample_count,
        "retrieval_llm_runtime_identity": runtime_identity,
        "retrieval_runtime_contract_sha256": (
            retrieval_runtime_contract["sha256"]
            if retrieval_runtime_contract is not None
            else None
        ),
        "retrieval_input_sha256": input_sha256,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-count", type=int, default=EXPECTED_SAMPLE_COUNT)
    args = parser.parse_args()

    report = build_holdout_gate(
        json.loads(args.retrieval_report.read_text(encoding="utf-8")),
        expected_sample_count=args.expected_sample_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
