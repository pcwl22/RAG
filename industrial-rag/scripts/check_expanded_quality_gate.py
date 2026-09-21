"""Combine full-suite retrieval and representative judge checks into one release gate."""

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

DEFAULT_MINIMUMS = {
    "in_domain_citation_recall": 0.95,
    "adversarial_citation_recall": 0.85,
    "comparison_citation_recall": 0.85,
    "no_answer_abstention_rate": 0.90,
}


def build_report(retrieval: dict[str, Any], ragas_gate: dict[str, Any]) -> dict[str, Any]:
    values = {
        "in_domain_citation_recall": retrieval["in_domain"]["citation_recall"],
        "adversarial_citation_recall": retrieval["groups"]["category:adversarial"][
            "citation_recall"
        ],
        "comparison_citation_recall": retrieval["groups"]["category:comparison"]["citation_recall"],
        "no_answer_abstention_rate": retrieval["no_answer"]["abstention_rate"],
    }
    checks = {
        name: {
            "value": value,
            "minimum": DEFAULT_MINIMUMS[name],
            "status": "passed" if value >= DEFAULT_MINIMUMS[name] else "failed",
        }
        for name, value in values.items()
    }
    checks["representative_judge_gate"] = {
        "value": bool(ragas_gate.get("passed")),
        "minimum": True,
        "status": "passed" if ragas_gate.get("passed") else "failed",
    }
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
        and contract.get("minimum_citation_recall") == DEFAULT_MINIMUMS["in_domain_citation_recall"]
        and runtime_identity is not None
        and retrieval_runtime_contract is not None
        and input_sha256 is not None
    )
    checks["retrieval_dataset_contract"] = {
        "value": contract_valid,
        "minimum": True,
        "status": "passed" if contract_valid else "failed",
    }
    representative_count = int(
        ragas_gate.get("judge_sample_count") or ragas_gate.get("ragas_sample_count") or 0
    )
    return {
        "passed": all(check["status"] == "passed" for check in checks.values()),
        "full_suite_sample_count": retrieval.get("sample_count", 0),
        "representative_judge_sample_count": representative_count,
        "retrieval_llm_runtime_identity": runtime_identity,
        "retrieval_runtime_contract_sha256": (
            retrieval_runtime_contract["sha256"]
            if retrieval_runtime_contract is not None
            else None
        ),
        "retrieval_input_sha256": input_sha256,
        # Historical alias retained for existing artifact readers.
        "representative_ragas_sample_count": representative_count,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--ragas-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retrieval = json.loads(args.retrieval_report.read_text(encoding="utf-8"))
    ragas_gate = json.loads(args.ragas_gate.read_text(encoding="utf-8"))
    report = build_report(retrieval, ragas_gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
