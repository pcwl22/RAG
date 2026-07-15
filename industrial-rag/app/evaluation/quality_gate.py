"""Combine Ragas and deterministic retrieval metrics into a CI quality gate."""
from __future__ import annotations

from typing import Any

DEFAULT_THRESHOLDS = {
    "answer_accuracy": 0.70,
    "faithfulness": 0.80,
    "context_precision": 0.60,
    "context_recall": 0.70,
    "citation_recall": 0.80,
    "citation_mrr": 0.70,
}


def build_quality_gate(
    ragas_report: dict[str, Any],
    retrieval_report: dict[str, Any],
    *,
    top_k: int,
    thresholds: dict[str, float] | None = None,
    require_all: bool = True,
) -> dict[str, Any]:
    """Build a machine-readable pass/fail report from both evaluation families."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    ragas_metrics = ragas_report.get("metrics") or {}
    values = {
        name: (ragas_metrics.get(name) or {}).get("mean")
        for name in (
            "answer_accuracy",
            "faithfulness",
            "context_precision",
            "context_recall",
        )
    }
    has_retrieval_labels = int(retrieval_report.get("evaluated_cases") or 0) > 0
    values["citation_recall"] = (
        retrieval_report.get(f"recall@{top_k}") if has_retrieval_labels else None
    )
    values["citation_mrr"] = (
        retrieval_report.get(f"mrr@{top_k}") if has_retrieval_labels else None
    )

    checks: dict[str, dict[str, Any]] = {}
    for name, minimum in limits.items():
        value = values.get(name)
        if value is None:
            status = "failed" if require_all else "skipped"
        else:
            status = "passed" if float(value) >= minimum else "failed"
        checks[name] = {"value": value, "minimum": minimum, "status": status}

    return {
        "passed": all(check["status"] != "failed" for check in checks.values()),
        "require_all": require_all,
        "top_k": top_k,
        "ragas_sample_count": ragas_report.get("sample_count", 0),
        "retrieval_evaluated_cases": retrieval_report.get("evaluated_cases", 0),
        "checks": checks,
    }
