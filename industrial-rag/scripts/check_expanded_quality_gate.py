"""Combine full-suite retrieval and representative Ragas checks into one release gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_MINIMUMS = {
    "in_domain_citation_recall": 0.95,
    "adversarial_citation_recall": 0.85,
    "comparison_citation_recall": 0.85,
    "no_answer_abstention_rate": 0.90,
}


def build_report(retrieval: dict[str, Any], ragas_gate: dict[str, Any]) -> dict[str, Any]:
    values = {
        "in_domain_citation_recall": retrieval["in_domain"]["citation_recall"],
        "adversarial_citation_recall": retrieval["groups"]["category:adversarial"]["citation_recall"],
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
    checks["representative_ragas_gate"] = {
        "value": bool(ragas_gate.get("passed")),
        "minimum": True,
        "status": "passed" if ragas_gate.get("passed") else "failed",
    }
    return {
        "passed": all(check["status"] == "passed" for check in checks.values()),
        "full_suite_sample_count": retrieval.get("sample_count", 0),
        "representative_ragas_sample_count": ragas_gate.get("ragas_sample_count", 0),
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
