"""Check the native text judge plus deterministic citation regression thresholds."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.quality_gate import build_quality_gate  # noqa: E402
from app.evaluation.retrieval_contract import (  # noqa: E402
    normalize_llm_runtime_identity,
    normalize_retrieval_runtime_contract,
    normalize_sha256,
)
from app.llm.request_policy import normalize_structured_output_policy  # noqa: E402
from scripts.score_retrieval import score  # noqa: E402


def _export_evaluation_contract(path: Path) -> tuple[str, dict[str, object]]:
    export_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    source_sha256: str | None = None
    runtime_identity: dict[str, object] | None = None
    retrieval_contract_sha256: str | None = None
    answer_policy_sha256: str | None = None
    sample_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample_count += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid export JSON at {path}:{line_number}") from exc
            contract = (sample.get("metadata") or {}).get("evaluation_contract") or {}
            sample_source = normalize_sha256(contract.get("source_dataset_sha256"))
            sample_identity = normalize_llm_runtime_identity(
                contract.get("llm_runtime_identity")
            )
            sample_retrieval_contract = normalize_retrieval_runtime_contract(
                contract.get("retrieval_runtime_contract")
            )
            sample_answer_policy = normalize_structured_output_policy(
                contract.get("answer_generation_policy")
            )
            if (
                sample_source is None
                or sample_identity is None
                or sample_retrieval_contract is None
                or sample_answer_policy is None
            ):
                raise ValueError(f"export sample lacks a valid evaluation contract: {line_number}")
            sample_retrieval_digest = str(sample_retrieval_contract["sha256"])
            if source_sha256 is None:
                source_sha256 = sample_source
                runtime_identity = sample_identity
                retrieval_contract_sha256 = sample_retrieval_digest
                answer_policy_sha256 = str(sample_answer_policy["sha256"])
            elif (
                sample_source != source_sha256
                or sample_identity != runtime_identity
                or sample_retrieval_digest != retrieval_contract_sha256
                or str(sample_answer_policy["sha256"]) != answer_policy_sha256
            ):
                raise ValueError("export samples contain mixed evaluation contracts")
    if (
        sample_count == 0
        or source_sha256 is None
        or runtime_identity is None
        or retrieval_contract_sha256 is None
        or answer_policy_sha256 is None
    ):
        raise ValueError("export contains no contracted evaluation samples")
    return export_sha256, {
        "export_sample_count": sample_count,
        "source_dataset_sha256": source_sha256,
        "retrieval_llm_runtime_identity": runtime_identity,
        "retrieval_runtime_contract_sha256": retrieval_contract_sha256,
        "answer_generation_policy_sha256": answer_policy_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ragas-report", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/ragas_eval/quality_gate.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-answer-accuracy", type=float, default=0.70)
    parser.add_argument("--min-faithfulness", type=float, default=0.80)
    parser.add_argument("--min-context-precision", type=float, default=0.60)
    parser.add_argument("--min-context-recall", type=float, default=0.70)
    parser.add_argument("--min-citation-recall", type=float, default=0.80)
    parser.add_argument("--min-citation-mrr", type=float, default=0.70)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ragas_report = json.loads(args.ragas_report.read_text(encoding="utf-8"))
    export_sha256, export_contract = _export_evaluation_contract(args.export)
    retrieval_report = score(args.export, args.top_k)
    thresholds = {
        "answer_accuracy": args.min_answer_accuracy,
        "faithfulness": args.min_faithfulness,
        "context_precision": args.min_context_precision,
        "context_recall": args.min_context_recall,
        "citation_recall": args.min_citation_recall,
        "citation_mrr": args.min_citation_mrr,
    }
    report = build_quality_gate(
        ragas_report,
        retrieval_report,
        top_k=args.top_k,
        thresholds=thresholds,
        require_all=not args.allow_missing,
    )
    judge_input_sha256 = normalize_sha256(ragas_report.get("input_sha256"))
    judge_request_policy = normalize_structured_output_policy(
        ragas_report.get("judge_request_policy")
    )
    evidence_contract_valid = (
        judge_input_sha256 == export_sha256
        and int(ragas_report.get("sample_count") or 0)
        == export_contract["export_sample_count"]
        and judge_request_policy is not None
    )
    report["checks"]["release_evidence_contract"] = {
        "value": evidence_contract_valid,
        "minimum": True,
        "status": "passed" if evidence_contract_valid else "failed",
    }
    report["passed"] = all(
        check["status"] != "failed" for check in report["checks"].values()
    )
    report["evaluated_export_sha256"] = export_sha256
    report["judge_request_policy_sha256"] = (
        str(judge_request_policy["sha256"])
        if judge_request_policy is not None
        else None
    )
    report.update(export_contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
