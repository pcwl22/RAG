"""Check Ragas plus deterministic citation metrics against regression thresholds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.quality_gate import build_quality_gate  # noqa: E402
from scripts.score_retrieval import score  # noqa: E402


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
