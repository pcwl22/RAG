"""Tests for combined Ragas and citation regression thresholds."""
from app.evaluation.quality_gate import build_quality_gate


def _ragas_report(value: float = 0.9):
    return {
        "sample_count": 9,
        "metrics": {
            name: {"mean": value}
            for name in (
                "answer_accuracy",
                "faithfulness",
                "context_precision",
                "context_recall",
            )
        },
    }


def test_quality_gate_passes_when_all_metrics_meet_thresholds():
    report = build_quality_gate(
        _ragas_report(),
        {"evaluated_cases": 9, "recall@5": 0.9, "mrr@5": 0.8},
        top_k=5,
    )

    assert report["passed"] is True
    assert report["checks"]["citation_mrr"]["status"] == "passed"


def test_quality_gate_fails_low_metric_and_missing_required_labels():
    report = build_quality_gate(
        _ragas_report(0.5),
        {"evaluated_cases": 0},
        top_k=5,
    )

    assert report["passed"] is False
    assert report["checks"]["faithfulness"]["status"] == "failed"
    assert report["checks"]["citation_recall"]["status"] == "failed"


def test_quality_gate_can_skip_missing_retrieval_labels():
    report = build_quality_gate(
        _ragas_report(),
        {"evaluated_cases": 0},
        top_k=5,
        require_all=False,
    )

    assert report["passed"] is True
    assert report["checks"]["citation_recall"]["status"] == "skipped"
