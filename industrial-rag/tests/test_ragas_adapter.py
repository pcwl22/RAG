"""Tests for the optional Ragas evaluation adapter."""
import json

import pytest

from app.evaluation.ragas_adapter import (
    build_ragas_sample,
    eligibility_summary,
    evaluate_samples,
    load_ragas_samples,
    validate_metric_names,
)


def test_build_ragas_sample_maps_lrage_export_fields():
    sample = build_ragas_sample(
        {
            "id": "legal-1",
            "query": "问题",
            "prediction": "系统答案",
            "target": "参考答案",
            "contexts": [
                {"text": "法条一"},
                {"metadata": {"article_text": "法条二"}},
                {"text": ""},
            ],
        }
    )

    assert sample.case_id == "legal-1"
    assert sample.user_input == "问题"
    assert sample.response == "系统答案"
    assert sample.reference == "参考答案"
    assert sample.retrieved_contexts == ["法条一", "法条二"]


def test_comparison_sample_groups_contexts_for_joint_relevance():
    sample = build_ragas_sample(
        {
            "query": "比较第一条和第二条",
            "answer": "答案",
            "target": "参考",
            "contexts": [{"text": "第一条"}, {"text": "第二条"}],
            "metadata": {"case_metadata": {"category": "comparison"}},
        }
    )
    assert sample.retrieved_contexts == ["第一条\n\n第二条"]
    assert sample.source_context_count == 2


def test_load_ragas_samples_validates_generated_response(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"query": "问题"}, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="response"):
        load_ragas_samples(path)


def test_eligibility_allows_reference_free_faithfulness():
    sample = build_ragas_sample(
        {"query": "Q", "answer": "A", "contexts": [{"text": "C"}]}
    )

    assert eligibility_summary(
        [sample], ["answer_accuracy", "faithfulness", "context_recall"]
    ) == {
        "answer_accuracy": 0,
        "faithfulness": 1,
        "context_recall": 0,
    }


def test_validate_metric_names_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unknown metrics"):
        validate_metric_names(["faithfulness", "made_up"])


def test_evaluate_samples_builds_aggregate_and_case_scores(monkeypatch):
    class Result:
        def __init__(self, value):
            self.value = value
            self.reason = None

    class FakeMetric:
        def batch_score(self, inputs):
            assert inputs[0]["retrieved_contexts"] == ["C"]
            return [Result(0.75) for _ in inputs]

    monkeypatch.setattr(
        "app.evaluation.ragas_adapter._metric_instances",
        lambda _llm, names: {name: FakeMetric() for name in names},
    )
    sample = build_ragas_sample(
        {"id": "1", "query": "Q", "answer": "A", "target": "R", "contexts": ["C"]}
    )

    report = evaluate_samples([sample], llm=object(), metric_names=["faithfulness"])

    assert report["metrics"]["faithfulness"] == {
        "mean": 0.75,
        "evaluated_cases": 1,
        "eligible_cases": 1,
    }
    assert report["samples"][0]["scores"]["faithfulness"] == 0.75
