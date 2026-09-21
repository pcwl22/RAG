"""Tests for the optional Ragas evaluation adapter."""
import asyncio
import json

import pytest

from app.evaluation.native_judge import EVALUATION_ENGINE_VERSION
from app.evaluation.ragas_adapter import (
    _is_retryable_evaluation_error,
    build_ragas_sample,
    eligibility_summary,
    evaluate_samples,
    load_ragas_samples,
    validate_metric_names,
)
from app.llm.request_policy import build_structured_output_policy


def _evaluation_context(model: str = "judge-a") -> dict[str, object]:
    policy = build_structured_output_policy(
        provider="openai_compatible",
        model_name=model,
        base_url="https://judge.example/v1",
    )
    return {
        "evaluation_engine": "industrial-rag-native-text-judge",
        "evaluation_engine_version": EVALUATION_ENGINE_VERSION,
        "judge_model": model,
        "judge_base_url": "https://judge.example/v1",
        "judge_request_policy_sha256": policy["sha256"],
        "max_tokens": 2048,
        "temperature": 0,
    }


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


def test_ragas_sample_excludes_system_added_evidence_section():
    sample = build_ragas_sample(
        {
            "query": "问题",
            "prediction": "【回答】\n结论：\n系统生成的结论\n\n依据：\n1. 自动追加的来源",
            "target": "参考",
            "contexts": [{"text": "法条"}],
        }
    )

    assert sample.response == "系统生成的结论"


@pytest.mark.parametrize(
    "response",
    [
        "第一项成立。\n依据合同约定，第二项也成立。",
        "有上下文支持的陈述。\n依据错误结论仍属于待评分的回答内容。",
    ],
)
def test_ragas_sample_preserves_non_envelope_basis_prose(response):
    sample = build_ragas_sample(
        {
            "query": "问题",
            "prediction": response,
            "target": "参考",
            "contexts": [{"text": "法条"}],
        }
    )

    assert sample.response == response


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
        validate_metric_names(["faithfulness", "multi_modal_faithfulness"])


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


def test_evaluate_samples_bounds_async_metric_concurrency(monkeypatch):
    class Result:
        value = 0.5
        reason = None

    class AsyncMetric:
        active = 0
        max_active = 0

        async def ascore(self, **_inputs):
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
            await asyncio.sleep(0)
            type(self).active -= 1
            return Result()

    metric = AsyncMetric()
    monkeypatch.setattr(
        "app.evaluation.ragas_adapter._metric_instances",
        lambda _llm, names: dict.fromkeys(names, metric),
    )
    samples = [
        build_ragas_sample(
            {"id": str(index), "query": "Q", "answer": "A", "contexts": ["C"]}
        )
        for index in range(3)
    ]

    report = evaluate_samples(
        samples,
        llm=object(),
        metric_names=["faithfulness"],
        max_concurrency=1,
    )

    assert report["metrics"]["faithfulness"]["evaluated_cases"] == 3
    assert metric.max_active == 1


def test_ragas_retry_classifier_only_allows_known_function_registry_400():
    class GatewayFunctionError(Exception):
        status_code = 400

    class ValidationError(Exception):
        status_code = 400

    assert _is_retryable_evaluation_error(
        GatewayFunctionError("internal function reference does not exist")
    )
    assert not _is_retryable_evaluation_error(ValidationError("invalid request schema"))


def test_evaluate_samples_resumes_completed_metric_from_checkpoint(monkeypatch, tmp_path):
    class Result:
        value = 0.75
        reason = None

    class FakeMetric:
        def batch_score(self, inputs):
            return [Result() for _ in inputs]

    sample = build_ragas_sample(
        {"id": "1", "query": "Q", "answer": "A", "contexts": ["C"]}
    )
    checkpoint = tmp_path / "ragas.partial.json"
    monkeypatch.setattr(
        "app.evaluation.ragas_adapter._metric_instances",
        lambda _llm, names: dict.fromkeys(names, FakeMetric()),
    )
    first = evaluate_samples(
        [sample],
        llm=object(),
        metric_names=["faithfulness"],
        checkpoint_path=checkpoint,
        evaluation_context=_evaluation_context(),
    )
    assert checkpoint.is_file()

    class UnexpectedMetric:
        def batch_score(self, _inputs):
            raise AssertionError("completed metrics must not be called on resume")

    monkeypatch.setattr(
        "app.evaluation.ragas_adapter._metric_instances",
        lambda _llm, names: dict.fromkeys(names, UnexpectedMetric()),
    )
    resumed = evaluate_samples(
        [sample],
        llm=object(),
        metric_names=["faithfulness"],
        checkpoint_path=checkpoint,
        resume=True,
        evaluation_context=_evaluation_context(),
    )

    assert first["metrics"] == resumed["metrics"]


def test_evaluate_samples_rejects_checkpoint_from_different_judge(monkeypatch, tmp_path):
    class Result:
        value = 0.75
        reason = None

    class FakeMetric:
        def batch_score(self, inputs):
            return [Result() for _ in inputs]

    sample = build_ragas_sample(
        {"id": "1", "query": "Q", "answer": "A", "contexts": ["C"]}
    )
    checkpoint = tmp_path / "evaluation.partial.json"
    monkeypatch.setattr(
        "app.evaluation.ragas_adapter._metric_instances",
        lambda _llm, names: dict.fromkeys(names, FakeMetric()),
    )
    evaluate_samples(
        [sample],
        llm=object(),
        metric_names=["faithfulness"],
        checkpoint_path=checkpoint,
        evaluation_context=_evaluation_context("judge-a"),
    )

    with pytest.raises(ValueError, match="judge contract"):
        evaluate_samples(
            [sample],
            llm=object(),
            metric_names=["faithfulness"],
            checkpoint_path=checkpoint,
            resume=True,
            evaluation_context=_evaluation_context("judge-b"),
        )


def test_evaluate_samples_requires_judge_identity_when_checkpointing(tmp_path):
    sample = build_ragas_sample(
        {"id": "1", "query": "Q", "answer": "A", "contexts": ["C"]}
    )

    with pytest.raises(ValueError, match="evaluation_context is required"):
        evaluate_samples(
            [sample],
            llm=object(),
            metric_names=["faithfulness"],
            checkpoint_path=tmp_path / "evaluation.partial.json",
        )
