"""Tests for combined Ragas and citation regression thresholds."""
import hashlib
import json

import pytest

from app.evaluation.native_judge import EVALUATION_ENGINE_VERSION
from app.evaluation.quality_gate import build_quality_gate
from app.evaluation.retrieval_contract import build_retrieval_runtime_contract
from app.llm.request_policy import build_structured_output_policy
from scripts.check_quality_gate import _export_evaluation_contract


def _ragas_report(value: float = 0.9):
    return {
        "sample_count": 9,
        "evaluation_engine": "industrial-rag-native-text-judge",
        "evaluation_engine_version": EVALUATION_ENGINE_VERSION,
        "judge_model": "judge-model",
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
    assert report["judge_sample_count"] == 9
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


def test_export_evaluation_contract_binds_source_and_runtime_identity(tmp_path):
    path = tmp_path / "export.jsonl"
    identity = {
        "provider": "openai_compatible",
        "model_name": "model-a",
        "endpoint_sha256": "a" * 64,
    }
    retrieval_contract = build_retrieval_runtime_contract(
        {
            "_meta": {"config_path": "config/base.yaml"},
            "rag": {"retrieval": {}},
        }
    )
    answer_policy = build_structured_output_policy(
        provider="openai_compatible",
        model_name="model-a",
        base_url="https://provider.example/v1",
    )
    sample = {
        "id": "case-1",
        "metadata": {
            "evaluation_contract": {
                "source_dataset_sha256": "b" * 64,
                "llm_runtime_identity": identity,
                "retrieval_runtime_contract": retrieval_contract,
                "answer_generation_policy": answer_policy,
            }
        },
    }
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    export_sha256, contract = _export_evaluation_contract(path)

    assert export_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert contract == {
        "export_sample_count": 1,
        "source_dataset_sha256": "b" * 64,
        "retrieval_llm_runtime_identity": identity,
        "retrieval_runtime_contract_sha256": retrieval_contract["sha256"],
        "answer_generation_policy_sha256": answer_policy["sha256"],
    }

    sample["metadata"]["evaluation_contract"]["source_dataset_sha256"] = "c" * 64
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sample) + "\n")
    with pytest.raises(ValueError, match="mixed evaluation contracts"):
        _export_evaluation_contract(path)
