"""Tests for LRAGE export adapter."""
import argparse
import json

import pytest

from app.evaluation.lrage_adapter import (
    build_lrage_sample,
    context_text,
    load_cases_jsonl,
    write_lrage_export,
)
from app.evaluation.retrieval_contract import build_retrieval_runtime_contract
from app.llm.request_policy import build_structured_output_policy
from scripts.export_lrage import _export_fingerprint, _load_checkpoint, _write_checkpoint


def test_context_text_prefers_model_visible_text():
    doc = {
        "content": "stored",
        "metadata": {
            "article_text": "article",
            "child_content": "child",
        },
    }

    assert context_text(doc) == "child"


def test_build_lrage_sample_contains_lrage_and_debug_fields():
    sample = build_lrage_sample(
        case_id="case-1",
        query="劳动合同问题？",
        answer="系统答案",
        expected_answer="参考答案",
        results=[
            {
                "id": "chunk-1",
                "score": 0.9,
                "content": "法条内容",
                "metadata": {
                    "filename": "劳动合同法.docx",
                    "legal_citation": "第四十六条",
                    "semantic_chunk_id": "劳动合同法_46条",
                },
            }
        ],
        understanding={"resolved_query": "劳动合同问题？"},
        sub_answers=[],
    )

    assert sample["id"] == "case-1"
    assert sample["Prompt"] == "劳动合同问题？"
    assert "法条内容" in sample["Document"]
    assert sample["Rubric"]
    assert sample["target"] == "参考答案"
    assert sample["prediction"] == "系统答案"
    assert sample["contexts"][0]["semantic_chunk_id"] == "劳动合同法_46条"
    assert sample["understanding"]["resolved_query"] == "劳动合同问题？"


def test_build_lrage_sample_generates_stable_id_and_preserves_zero_rerank_prob():
    first = build_lrage_sample(
        query="same question",
        answer="answer",
        results=[{"id": "chunk-1", "metadata": {"rerank_prob": 0.0}}],
    )
    second = build_lrage_sample(query="same question", answer="answer")

    assert first["id"] == second["id"]
    assert first["id"].startswith("case-")
    assert first["contexts"][0]["rerank_prob"] == 0.0


def test_write_lrage_export_outputs_jsonl_yaml_and_manifest(tmp_path):
    sample = build_lrage_sample(query="Q", answer="A")
    paths = write_lrage_export(
        output_dir=tmp_path,
        task_name="demo_task",
        samples=[sample],
        evaluation_contract={"source_dataset_sha256": "a" * 64},
    )

    assert paths.dataset_jsonl.exists()
    assert paths.task_yaml.exists()
    assert paths.manifest_json.exists()

    row = json.loads(paths.dataset_jsonl.read_text(encoding="utf-8").strip())
    assert row["Prompt"] == "Q"
    assert "dataset_path: json" in paths.task_yaml.read_text(encoding="utf-8")
    assert "metric: LLM-Eval" in paths.task_yaml.read_text(encoding="utf-8")

    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert manifest["sample_count"] == 1
    assert manifest["evaluation_contract"]["source_dataset_sha256"] == "a" * 64
    assert manifest["schema"]["lrage_fields"] == ["Prompt", "Document", "Rubric", "target"]


def test_load_cases_jsonl_validates_required_query(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id":"1","query":"Q"}\n', encoding="utf-8")

    assert load_cases_jsonl(path) == [{"id": "1", "query": "Q"}]

    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text('{"id":"1"}\n', encoding="utf-8")

    try:
        load_cases_jsonl(bad_path)
    except ValueError as exc:
        assert "query" in str(exc)
    else:
        raise AssertionError("Expected missing query to fail")


def test_export_checkpoint_is_resumable_but_rejects_foreign_cases(tmp_path):
    checkpoint = tmp_path / "task.partial.json"
    sample = build_lrage_sample(case_id="case-1", query="Q", answer="A")
    _write_checkpoint(checkpoint, fingerprint="run-a", context={}, records=[sample])

    assert (
        _load_checkpoint(
            checkpoint,
            fingerprint="run-a",
            cases=[{"id": "case-1", "query": "Q"}],
        )
        == [sample]
    )

    with pytest.raises(ValueError, match="not an input prefix"):
        _load_checkpoint(
            checkpoint,
            fingerprint="run-a",
            cases=[{"id": "case-2", "query": "Q"}],
        )
    with pytest.raises(ValueError, match="does not match"):
        _load_checkpoint(
            checkpoint,
            fingerprint="run-b",
            cases=[{"id": "case-1", "query": "Q"}],
        )


def test_export_checkpoint_fingerprint_binds_retrieval_runtime_contract():
    args = argparse.Namespace(
        limit=0,
        top_k=5,
        similarity_threshold=0.05,
        enable_rerank=True,
        partition=None,
        disable_coreference=False,
        disable_decomposition=False,
        disable_rewrite=False,
        max_context_chars=4000,
        max_document_chars=16000,
    )

    def contract(score_threshold):
        return build_retrieval_runtime_contract(
            {
                "_meta": {"config_path": "config/base.yaml"},
                "reranker": {"enabled": True, "score_threshold": score_threshold},
                "rag": {"retrieval": {}},
            }
        )

    identity = {
        "provider": "openai_compatible",
        "model_name": "model-a",
        "endpoint_sha256": "a" * 64,
    }
    answer_policy = build_structured_output_policy(
        provider="openai_compatible",
        model_name="model-a",
        base_url="https://provider.example/v1",
    )
    first, context = _export_fingerprint(
        args,
        source_dataset_sha256="b" * 64,
        llm_runtime_identity=identity,
        retrieval_runtime_contract=contract(None),
        answer_generation_policy=answer_policy,
        sample_count=40,
    )
    second, _ = _export_fingerprint(
        args,
        source_dataset_sha256="b" * 64,
        llm_runtime_identity=identity,
        retrieval_runtime_contract=contract(0.5),
        answer_generation_policy=answer_policy,
        sample_count=40,
    )

    assert first != second
    assert context["retrieval_runtime_contract"]["snapshot"]["reranker"][
        "score_threshold"
    ] is None
    assert context["answer_generation_policy"] == answer_policy
