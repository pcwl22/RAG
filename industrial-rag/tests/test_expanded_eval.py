"""Expanded evaluation suite helper tests."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from app.evaluation.retrieval_contract import build_retrieval_runtime_contract
from scripts import build_expanded_eval as expanded_eval
from scripts.build_expanded_eval import build_suite
from scripts.check_expanded_quality_gate import build_report
from scripts.diagnose_retrieval_case import _deterministic_understanding
from scripts.evaluate_retrieval_suite import (
    _aggregate,
    _checkpoint_path,
    _evaluation_fingerprint,
    _EvaluationLLMFailureTracker,
    _llm_runtime_identity,
    _load_checkpoint,
    _maximum_allowed_misses,
    _require_expected_llm_identity,
    _require_minimum_citation_recall_attainable,
    _require_no_new_llm_failures,
    _score_case,
    _write_checkpoint,
)
from scripts.sample_eval_suite import sample_cases
from scripts.validate_evaluation_assets import load_and_validate
from scripts.validate_release_baseline import (
    CURRENT_SCHEMA_VERSION,
    REQUIRED_IMPLEMENTATION_PATHS,
    implementation_sha256,
    validate,
)


def _retrieval_contract(*, score_threshold=None):
    return build_retrieval_runtime_contract(
        {
            "_meta": {"config_path": "config/base.yaml"},
            "embedding": {"model_name": "embedding-a", "model_revision": "revision-a"},
            "reranker": {"enabled": True, "score_threshold": score_threshold},
            "rag": {"retrieval": {"rerank_min_candidates": 80}},
        }
    )


def test_committed_evaluation_assets_keep_release_quotas():
    project_root = Path(__file__).resolve().parents[1]
    report = load_and_validate(
        project_root / "eval" / "legal_expanded_240.jsonl",
        expected_count=240,
        expected_categories={
            "standard": 140,
            "adversarial": 40,
            "comparison": 20,
            "no_answer": 40,
        },
    )

    assert report["sample_count"] == 240


def test_committed_holdout_asset_is_leakage_free():
    project_root = Path(__file__).resolve().parents[1]
    report = load_and_validate(
        project_root / "eval" / "legal_holdout_150.jsonl",
        expected_count=150,
        expected_categories={"holdout_fact_pattern": 150},
        reject_citation_leakage=True,
    )

    assert report["sample_count"] == 150


def _articles(domain, count):
    law = {
        "civil": "中华人民共和国民法典",
        "criminal": "中华人民共和国刑法",
        "labor": "中华人民共和国劳动合同法",
    }[domain]
    return [
        {
            "id": f"{domain}-{i}",
            "content": f"第{i}条 测试规则内容不少于二十个汉字用于评估。",
            "law_name": law,
            "article": f"第{i}条",
        }
        for i in range(1, count + 1)
    ]


def test_expanded_suite_has_expected_strata_and_size():
    suite = build_suite(
        {
            "civil": _articles("civil", 90),
            "criminal": _articles("criminal", 90),
            "labor": _articles("labor", 70),
        }
    )
    categories = [case["metadata"]["category"] for case in suite]
    assert len(suite) == 240
    assert categories.count("standard") == 140
    assert categories.count("adversarial") == 40
    assert categories.count("comparison") == 20
    assert categories.count("no_answer") == 40


def test_evaluation_article_load_is_scoped_to_one_tenant(monkeypatch):
    import psycopg2

    tenant_id = "00000000-0000-0000-0000-00000000000a"
    statements = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params=None):
            statements.append((" ".join(sql.split()), params))

        def fetchall(self):
            return []

    class FakeConnection:
        def cursor(self, **kwargs):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr(
        expanded_eval,
        "get_settings",
        lambda: {
            "postgres": {
                "host": "db",
                "port": 5432,
                "database": "rag",
                "user": "app",
                "password": "secret",
            }
        },
    )
    monkeypatch.setattr(psycopg2, "connect", lambda **kwargs: FakeConnection())

    assert expanded_eval._load_articles(tenant_id) == {}
    assert any(sql == "SET LOCAL ROLE rag_app" for sql, _params in statements)
    article_query = next((sql, params) for sql, params in statements if "FROM documents" in sql)
    assert "tenant_id = %s::uuid" in article_query[0]
    assert article_query[1] == (tenant_id,)


def test_retrieval_scoring_handles_multi_citation_and_abstention():
    docs = [
        {"id": "a", "metadata": {"article_number": "第一条"}},
        {"id": "b", "metadata": {"article_number": "第二条"}},
    ]
    score = _score_case({"expected_citations": ["第一条", "第二条"]}, docs, 5)
    assert score["citation_hits"] == 2
    assert score["reciprocal_rank_sum"] == 1.5
    assert _score_case({"expected_citations": []}, [], 5)["abstained"] is True
    assert _aggregate([score])["citation_recall"] == 1.0


def test_retrieval_scoring_prefers_exact_article_id_over_same_number_in_other_law():
    case = {
        "expected_citations": ["第二十九条"],
        "expected_sources": ["中华人民共和国刑法"],
        "metadata": {"article_ids": ["刑法_总则_犯罪_共同犯罪_29条"]},
    }
    docs = [
        {
            "id": "other-law-29",
            "metadata": {
                "law_name": "其他法律",
                "article_number": "第二十九条",
                "semantic_chunk_id": "其他法律_29条",
            },
        },
        {
            "id": "row-id",
            "metadata": {
                "law_name": "中华人民共和国刑法",
                "article_number": "第二十九条",
                "semantic_chunk_id": "刑法_总则_犯罪_共同犯罪_29条",
            },
        },
    ]

    score = _score_case(case, docs, 5)

    assert score["citation_hits"] == 1
    assert score["ranks"] == [2]


def test_stratified_sample_balances_categories_and_domains():
    cases = []
    for category in ("standard", "adversarial", "comparison"):
        for domain in ("civil", "criminal", "labor"):
            cases.extend(
                {
                    "id": f"{category}-{domain}-{i}",
                    "metadata": {"category": category, "domain": domain},
                }
                for i in range(5)
            )
    cases.extend(
        {"id": f"no-answer-{i}", "metadata": {"category": "no_answer", "domain": "out_of_scope"}}
        for i in range(12)
    )
    selected = sample_cases(cases)
    assert len(selected) == 40
    assert sum(case["metadata"]["category"] == "no_answer" for case in selected) == 10


def test_expanded_gate_requires_retrieval_abstention_and_judge():
    retrieval = {
        "sample_count": 240,
        "evaluation_contract": {
            "input_sha256": "b" * 64,
            "citation_leakage_checked": True,
            "use_production_pipeline": True,
            "production_decomposition": True,
            "fail_closed_query_understanding": True,
            "checkpoint_fingerprint": "f" * 64,
            "minimum_citation_recall": 0.95,
            "llm_runtime_identity": {
                "provider": "openai_compatible",
                "model_name": "model-a",
                "endpoint_sha256": "a" * 64,
            },
            "retrieval_runtime_contract": _retrieval_contract(),
        },
        "in_domain": {"citation_recall": 1.0},
        "no_answer": {"abstention_rate": 0.925},
        "groups": {
            "category:adversarial": {"citation_recall": 1.0},
            "category:comparison": {"citation_recall": 1.0},
        },
    }
    report = build_report(retrieval, {"passed": True, "judge_sample_count": 40})
    assert report["passed"] is True
    retrieval["no_answer"]["abstention_rate"] = 0.5
    assert build_report(retrieval, {"passed": True})["passed"] is False


def test_expanded_gate_rejects_query_understanding_degradation():
    retrieval = {
        "sample_count": 240,
        "evaluation_contract": {
            "input_sha256": "b" * 64,
            "citation_leakage_checked": True,
            "use_production_pipeline": True,
            "production_decomposition": True,
            "fail_closed_query_understanding": False,
            "checkpoint_fingerprint": "f" * 64,
            "minimum_citation_recall": 0.95,
            "llm_runtime_identity": {
                "provider": "openai_compatible",
                "model_name": "model-a",
                "endpoint_sha256": "a" * 64,
            },
            "retrieval_runtime_contract": _retrieval_contract(),
        },
        "in_domain": {"citation_recall": 1.0},
        "no_answer": {"abstention_rate": 1.0},
        "groups": {
            "category:adversarial": {"citation_recall": 1.0},
            "category:comparison": {"citation_recall": 1.0},
        },
    }

    report = build_report(retrieval, {"passed": True, "judge_sample_count": 40})

    assert report["passed"] is False
    assert report["checks"]["retrieval_dataset_contract"]["status"] == "failed"


def test_production_evaluation_tracks_swallowed_llm_failure():
    class FailingLLM:
        async def generate(self, *_args, **_kwargs):
            raise RuntimeError("provider body must not enter the gate error")

    async def run() -> None:
        tracker = _EvaluationLLMFailureTracker(FailingLLM())
        previous = tracker.failure_count
        with pytest.raises(RuntimeError, match="evaluation LLM request failed") as upstream:
            await tracker.generate(prompt="untrusted question")
        assert "provider body" not in str(upstream.value)
        with pytest.raises(RuntimeError, match="evaluation case safe-id") as error:
            _require_no_new_llm_failures(tracker, previous, "safe-id")
        assert "provider body" not in str(error.value)

    asyncio.run(run())


def test_retrieval_checkpoint_is_atomic_and_bound_to_input_and_options(tmp_path):
    input_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "report.json"
    cases = [
        {
            "id": "case-1",
            "query": "question",
            "expected_citations": ["第一条"],
            "metadata": {"category": "standard", "domain": "civil"},
        }
    ]
    input_path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    fingerprint, context = _evaluation_fingerprint(
        input_path,
        sample_count=1,
        top_k=5,
        limit=0,
        enable_rerank=None,
        enable_rrf=None,
        enable_dynamic_topk=None,
        similarity_threshold=None,
        enable_query_rewrite=False,
        rewrite_concurrency=4,
        use_production_pipeline=True,
        production_decomposition=True,
        understanding_concurrency=4,
        llm_runtime_identity={
            "provider": "openai_compatible",
            "model_name": "model-a",
            "endpoint_sha256": "a" * 64,
        },
        retrieval_runtime_contract=_retrieval_contract(),
        minimum_citation_recall=0.95,
    )
    checkpoint_path = _checkpoint_path(output_path)
    rows = [
        {
            "id": "case-1",
            "category": "standard",
            "domain": "civil",
            "citation_hits": 1,
        }
    ]

    _write_checkpoint(
        checkpoint_path,
        fingerprint=fingerprint,
        context=context,
        rows=rows,
    )

    assert not checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp").exists()
    assert (
        _load_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            cases=cases,
        )
        == rows
    )
    with pytest.raises(ValueError, match="does not match"):
        _load_checkpoint(
            checkpoint_path,
            fingerprint="different-run",
            cases=cases,
        )

    different_fingerprint, _ = _evaluation_fingerprint(
        input_path,
        sample_count=1,
        top_k=5,
        limit=0,
        enable_rerank=None,
        enable_rrf=None,
        enable_dynamic_topk=None,
        similarity_threshold=None,
        enable_query_rewrite=False,
        rewrite_concurrency=4,
        use_production_pipeline=True,
        production_decomposition=True,
        understanding_concurrency=1,
        llm_runtime_identity={
            "provider": "openai_compatible",
            "model_name": "model-a",
            "endpoint_sha256": "a" * 64,
        },
        retrieval_runtime_contract=_retrieval_contract(),
        minimum_citation_recall=0.95,
    )
    assert different_fingerprint != fingerprint


def test_retrieval_checkpoint_fingerprint_changes_with_llm_model(tmp_path):
    input_path = tmp_path / "cases.jsonl"
    input_path.write_text(
        '{"id":"case-1","query":"question"}\n',
        encoding="utf-8",
    )
    options = {
        "sample_count": 1,
        "top_k": 5,
        "limit": 0,
        "enable_rerank": None,
        "enable_rrf": None,
        "enable_dynamic_topk": None,
        "similarity_threshold": None,
        "enable_query_rewrite": False,
        "rewrite_concurrency": 4,
        "use_production_pipeline": True,
        "production_decomposition": True,
        "understanding_concurrency": 1,
        "minimum_citation_recall": 0.95,
        "retrieval_runtime_contract": _retrieval_contract(),
    }

    first, _ = _evaluation_fingerprint(
        input_path,
        llm_runtime_identity={
            "provider": "openai_compatible",
            "model_name": "model-a",
            "endpoint_sha256": "a" * 64,
        },
        **options,
    )
    second, _ = _evaluation_fingerprint(
        input_path,
        llm_runtime_identity={
            "provider": "openai_compatible",
            "model_name": "model-b",
            "endpoint_sha256": "a" * 64,
        },
        **options,
    )

    assert first != second


def test_retrieval_checkpoint_fingerprint_changes_with_retrieval_runtime_contract(tmp_path):
    input_path = tmp_path / "cases.jsonl"
    input_path.write_text('{"id":"case-1","query":"question"}\n', encoding="utf-8")
    options = {
        "sample_count": 1,
        "top_k": 5,
        "limit": 0,
        "enable_rerank": None,
        "enable_rrf": None,
        "enable_dynamic_topk": None,
        "similarity_threshold": None,
        "enable_query_rewrite": False,
        "rewrite_concurrency": 4,
        "use_production_pipeline": True,
        "production_decomposition": True,
        "understanding_concurrency": 1,
        "llm_runtime_identity": {
            "provider": "openai_compatible",
            "model_name": "model-a",
            "endpoint_sha256": "a" * 64,
        },
        "minimum_citation_recall": 0.95,
    }

    first, first_context = _evaluation_fingerprint(
        input_path,
        retrieval_runtime_contract=_retrieval_contract(score_threshold=None),
        **options,
    )
    second, _ = _evaluation_fingerprint(
        input_path,
        retrieval_runtime_contract=_retrieval_contract(score_threshold=0.5),
        **options,
    )

    assert first != second
    assert first_context["retrieval_runtime_contract"]["snapshot"]["reranker"][
        "score_threshold"
    ] is None


def test_deterministic_diagnostic_understanding_splits_and_maps_comparison_query():
    understanding = _deterministic_understanding(
        "情景一：甲教唆十五岁的乙盗窃，乙未实施。"
        "情景二：甲服刑时发现判决前还有盗窃罪未处理，应如何处理？"
        "请分别判断两个情景。"
    )

    assert understanding["is_decomposed"] is True
    assert len(understanding["subqueries"]) == 2
    assert {
        mapping["concept_id"] for mapping in understanding["concept_article_mappings"]
    } == {
        "criminal_instigation_when_instigated_person_does_not_offend",
        "criminal_omitted_offense_after_judgment",
    }
    assert any("刑法_总则_犯罪_共同犯罪_29条" in query for query in understanding["retrieval_queries"])


def test_retrieval_checkpoint_identity_excludes_llm_secrets():
    class FakeLLM:
        provider = "openai_compatible"
        config = {
            "model_name": "model-a",
            "base_url": "https://gateway.example/v1?token=private",
            "api_key": "super-secret",
        }

    identity = _llm_runtime_identity(FakeLLM())

    assert identity["provider"] == "openai_compatible"
    assert identity["model_name"] == "model-a"
    assert len(identity["endpoint_sha256"]) == 64
    serialized = json.dumps(identity)
    assert "gateway.example" not in serialized
    assert "private" not in serialized
    assert "super-secret" not in serialized


def test_expected_llm_identity_accepts_one_exact_operator_selection():
    endpoint_digest = hashlib.sha256(b"https://api.example.test").hexdigest()
    identity = {
        "provider": "openai_compatible",
        "model_name": "approved-model",
        "endpoint_sha256": endpoint_digest,
    }

    _require_expected_llm_identity(
        identity,
        expected_provider="OPENAI_COMPATIBLE",
        expected_model="approved-model",
        expected_endpoint_sha256=f"sha256:{endpoint_digest}",
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_provider": "other"}, "provider"),
        ({"expected_model": "other-model"}, "model"),
        ({"expected_endpoint_sha256": "f" * 64}, "endpoint"),
    ],
)
def test_expected_llm_identity_fails_closed_before_request(overrides, message):
    identity = {
        "provider": "openai_compatible",
        "model_name": "approved-model",
        "endpoint_sha256": "a" * 64,
    }
    expected = {
        "expected_provider": "openai_compatible",
        "expected_model": "approved-model",
        "expected_endpoint_sha256": "a" * 64,
    }
    expected.update(overrides)

    with pytest.raises(ValueError, match=message):
        _require_expected_llm_identity(identity, **expected)


def test_minimum_recall_stops_only_after_full_suite_gate_is_impossible():
    assert _maximum_allowed_misses(220, 0.95) == 11
    attainable_rows = [{"expected_citations": 1, "citation_hits": 0} for _ in range(11)]
    _require_minimum_citation_recall_attainable(
        attainable_rows,
        total_expected_citations=220,
        minimum_citation_recall=0.95,
    )

    with pytest.raises(RuntimeError, match="no longer attainable"):
        _require_minimum_citation_recall_attainable(
            [*attainable_rows, {"expected_citations": 1, "citation_hits": 0}],
            total_expected_citations=220,
            minimum_citation_recall=0.95,
        )


def test_retrieval_checkpoint_rejects_non_prefix_rows(tmp_path):
    checkpoint_path = tmp_path / "report.partial.json"
    cases = [
        {
            "id": "case-1",
            "metadata": {"category": "standard", "domain": "civil"},
        }
    ]
    _write_checkpoint(
        checkpoint_path,
        fingerprint="run",
        context={},
        rows=[
            {
                "id": "different-case",
                "category": "standard",
                "domain": "civil",
            }
        ],
    )

    with pytest.raises(ValueError, match="not an input prefix"):
        _load_checkpoint(checkpoint_path, fingerprint="run", cases=cases)


def test_release_baseline_is_bound_to_rag_implementation(tmp_path):
    eval_dir = tmp_path / "eval"
    implementation_dir = tmp_path / "app" / "retrieval"
    eval_dir.mkdir()
    implementation_dir.mkdir(parents=True)
    datasets = {}
    for name, sample_count in {
        "legal_expanded_240.jsonl": 240,
        "legal_expanded_ragas_40.jsonl": 40,
        "legal_holdout_150.jsonl": 150,
    }.items():
        dataset = eval_dir / name
        dataset.write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"{name}-{index}",
                        "query": "question",
                        "expected_citations": [],
                    }
                )
                + "\n"
                for index in range(sample_count)
            ),
            encoding="utf-8",
        )
        datasets[name] = {
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "sample_count": sample_count,
        }
    implementation_file = implementation_dir / "engine.py"
    implementation_file.write_text("THRESHOLD = 0.5\n", encoding="utf-8")
    for relative in REQUIRED_IMPLEMENTATION_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")
    paths = ["app/retrieval", *sorted(REQUIRED_IMPLEMENTATION_PATHS)]
    baseline = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "status": "approved",
        "datasets": datasets,
        "checks": {"quality": {"value": 1.0, "minimum": 0.9}},
        "implementation": {
            "paths": paths,
            "sha256": implementation_sha256(tmp_path, paths),
        },
        "evaluation_contract": {
            "retrieval_top_k": 5,
            "evaluation_engine": "industrial-rag-native-text-judge",
            "evaluation_engine_version": "1.1",
            "judge_model": "judge-model",
            "retrieval_llm_runtime_identity": {
                "provider": "openai_compatible",
                "model_name": "model-a",
                "endpoint_sha256": "a" * 64,
            },
            "retrieval_runtime_contract_sha256": "b" * 64,
            "answer_generation_policy_sha256": "c" * 64,
            "judge_request_policy_sha256": "d" * 64,
            "evaluation_lock": "requirements-evaluation.lock.txt",
            "holdout_dataset": "legal_holdout_150.jsonl",
            "citation_leakage_policy": "reject",
        },
    }
    baseline_path = eval_dir / "release_baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    assert validate(baseline_path, tmp_path)["passed"] is True
    implementation_file.write_text("THRESHOLD = 0.6\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="implementation changed"):
        validate(baseline_path, tmp_path)


def test_committed_release_baseline_is_approved_after_real_evaluation():
    result = validate()

    assert result["passed"] is True
    assert result["dataset_count"] == 3
    assert result["check_count"] == 12
