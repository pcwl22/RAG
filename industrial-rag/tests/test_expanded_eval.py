"""Expanded evaluation suite helper tests."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_expanded_eval as expanded_eval
from scripts.build_expanded_eval import build_suite
from scripts.check_expanded_quality_gate import build_report
from scripts.evaluate_retrieval_suite import _aggregate, _score_case
from scripts.sample_eval_suite import sample_cases
from scripts.validate_evaluation_assets import load_and_validate
from scripts.validate_release_baseline import implementation_sha256, validate


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


def _articles(domain, count):
    law = {"civil": "中华人民共和国民法典", "criminal": "中华人民共和国刑法", "labor": "中华人民共和国劳动合同法"}[domain]
    return [
        {"id": f"{domain}-{i}", "content": f"第{i}条 测试规则内容不少于二十个汉字用于评估。", "law_name": law, "article": f"第{i}条"}
        for i in range(1, count + 1)
    ]


def test_expanded_suite_has_expected_strata_and_size():
    suite = build_suite({"civil": _articles("civil", 90), "criminal": _articles("criminal", 90), "labor": _articles("labor", 70)})
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
        lambda: {"postgres": {"host": "db", "port": 5432, "database": "rag", "user": "app", "password": "secret"}},
    )
    monkeypatch.setattr(psycopg2, "connect", lambda **kwargs: FakeConnection())

    assert expanded_eval._load_articles(tenant_id) == {}
    assert any(sql == "SET LOCAL ROLE rag_app" for sql, _params in statements)
    article_query = next((sql, params) for sql, params in statements if "FROM documents" in sql)
    assert "tenant_id = %s::uuid" in article_query[0]
    assert article_query[1] == (tenant_id,)


def test_retrieval_scoring_handles_multi_citation_and_abstention():
    docs = [{"id": "a", "metadata": {"article_number": "第一条"}}, {"id": "b", "metadata": {"article_number": "第二条"}}]
    score = _score_case({"expected_citations": ["第一条", "第二条"]}, docs, 5)
    assert score["citation_hits"] == 2
    assert score["reciprocal_rank_sum"] == 1.5
    assert _score_case({"expected_citations": []}, [], 5)["abstained"] is True
    assert _aggregate([score])["citation_recall"] == 1.0


def test_stratified_sample_balances_categories_and_domains():
    cases = []
    for category in ("standard", "adversarial", "comparison"):
        for domain in ("civil", "criminal", "labor"):
            cases.extend(
                {"id": f"{category}-{domain}-{i}", "metadata": {"category": category, "domain": domain}}
                for i in range(5)
            )
    cases.extend(
        {"id": f"no-answer-{i}", "metadata": {"category": "no_answer", "domain": "out_of_scope"}}
        for i in range(12)
    )
    selected = sample_cases(cases)
    assert len(selected) == 40
    assert sum(case["metadata"]["category"] == "no_answer" for case in selected) == 10


def test_expanded_gate_requires_retrieval_abstention_and_ragas():
    retrieval = {
        "sample_count": 240,
        "in_domain": {"citation_recall": 1.0},
        "no_answer": {"abstention_rate": 0.925},
        "groups": {
            "category:adversarial": {"citation_recall": 1.0},
            "category:comparison": {"citation_recall": 1.0},
        },
    }
    report = build_report(retrieval, {"passed": True, "ragas_sample_count": 40})
    assert report["passed"] is True
    retrieval["no_answer"]["abstention_rate"] = 0.5
    assert build_report(retrieval, {"passed": True})["passed"] is False


def test_release_baseline_is_bound_to_rag_implementation(tmp_path):
    eval_dir = tmp_path / "eval"
    implementation_dir = tmp_path / "app" / "retrieval"
    eval_dir.mkdir()
    implementation_dir.mkdir(parents=True)
    dataset = eval_dir / "suite.jsonl"
    dataset.write_text('{"id":"case-1"}\n', encoding="utf-8")
    implementation_file = implementation_dir / "engine.py"
    implementation_file.write_text("THRESHOLD = 0.5\n", encoding="utf-8")
    paths = ["app/retrieval"]
    baseline = {
        "schema_version": 2,
        "datasets": {
            "suite.jsonl": {
                "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "sample_count": 1,
            }
        },
        "checks": {"quality": {"value": 1.0, "minimum": 0.9}},
        "implementation": {
            "paths": paths,
            "sha256": implementation_sha256(tmp_path, paths),
        },
        "evaluation_contract": {
            "retrieval_top_k": 5,
            "ragas_version": "0.4.3",
            "judge_model": "judge-model",
        },
    }
    baseline_path = eval_dir / "release_baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    assert validate(baseline_path, tmp_path)["passed"] is True
    implementation_file.write_text("THRESHOLD = 0.6\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="implementation changed"):
        validate(baseline_path, tmp_path)
