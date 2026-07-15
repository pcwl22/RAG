"""Expanded evaluation suite helper tests."""
from scripts.build_expanded_eval import build_suite
from scripts.check_expanded_quality_gate import build_report
from scripts.evaluate_retrieval_suite import _aggregate, _score_case
from scripts.sample_eval_suite import sample_cases


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
