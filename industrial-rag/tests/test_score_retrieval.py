"""Tests for citation-level retrieval scoring."""
import json

from scripts.score_retrieval import score


def test_multi_citation_case_requires_each_expected_citation(tmp_path):
    export = tmp_path / "samples.jsonl"
    export.write_text(
        json.dumps(
            {
                "metadata": {"expected_citations": ["第二百六十四条", "第二百七十一条"]},
                "contexts": [{"citation": "中华人民共和国刑法第二百六十四条"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = score(export, top_k=5)

    assert report["evaluated_cases"] == 1
    assert report["evaluated_citations"] == 2
    assert report["recall@5"] == 0.5
    assert report["mrr@5"] == 0.5
