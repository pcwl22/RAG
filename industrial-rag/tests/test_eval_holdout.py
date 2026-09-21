"""Guards for the fact-pattern hold-out suite.

The committed expanded suites are derived from screened fact patterns and must
not contain the citation they are scored on. ``eval/legal_holdout_150.jsonl`` is
the source corpus for those cases and remains the independent retrieval contract.
These tests pin the properties that keep the suites from drifting back to
answer-echo evaluation; the rejection test uses a synthetic historical leak.
"""
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.build_holdout_eval import (
    MAX_SHARED_SUBSTRING,
    Article,
    is_discriminative,
    longest_shared_substring,
    screen_question,
)
from scripts.validate_evaluation_assets import load_and_validate

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
HOLDOUT_PATH = EVAL_DIR / "legal_holdout_150.jsonl"
EXPANDED_PATH = EVAL_DIR / "legal_expanded_240.jsonl"
CORRECTIONS_PATH = EVAL_DIR / "legal_holdout_semantic_corrections.json"


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _as_article(record: dict[str, Any]) -> Article:
    return Article(
        law_name=(record.get("expected_sources") or [""])[0],
        article_number=(record.get("expected_citations") or [""])[0],
        article_text=record.get("expected_answer") or "",
        semantic_chunk_id="n/a",
        domain="n/a",
        source_file="n/a",
    )


def _cited(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("expected_citations")]


def _names_own_citation(record: dict[str, Any]) -> bool:
    return any(citation in record["query"] for citation in record["expected_citations"])


def test_holdout_questions_never_name_their_own_citation():
    """The whole point of the suite: answering requires retrieving, not echoing."""
    records = _load(HOLDOUT_PATH)
    assert records, "hold-out suite is empty"

    leaked = [record["id"] for record in _cited(records) if _names_own_citation(record)]

    assert leaked == [], f"hold-out questions leak their own answer: {leaked[:5]}"


def test_holdout_questions_do_not_paraphrase_their_statute():
    """A near-copy of the statute is solvable by lexical overlap, not retrieval."""
    offenders = [
        (record["id"], longest_shared_substring(record["query"], record["expected_answer"]))
        for record in _load(HOLDOUT_PATH)
    ]
    too_close = [item for item in offenders if item[1] > MAX_SHARED_SUBSTRING]

    assert too_close == [], f"questions paraphrase their statute: {too_close[:5]}"


def test_holdout_covers_every_legal_domain():
    domains = {record["metadata"]["domain"] for record in _load(HOLDOUT_PATH)}

    assert domains == {"civil", "criminal", "labor"}


def test_legislative_purpose_clause_is_not_used_as_a_fact_pattern_target():
    assert not is_discriminative(
        "第一条　为了完善劳动合同制度，明确双方权利和义务，制定本法。"
    )
    assert is_discriminative(
        "第七条　用人单位自用工之日起即与劳动者建立劳动关系。"
    )


def test_reviewed_semantic_corrections_are_applied_to_holdout():
    rows_by_id = {row["id"]: row for row in _load(HOLDOUT_PATH)}
    corrections = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))

    for correction in corrections:
        row_id = correction.get("replacement_id", correction["id"])
        row = rows_by_id[row_id]
        assert row["query"] == correction["query"]
        if "expected_citations" in correction:
            assert row["expected_citations"] == correction["expected_citations"]
        if "expected_answer" in correction:
            assert row["expected_answer"] == correction["expected_answer"]
        if "replacement_article_id" in correction:
            assert row["metadata"]["article_ids"] == [
                correction["replacement_article_id"]
            ]


def test_expanded_questions_pass_the_fact_pattern_screen():
    """Every cited case in the committed suite remains a usable fact pattern."""
    rejected = [
        record["id"]
        for record in _cited(_load(EXPANDED_PATH))
        if screen_question(record["query"], _as_article(record)) is not None
    ]

    assert rejected == [], f"screen rejected current fact patterns: {rejected[:5]}"


def test_asset_gate_rejects_a_synthetic_leaky_question(tmp_path: Path):
    """The release gate still rejects the historical citation-echo failure."""
    leaky_path = tmp_path / "leaky.jsonl"
    leaky_path.write_text(
        json.dumps(
            {
                "id": "synthetic-leak-01",
                "query": "请说明第一条规定的主要内容。",
                "expected_citations": ["第一条"],
                "expected_sources": ["测试法"],
                "expected_answer": "第一条 测试内容。",
                "metadata": {"category": "standard", "domain": "civil"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="citation leakage"):
        load_and_validate(
            leaky_path,
            expected_count=1,
            expected_categories={"standard": 1},
            reject_citation_leakage=True,
        )
