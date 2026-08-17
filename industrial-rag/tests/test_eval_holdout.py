"""Guards for the fact-pattern hold-out suite.

Every citation-bearing question in ``eval/legal_expanded_240.jsonl`` contains,
verbatim, the article number it is scored on. Its ``citation_recall = 1.0`` in
``eval/release_baseline.json`` therefore measures string echo, not retrieval.

``eval/legal_holdout_150.jsonl`` exists to measure what the expanded suite cannot.
These tests pin the properties that make it different so it cannot drift back into
the same defect.
"""
import json
from pathlib import Path
from typing import Any

from scripts.build_holdout_eval import (
    MAX_SHARED_SUBSTRING,
    Article,
    longest_shared_substring,
    screen_question,
)

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
HOLDOUT_PATH = EVAL_DIR / "legal_holdout_150.jsonl"
EXPANDED_PATH = EVAL_DIR / "legal_expanded_240.jsonl"


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


def test_screen_rejects_every_leaky_expanded_question():
    """The screen must have teeth, so prove it against the known-bad suite."""
    slipped = [
        record["id"]
        for record in _cited(_load(EXPANDED_PATH))
        if screen_question(record["query"], _as_article(record)) is None
    ]

    assert slipped == [], f"screen accepted leaky questions: {slipped[:5]}"


def test_expanded_suite_still_cannot_measure_retrieval():
    """Tripwire: stops the expanded suite's 1.0 being quoted as a retrieval result.

    This asserts a defect on purpose. If the suite is ever regenerated so that
    questions no longer hand over their own answer, this test fails -- at which
    point re-run the release evaluation, update eval/release_baseline.json, and
    delete this guard.
    """
    records = _cited(_load(EXPANDED_PATH))
    leaking = [record for record in records if _names_own_citation(record)]

    assert len(leaking) == len(records) == 200, (
        "legal_expanded_240.jsonl no longer leaks every answer; re-approve the "
        "release baseline and remove this tripwire"
    )
