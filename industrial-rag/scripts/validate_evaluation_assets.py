"""Validate committed evaluation datasets before code or prompt changes merge."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _citation_variants(citation: str) -> set[str]:
    """Return article-number spellings that must not appear in a question."""
    citation = str(citation).strip()
    if not citation:
        return set()

    variants = {citation}
    core = citation.removeprefix("第")
    numeral = ""
    if "条" in core:
        numeral, suffix = core.split("条", 1)
        article_suffix = f"条{suffix}"
        variants.update({f"第{numeral}{article_suffix}", f"{numeral}{article_suffix}"})
    elif core:
        numeral = core
        variants.update({f"第{numeral}条", f"{numeral}条"})

    # The committed suites use Chinese article numbers.  Cover the common
    # Arabic spelling too, so the gate cannot be bypassed by changing only the
    # numeral format.
    chinese_digits = "零一二三四五六七八九"
    units = {"十": 10, "百": 100, "千": 1000}
    if numeral and all(char in chinese_digits or char in units for char in numeral):
        section = 0
        current = 0
        for char in numeral.split("之", 1)[0]:
            if char in chinese_digits:
                current = chinese_digits.index(char)
            elif char in units:
                section += (current or 1) * units[char]
                current = 0
        number = section + current
        if number:
            variants.update({str(number), f"第{number}条", f"{number}条"})
    return {variant for variant in variants if variant}


def citation_leak_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Find cases where the question contains its own expected citation."""
    leaked: list[str] = []
    for row in rows:
        query = str(row.get("query") or "")
        citations = row.get("expected_citations") or []
        if any(variant in query for citation in citations for variant in _citation_variants(str(citation))):
            leaked.append(str(row.get("id")))
    return leaked


def load_and_validate(
    path: Path,
    *,
    expected_count: int,
    expected_categories: dict[str, int],
    reject_citation_leakage: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        for field in ("id", "query", "metadata"):
            if not row.get(field):
                raise ValueError(f"{path}:{line_number}: missing {field}")
        category = row["metadata"].get("category")
        if category != "no_answer":
            for field in ("expected_citations", "expected_sources", "expected_answer"):
                if not row.get(field):
                    raise ValueError(f"{path}:{line_number}: missing {field}")
        rows.append(row)

    if len(rows) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} rows, found {len(rows)}")
    ids = [str(row["id"]) for row in rows]
    duplicate_ids = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"{path}: duplicate ids: {', '.join(duplicate_ids[:5])}")
    categories = Counter(str(row["metadata"].get("category")) for row in rows)
    if dict(categories) != expected_categories:
        raise ValueError(
            f"{path}: category quotas changed; expected {expected_categories}, found {dict(categories)}"
        )
    if reject_citation_leakage:
        leaked = citation_leak_ids(rows)
        if leaked:
            raise ValueError(
                f"{path}: citation leakage detected in {len(leaked)} cases; "
                f"examples: {', '.join(leaked[:5])}"
            )
    return {"path": str(path), "sample_count": len(rows), "categories": dict(categories)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, default=Path("eval/legal_expanded_240.jsonl"))
    parser.add_argument("--ragas", type=Path, default=Path("eval/legal_expanded_ragas_40.jsonl"))
    parser.add_argument("--holdout", type=Path, default=Path("eval/legal_holdout_150.jsonl"))
    args = parser.parse_args()
    reports = [
        load_and_validate(
            args.full,
            expected_count=240,
            expected_categories={
                "standard": 140,
                "adversarial": 40,
                "comparison": 20,
                "no_answer": 40,
            },
            reject_citation_leakage=True,
        ),
        load_and_validate(
            args.ragas,
            expected_count=40,
            expected_categories={
                "standard": 10,
                "adversarial": 10,
                "comparison": 10,
                "no_answer": 10,
            },
            reject_citation_leakage=True,
        ),
        load_and_validate(
            args.holdout,
            expected_count=150,
            expected_categories={"holdout_fact_pattern": 150},
            reject_citation_leakage=True,
        ),
    ]
    print(json.dumps({"passed": True, "datasets": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
