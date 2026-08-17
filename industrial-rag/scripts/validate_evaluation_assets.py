"""Validate committed evaluation datasets before code or prompt changes merge."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_and_validate(
    path: Path,
    *,
    expected_count: int,
    expected_categories: dict[str, int],
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
    return {"path": str(path), "sample_count": len(rows), "categories": dict(categories)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, default=Path("eval/legal_expanded_240.jsonl"))
    parser.add_argument("--ragas", type=Path, default=Path("eval/legal_expanded_ragas_40.jsonl"))
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
        ),
    ]
    print(json.dumps({"passed": True, "datasets": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
