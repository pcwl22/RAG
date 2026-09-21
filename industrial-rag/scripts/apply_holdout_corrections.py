"""Apply reviewed semantic corrections to the committed holdout JSONL atomically."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("eval/legal_holdout_150.jsonl")
DEFAULT_CORRECTIONS = Path("eval/legal_holdout_semantic_corrections.json")


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"correction {field} must be a non-empty string")
    return value.strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid or empty holdout JSONL: {path}")
    return rows


def load_corrections(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"invalid or empty correction manifest: {path}")
    if any(not isinstance(item, dict) for item in payload):
        raise ValueError(f"correction manifest entries must be objects: {path}")
    return payload


def apply_corrections(
    rows: list[dict[str, Any]], corrections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    repaired = deepcopy(rows)
    index_by_id = {str(row.get("id") or ""): index for index, row in enumerate(repaired)}
    if "" in index_by_id or len(index_by_id) != len(repaired):
        raise ValueError("holdout rows must have unique non-empty ids")

    for correction in corrections:
        source_id = _required_text(correction.get("id"), "id")
        replacement_id = _required_text(
            correction.get("replacement_id", source_id), "replacement_id"
        )
        index = index_by_id.get(source_id)
        already_replaced = index is None and replacement_id != source_id
        if already_replaced:
            index = index_by_id.get(replacement_id)
        if index is None:
            raise ValueError(f"correction target not found: {source_id}")

        row = deepcopy(repaired[index])
        metadata = dict(row.get("metadata") or {})
        article_ids = list(metadata.get("article_ids") or [])
        expected_original = _required_text(
            correction.get("expected_original_article_id"),
            "expected_original_article_id",
        )
        replacement_article = correction.get("replacement_article_id")
        acceptable_article_ids = {expected_original}
        if replacement_article is not None:
            acceptable_article_ids.add(_required_text(replacement_article, "replacement_article_id"))
        if len(article_ids) != 1 or str(article_ids[0]) not in acceptable_article_ids:
            raise ValueError(
                f"correction {source_id} no longer targets the reviewed article: {article_ids}"
            )

        row["id"] = replacement_id
        row["query"] = _required_text(correction.get("query"), "query")
        for field in ("expected_citations", "expected_sources", "expected_answer"):
            if field in correction:
                row[field] = deepcopy(correction[field])
        if replacement_article is not None:
            metadata["article_ids"] = [
                _required_text(replacement_article, "replacement_article_id")
            ]
        row["metadata"] = metadata

        if replacement_id != source_id and replacement_id in index_by_id and not already_replaced:
            raise ValueError(f"replacement id already exists: {replacement_id}")
        repaired[index] = row
        index_by_id.pop(source_id, None)
        index_by_id[replacement_id] = index

    return repaired


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    repaired = apply_corrections(rows, load_corrections(args.corrections))
    write_jsonl_atomic(args.input, repaired)
    print(json.dumps({"path": str(args.input.resolve()), "count": len(repaired)}, ensure_ascii=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    return parser


if __name__ == "__main__":
    main()
