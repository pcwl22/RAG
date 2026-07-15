"""Score retrieval recall and reciprocal rank from an LRAGE export JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _context_text(context: dict) -> str:
    values = [
        context.get("citation"),
        context.get("article"),
        context.get("source"),
        context.get("title"),
        context.get("text"),
    ]
    return " ".join(str(value) for value in values if value)


def score(path: Path, top_k: int) -> dict[str, float | int]:
    evaluated = 0
    evaluated_citations = 0
    hits = 0
    reciprocal_rank = 0.0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            expected = sample.get("metadata", {}).get("expected_citations") or []
            if not expected:
                continue
            evaluated += 1
            evaluated_citations += len(expected)
            contexts = sample.get("contexts", [])[:top_k]
            for citation in expected:
                rank = next(
                    (
                        index
                        for index, context in enumerate(contexts, 1)
                        if str(citation) in _context_text(context)
                    ),
                    None,
                )
                if rank is not None:
                    hits += 1
                    reciprocal_rank += 1.0 / rank

    return {
        "evaluated_cases": evaluated,
        "evaluated_citations": evaluated_citations,
        f"recall@{top_k}": hits / evaluated_citations if evaluated_citations else 0.0,
        f"mrr@{top_k}": reciprocal_rank / evaluated_citations if evaluated_citations else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(score(args.export, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
