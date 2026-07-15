"""Create a deterministic category/domain-balanced sample from an evaluation JSONL."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DOMAIN_QUOTAS = {"civil": 4, "criminal": 3, "labor": 3}


def sample_cases(cases: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for case in cases:
        metadata = case.get("metadata") or {}
        grouped[(metadata.get("category", ""), metadata.get("domain", ""))].append(case)

    selected: list[dict] = []
    for category in ("standard", "adversarial", "comparison"):
        for domain, quota in DOMAIN_QUOTAS.items():
            items = grouped[(category, domain)]
            if len(items) < quota:
                raise ValueError(f"Need {quota} {category}/{domain} cases, found {len(items)}")
            selected.extend(items[:quota])
    no_answer = grouped[("no_answer", "out_of_scope")]
    if len(no_answer) < 10:
        raise ValueError("Need at least 10 no-answer cases")
    selected.extend(no_answer[:10])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = sample_cases(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in selected),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "sample_count": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
