"""Select Kubernetes objects from a rendered multi-document manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def select(
    input_path: Path,
    output_path: Path,
    *,
    kinds: set[str] | None = None,
    names: set[str] | None = None,
    exclude_kinds: set[str] | None = None,
) -> int:
    documents = [
        item
        for item in yaml.safe_load_all(input_path.read_text(encoding="utf-8"))
        if isinstance(item, dict) and item.get("kind")
    ]
    selected: list[dict[str, Any]] = []
    for item in documents:
        kind = str(item.get("kind"))
        name = str((item.get("metadata") or {}).get("name"))
        if kinds and kind not in kinds:
            continue
        if names and name not in names:
            continue
        if exclude_kinds and kind in exclude_kinds:
            continue
        selected.append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "---\n".join(
            yaml.safe_dump(item, sort_keys=False, allow_unicode=True) for item in selected
        ),
        encoding="utf-8",
        newline="\n",
    )
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", dest="kinds", action="append")
    parser.add_argument("--name", dest="names", action="append")
    parser.add_argument("--exclude-kind", dest="exclude_kinds", action="append")
    args = parser.parse_args()
    count = select(
        args.input,
        args.output,
        kinds=set(args.kinds or []),
        names=set(args.names or []),
        exclude_kinds=set(args.exclude_kinds or []),
    )
    print(f"Selected {count} Kubernetes resources.")


if __name__ == "__main__":
    main()
