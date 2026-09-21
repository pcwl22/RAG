"""Create the immutable model-bundle manifest from protected model files."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.validate_model_bundle_manifest import tree_sha256
else:
    _validator_module = importlib.import_module(
        "scripts.validate_model_bundle_manifest"
        if __package__
        else "validate_model_bundle_manifest"
    )
    tree_sha256: Callable[[Path], str] = _validator_module.tree_sha256

MODEL_SPECS = {
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "revision": "bge-m3-local-2026-06",
    },
    "bge-reranker-v2-m3": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "revision": "bge-reranker-v2-m3-local-2026-06",
    },
}


def build_manifest(models_root: Path) -> dict[str, object]:
    root = models_root.resolve()
    models: dict[str, dict[str, str]] = {}
    for name, specification in MODEL_SPECS.items():
        model_path = (root / name).resolve()
        if root not in model_path.parents or not model_path.is_dir():
            raise FileNotFoundError(f"missing model directory: {model_path}")
        models[name] = {
            **specification,
            "path": name,
            "sha256": tree_sha256(model_path),
        }
    return {"schema_version": 1, "models": models}


def write_manifest(models_root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_manifest(models_root), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("model-manifest.json"))
    args = parser.parse_args()
    write_manifest(args.models_root, args.output)
    print(f"Wrote model bundle manifest: {args.output}")


if __name__ == "__main__":
    main()
