"""Create a deterministic, non-production model bundle for Dockerfile CI builds."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.build_model_bundle_manifest import MODEL_SPECS, write_manifest
else:
    _builder = importlib.import_module(
        "scripts.build_model_bundle_manifest" if __package__ else "build_model_bundle_manifest"
    )
    MODEL_SPECS: Mapping[str, object] = _builder.MODEL_SPECS
    write_manifest: Callable[[Path, Path], None] = _builder.write_manifest


def create_fixture(output: Path) -> str:
    """Create the minimal tree expected by the production image contract."""
    root = output.resolve()
    models = root / "models"
    for name in MODEL_SPECS:
        model = models / name
        model.mkdir(parents=True, exist_ok=True)
        (model / "config.json").write_text(
            '{"ci_fixture":true,"not_for_inference":true}\n',
            encoding="utf-8",
            newline="\n",
        )
        (model / "weights.ci-placeholder").write_bytes(
            f"industrial-rag-ci-fixture:{name}\n".encode()
        )

    manifest = root / "model-manifest.json"
    write_manifest(models, manifest)
    return "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--github-output",
        type=Path,
        default=Path(os.environ["GITHUB_OUTPUT"]) if os.getenv("GITHUB_OUTPUT") else None,
    )
    args = parser.parse_args()
    digest = create_fixture(args.output)
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"manifest_sha256={digest}\n")
    print(digest)


if __name__ == "__main__":
    main()
