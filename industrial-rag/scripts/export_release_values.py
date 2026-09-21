"""Export immutable values from one strict protected release snapshot."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.validate_production_env import (
        is_immutable_image_reference,
        load_release_env_file,
    )
else:
    _validator = importlib.import_module(
        "scripts.validate_production_env" if __package__ else "validate_production_env"
    )
    is_immutable_image_reference: Callable[[str], bool] = (
        _validator.is_immutable_image_reference
    )
    load_release_env_file: Callable[[Path], dict[str, str]] = (
        _validator.load_release_env_file
    )

RELEASE_OUTPUTS = {
    "api": "API_IMAGE",
    "worker": "WORKER_IMAGE",
    "frontend": "FRONTEND_IMAGE",
    "model_bundle": "MODEL_BUNDLE_IMAGE",
    "model_bundle_digest": "MODEL_BUNDLE_DIGEST",
    "model_manifest_sha256": "MODEL_MANIFEST_SHA256",
    "ingress_host": "INGRESS_HOST",
}


def release_outputs(env_file: Path) -> dict[str, str]:
    values = load_release_env_file(env_file)
    outputs: dict[str, str] = {}
    for output_name, source_name in RELEASE_OUTPUTS.items():
        value = values.get(source_name, "").strip()
        if not value:
            raise ValueError(f"{source_name} is required for release image verification")
        if "\n" in value or "\r" in value:
            raise ValueError(f"{source_name} must be a single-line value")
        if source_name.endswith("_IMAGE") and not is_immutable_image_reference(value):
            raise ValueError(f"{source_name} must be a safe immutable OCI image reference")
        outputs[output_name] = value
    return outputs


def write_github_outputs(env_file: Path, output_file: Path) -> None:
    outputs = release_outputs(env_file)
    with output_file.open("a", encoding="utf-8", newline="\n") as stream:
        for name, value in outputs.items():
            stream.write(f"{name}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    write_github_outputs(args.env_file, args.github_output)
    print("Exported strict release image references.")


if __name__ == "__main__":
    main()
