"""Create the immutable descriptor for runtime Hugging Face downloads.

This command deliberately reads no model weights.  It emits only approved
repository URLs, immutable Hugging Face commit revisions, and pre-approved
deterministic tree hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HUGGING_FACE_HUB = "https://huggingface.co"

MODEL_SPECS = {
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "sha256": "b1634ec79b0cd6e439ac1471698e15e6a23ea0365908a1c532a8e94b8c048c55",
    },
    "bge-reranker-v2-m3": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "sha256": "5ed918e794d4ce634a4e4fa87e8d0c0436c73176d09ff7531fca5920511aa474",
    },
}


def build_manifest() -> dict[str, object]:
    models: dict[str, dict[str, str]] = {}
    for name, specification in MODEL_SPECS.items():
        model_id = specification["model_id"]
        revision = specification["revision"]
        source_url = f"{HUGGING_FACE_HUB}/{model_id}"
        models[name] = {
            "model_id": model_id,
            "revision": revision,
            "source_url": source_url,
            "download_url": f"{source_url}/tree/{revision}",
            "path": name,
            "sha256": specification["sha256"],
        }
    return {"schema_version": 2, "hub": HUGGING_FACE_HUB, "models": models}


def write_manifest(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("model-manifest.json"))
    args = parser.parse_args()
    write_manifest(args.output)
    print(f"Wrote model source manifest: {args.output}")


if __name__ == "__main__":
    main()
