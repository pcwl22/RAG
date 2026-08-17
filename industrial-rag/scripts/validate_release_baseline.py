"""Validate that an approved RAG baseline matches datasets and implementation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval"
BASELINE_PATH = EVAL_DIR / "release_baseline.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _implementation_files(project_root: Path, paths: list[str]) -> list[Path]:
    files: set[Path] = set()
    for configured_path in paths:
        path = (project_root / configured_path).resolve()
        try:
            path.relative_to(project_root.resolve())
        except ValueError as exc:
            raise ValueError(f"implementation path escapes project root: {configured_path}") from exc
        if path.is_file():
            files.add(path)
            continue
        if path.is_dir():
            files.update(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix not in {".pyc", ".pyo"}
            )
            continue
        raise FileNotFoundError(f"missing implementation path: {configured_path}")
    return sorted(files, key=lambda item: item.relative_to(project_root).as_posix())


def implementation_sha256(project_root: Path, paths: list[str]) -> str:
    """Hash implementation paths deterministically, including relative filenames."""
    digest = hashlib.sha256()
    for path in _implementation_files(project_root, paths):
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate(
    baseline_path: Path = BASELINE_PATH,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if int(baseline.get("schema_version", 0)) < 2:
        errors.append("release baseline schema_version must be at least 2")

    for name, expected in baseline.get("datasets", {}).items():
        path = project_root / "eval" / name
        if not path.is_file():
            errors.append(f"missing dataset: {name}")
            continue
        if _sha256(path) != expected.get("sha256"):
            errors.append(f"dataset hash changed: {name}; rerun and approve the release evaluation")
        if _sample_count(path) != int(expected.get("sample_count", -1)):
            errors.append(f"dataset sample count changed: {name}")

    for name, check in baseline.get("checks", {}).items():
        if float(check.get("value", -1)) < float(check.get("minimum", 0)):
            errors.append(f"approved baseline check is below threshold: {name}")

    implementation = baseline.get("implementation") or {}
    implementation_paths = implementation.get("paths") or []
    expected_implementation_hash = implementation.get("sha256")
    if not implementation_paths or not expected_implementation_hash:
        errors.append("release baseline must include implementation paths and sha256")
    else:
        try:
            actual_hash = implementation_sha256(project_root, implementation_paths)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if actual_hash != expected_implementation_hash:
                errors.append(
                    "RAG implementation changed; rerun the release evaluation and approve "
                    "a new implementation sha256"
                )

    contract = baseline.get("evaluation_contract") or {}
    for field in ("retrieval_top_k", "ragas_version", "judge_model"):
        if not contract.get(field):
            errors.append(f"release baseline evaluation_contract is missing {field}")

    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "passed": True,
        "baseline": str(baseline_path.relative_to(project_root)),
        "dataset_count": len(baseline["datasets"]),
        "check_count": len(baseline["checks"]),
        "implementation_sha256": expected_implementation_hash,
    }


if __name__ == "__main__":
    if "--print-implementation-sha256" in sys.argv:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        paths = (baseline.get("implementation") or {}).get("paths") or []
        if not paths:
            raise SystemExit("release baseline has no implementation paths")
        print(implementation_sha256(PROJECT_ROOT, paths))
    else:
        print(json.dumps(validate(), ensure_ascii=False, indent=2))
