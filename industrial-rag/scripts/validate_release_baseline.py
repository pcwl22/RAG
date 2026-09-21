"""Validate that an approved RAG baseline matches datasets and implementation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.retrieval_contract import (  # noqa: E402
    normalize_llm_runtime_identity,
    normalize_sha256,
)

EVAL_DIR = PROJECT_ROOT / "eval"
BASELINE_PATH = EVAL_DIR / "release_baseline.json"
CURRENT_SCHEMA_VERSION = 9
HASH_CONTRACT = "sha256-canonical-text-v1"
EXPECTED_EVALUATION_ENGINE = "industrial-rag-native-text-judge"
EXPECTED_EVALUATION_ENGINE_VERSION = "1.1"
REQUIRED_DATASETS = {
    "legal_expanded_240.jsonl": 240,
    "legal_expanded_ragas_40.jsonl": 40,
    "legal_holdout_150.jsonl": 150,
}
REQUIRED_IMPLEMENTATION_PATHS = frozenset(
    {
        "app/embedding/model_bundle.py",
        "constraints-docker.txt",
        "docker/api/Dockerfile.production",
        "docker/model-bundle/Dockerfile.production",
        "docker/model-bundle/model-manifest.example.json",
        "docker/worker/Dockerfile.production",
        "pyproject.toml",
        "requirements-evaluation.lock.txt",
        "requirements-evaluation.txt",
        "requirements-gpu-verified.txt",
        "requirements-gpu.lock.txt",
        "requirements-runtime.lock.txt",
        "requirements-torch-cpu.lock.txt",
        "requirements-torch-cu126.lock.txt",
        "scripts/audit_dependency_locks.py",
        "scripts/approve_release_baseline.py",
        "scripts/build_model_bundle_manifest.py",
        "scripts/check_expanded_quality_gate.py",
        "scripts/check_holdout_quality_gate.py",
        "scripts/check_quality_gate.py",
        "scripts/evaluate_ragas.py",
        "scripts/evaluate_retrieval_suite.py",
        "scripts/export_lrage.py",
        "scripts/reembed_postgres_corpus.py",
        "scripts/score_retrieval.py",
        "scripts/validate_evaluation_assets.py",
        "scripts/validate_dependency_lock.py",
        "scripts/validate_model_bundle_manifest.py",
        "scripts/validate_release_baseline.py",
        "security/evaluation_dependency_exceptions.toml",
    }
)


def canonical_file_bytes(path: Path) -> bytes:
    """Return portable bytes for a text file and unchanged bytes for binary data."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def canonical_file_sha256(path: Path) -> str:
    """Hash a file under the release baseline's cross-platform byte contract."""
    return hashlib.sha256(canonical_file_bytes(path)).hexdigest()


def _sample_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _citation_variants(citation: str) -> set[str]:
    """Return article-number spellings that must not appear in a question."""
    citation = str(citation).strip()
    if not citation:
        return set()
    variants = {citation}
    core = citation.removeprefix("第")
    numeral = ""
    if "条" in core:
        numeral, suffix = core.split("条", 1)
        article_suffix = f"条{suffix}"
        variants.update({f"第{numeral}{article_suffix}", f"{numeral}{article_suffix}"})
    elif core:
        numeral = core
        variants.update({f"第{numeral}条", f"{numeral}条"})
    digits = "零一二三四五六七八九"
    units = {"十": 10, "百": 100, "千": 1000}
    if numeral and all(char in digits or char in units for char in numeral):
        section = 0
        current = 0
        for char in numeral.split("之", 1)[0]:
            if char in digits:
                current = digits.index(char)
            else:
                section += (current or 1) * units[char]
                current = 0
        number = section + current
        if number:
            variants.update({str(number), f"第{number}条", f"{number}条"})
    return {variant for variant in variants if variant}


def _citation_leak_ids(path: Path) -> list[str]:
    leaked: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            query = str(row.get("query") or "")
            citations = row.get("expected_citations") or []
            if any(
                variant in query
                for citation in citations
                for variant in _citation_variants(str(citation))
            ):
                leaked.append(str(row.get("id") or line_number))
    return leaked


def _implementation_files(project_root: Path, paths: list[str]) -> list[Path]:
    files: set[Path] = set()
    for configured_path in paths:
        path = (project_root / configured_path).resolve()
        try:
            path.relative_to(project_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"implementation path escapes project root: {configured_path}"
            ) from exc
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
        digest.update(canonical_file_bytes(path))
    return digest.hexdigest()


def validate(
    baseline_path: Path = BASELINE_PATH,
    project_root: Path = PROJECT_ROOT,
    *,
    allow_blocked: bool = False,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    schema_version = int(baseline.get("schema_version", 0))
    if schema_version != CURRENT_SCHEMA_VERSION:
        errors.append(f"release baseline schema_version must equal {CURRENT_SCHEMA_VERSION}")
    if baseline.get("hash_contract") != HASH_CONTRACT:
        errors.append(f"release baseline hash_contract must equal {HASH_CONTRACT}")

    datasets = baseline.get("datasets", {})
    if not isinstance(datasets, dict) or set(datasets) != set(REQUIRED_DATASETS):
        errors.append("release baseline must bind exactly the three protected evaluation datasets")
        datasets = datasets if isinstance(datasets, dict) else {}
    for name, expected in datasets.items():
        path = project_root / "eval" / name
        if not path.is_file():
            errors.append(f"missing dataset: {name}")
            continue
        if canonical_file_sha256(path) != expected.get("sha256"):
            errors.append(f"dataset hash changed: {name}; rerun and approve the release evaluation")
        if _sample_count(path) != int(expected.get("sample_count", -1)):
            errors.append(f"dataset sample count changed: {name}")
        if name in REQUIRED_DATASETS and int(expected.get("sample_count", -1)) != REQUIRED_DATASETS[name]:
            errors.append(
                f"release baseline dataset sample count must equal {REQUIRED_DATASETS[name]}: {name}"
            )
        try:
            leaked = _citation_leak_ids(path)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if leaked:
                errors.append(
                    f"dataset contains citation leakage: {name} "
                    f"({len(leaked)} cases; examples: {', '.join(leaked[:5])})"
                )

    if schema_version >= 3:
        if baseline.get("status") != "approved" and not allow_blocked:
            errors.append(
                "release baseline status is not approved; complete a real leakage-free evaluation"
            )
        contract = baseline.get("evaluation_contract") or {}
        holdout_name = contract.get("holdout_dataset")
        if holdout_name != "legal_holdout_150.jsonl":
            errors.append(
                "release baseline evaluation_contract must bind the holdout_dataset "
                "to legal_holdout_150.jsonl"
            )
        if holdout_name not in baseline.get("datasets", {}):
            errors.append("release baseline must include the holdout dataset")
        if contract.get("citation_leakage_policy") != "reject":
            errors.append(
                "release baseline evaluation_contract must set citation_leakage_policy=reject"
            )

    for name, check in baseline.get("checks", {}).items():
        if float(check.get("value", -1)) < float(check.get("minimum", 0)):
            errors.append(f"approved baseline check is below threshold: {name}")

    implementation = baseline.get("implementation") or {}
    implementation_paths = implementation.get("paths") or []
    expected_implementation_hash = implementation.get("sha256")
    if not implementation_paths or not expected_implementation_hash:
        errors.append("release baseline must include implementation paths and sha256")
    else:
        missing_paths = sorted(REQUIRED_IMPLEMENTATION_PATHS - set(implementation_paths))
        if missing_paths:
            errors.append(
                "release baseline implementation fingerprint omits required paths: "
                + ", ".join(missing_paths)
            )
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
    for field in (
        "retrieval_top_k",
        "evaluation_engine",
        "evaluation_engine_version",
        "judge_model",
        "answer_generation_policy_sha256",
        "judge_request_policy_sha256",
    ):
        if not contract.get(field):
            errors.append(f"release baseline evaluation_contract is missing {field}")
    if contract.get("evaluation_engine") != EXPECTED_EVALUATION_ENGINE:
        errors.append("release baseline evaluation_contract has an unexpected evaluation_engine")
    if contract.get("evaluation_engine_version") != EXPECTED_EVALUATION_ENGINE_VERSION:
        errors.append(
            "release baseline evaluation_contract has an unexpected evaluation_engine_version"
        )
    if "ragas_version" in contract:
        errors.append("release baseline evaluation_contract must not contain ragas_version")
    if contract.get("evaluation_lock") != "requirements-evaluation.lock.txt":
        errors.append(
            "release baseline evaluation_contract must bind requirements-evaluation.lock.txt"
        )
    if (
        baseline.get("status") == "approved"
        and normalize_llm_runtime_identity(contract.get("retrieval_llm_runtime_identity")) is None
    ):
        errors.append("approved release baseline must bind a valid retrieval LLM runtime identity")
    if (
        baseline.get("status") == "approved"
        and normalize_sha256(contract.get("retrieval_runtime_contract_sha256")) is None
    ):
        errors.append(
            "approved release baseline must bind a valid retrieval runtime contract"
        )
    if (
        baseline.get("status") == "approved"
        and normalize_sha256(contract.get("answer_generation_policy_sha256")) is None
    ):
        errors.append("approved release baseline must bind a valid answer generation policy")
    if (
        baseline.get("status") == "approved"
        and normalize_sha256(contract.get("judge_request_policy_sha256")) is None
    ):
        errors.append("approved release baseline must bind a valid judge request policy")

    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "passed": True,
        "baseline": str(baseline_path.relative_to(project_root)),
        "dataset_count": len(baseline["datasets"]),
        "check_count": len(baseline["checks"]),
        "hash_contract": HASH_CONTRACT,
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
        print(
            json.dumps(
                validate(allow_blocked="--allow-blocked" in sys.argv),
                ensure_ascii=False,
                indent=2,
            )
        )
