"""Approve a release baseline only after every protected quality gate passes.

This command is intentionally explicit: it consumes machine-generated reports,
recomputes the implementation fingerprint, and refuses to write an approved
baseline when any required suite is incomplete or failed.  It never edits the
evaluation reports themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.retrieval_contract import (  # noqa: E402
    build_llm_runtime_identity,
    normalize_llm_runtime_identity,
    normalize_sha256,
)
from app.utils.strict_dotenv import load_release_env_file  # noqa: E402

CURRENT_SCHEMA_VERSION = 9
EXPECTED_EVALUATION_ENGINE = "industrial-rag-native-text-judge"
EXPECTED_EVALUATION_ENGINE_VERSION = "1.1"
PROTECTED_DATASETS = {
    "legal_expanded_240.jsonl": 240,
    "legal_expanded_ragas_40.jsonl": 40,
    "legal_holdout_150.jsonl": 150,
}

if TYPE_CHECKING:
    from scripts.validate_release_baseline import (
        HASH_CONTRACT,
        canonical_file_sha256,
        implementation_sha256,
        validate,
    )
else:
    _validator = importlib.import_module(
        "scripts.validate_release_baseline" if __package__ else "validate_release_baseline"
    )
    HASH_CONTRACT: str = _validator.HASH_CONTRACT
    canonical_file_sha256: Callable[[Path], str] = _validator.canonical_file_sha256
    implementation_sha256: Callable[[Path, list[str]], str] = _validator.implementation_sha256
    validate: Callable[..., dict[str, Any]] = _validator.validate


def _require_passed(report: dict[str, Any], name: str) -> None:
    if report.get("passed") is not True:
        raise ValueError(f"{name} did not pass")
    checks = report.get("checks") or {}
    failed = [key for key, check in checks.items() if check.get("status") != "passed"]
    if failed:
        raise ValueError(f"{name} contains non-passing checks: {', '.join(sorted(failed))}")


def _require_count(report: dict[str, Any], field: str, expected: int, name: str) -> None:
    actual = int(report.get(field) or 0)
    if actual != expected:
        raise ValueError(f"{name} must contain {expected} samples; found {actual}")


def _require_matching_retrieval_runtime_identity(
    expanded_gate: dict[str, Any],
    holdout_gate: dict[str, Any],
    production_identity: dict[str, Any] | None = None,
    ragas_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expanded_identity = normalize_llm_runtime_identity(
        expanded_gate.get("retrieval_llm_runtime_identity")
    )
    holdout_identity = normalize_llm_runtime_identity(
        holdout_gate.get("retrieval_llm_runtime_identity")
    )
    if expanded_identity is None or holdout_identity is None:
        raise ValueError("retrieval quality gates are missing a valid LLM runtime identity")
    if expanded_identity != holdout_identity:
        raise ValueError("retrieval quality gates used different LLM runtime identities")
    if ragas_gate is not None:
        ragas_identity = normalize_llm_runtime_identity(
            ragas_gate.get("retrieval_llm_runtime_identity")
        )
        if ragas_identity is None:
            raise ValueError("native judge quality gate is missing a valid retrieval LLM identity")
        if expanded_identity != ragas_identity:
            raise ValueError("quality gates used different retrieval LLM runtime identities")
    if production_identity is not None:
        normalized_production = normalize_llm_runtime_identity(production_identity)
        if normalized_production is None:
            raise ValueError("production environment has an invalid LLM runtime identity")
        if expanded_identity != normalized_production:
            raise ValueError(
                "production environment does not match the evaluated LLM runtime identity"
            )
    return expanded_identity


def _require_matching_retrieval_runtime_contract(
    expanded_gate: dict[str, Any],
    holdout_gate: dict[str, Any],
    ragas_gate: dict[str, Any],
) -> str:
    """Require every protected retrieval suite to use the same effective config."""
    digests = [
        normalize_sha256(report.get("retrieval_runtime_contract_sha256"))
        for report in (expanded_gate, holdout_gate, ragas_gate)
    ]
    if any(digest is None for digest in digests):
        raise ValueError("quality gates are missing a valid retrieval runtime contract")
    if len(set(digests)) != 1:
        raise ValueError("quality gates used different retrieval runtime contracts")
    return str(digests[0])


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _protected_dataset_contract(
    project_root: Path,
    *,
    ragas_gate: dict[str, Any],
    expanded_gate: dict[str, Any],
    holdout_gate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    reported_hashes = {
        "legal_expanded_240.jsonl": normalize_sha256(
            expanded_gate.get("retrieval_input_sha256")
        ),
        "legal_expanded_ragas_40.jsonl": normalize_sha256(
            ragas_gate.get("source_dataset_sha256")
        ),
        "legal_holdout_150.jsonl": normalize_sha256(
            holdout_gate.get("retrieval_input_sha256")
        ),
    }
    contract: dict[str, dict[str, Any]] = {}
    for name, expected_count in PROTECTED_DATASETS.items():
        path = project_root / "eval" / name
        if not path.is_file():
            raise ValueError(f"protected evaluation dataset is missing: {name}")
        actual_hash = canonical_file_sha256(path)
        accepted_report_hashes = {actual_hash, _raw_sha256(path)}
        if reported_hashes[name] not in accepted_report_hashes:
            raise ValueError(f"quality gate is not bound to the current dataset: {name}")
        actual_count = _sample_count(path)
        if actual_count != expected_count:
            raise ValueError(
                f"protected evaluation dataset must contain {expected_count} samples: {name}"
            )
        contract[name] = {"sha256": actual_hash, "sample_count": actual_count}
    return contract


def _production_runtime_identity(env_path: Path) -> dict[str, Any]:
    values = load_release_env_file(env_path)
    return build_llm_runtime_identity(
        provider="openai_compatible",
        model_name=values.get("DEEPSEEK_MODEL", ""),
        base_url=values.get("DEEPSEEK_API_URL", ""),
    )


def _require_judge_count(report: dict[str, Any], expected: int) -> None:
    actual = int(report.get("judge_sample_count") or report.get("ragas_sample_count") or 0)
    if actual != expected:
        raise ValueError(
            f"native judge quality gate must contain {expected} samples; found {actual}"
        )


def _copy_numeric_checks(
    destination: dict[str, dict[str, Any]],
    source: dict[str, Any],
    names: tuple[str, ...],
    *,
    prefix: str = "",
) -> None:
    for name in names:
        check = source.get(name)
        if not isinstance(check, dict) or check.get("value") is None:
            raise ValueError(f"required check is missing from protected report: {name}")
        value = check["value"]
        if isinstance(value, bool):
            raise ValueError(f"numeric check has a boolean value: {name}")
        destination[f"{prefix}{name}"] = {
            "value": float(value),
            "minimum": float(check.get("minimum", 0)),
        }


def approve(
    baseline_path: Path,
    project_root: Path,
    *,
    ragas_gate_path: Path,
    expanded_gate_path: Path,
    holdout_gate_path: Path,
    production_env_path: Path,
) -> dict[str, Any]:
    baseline = cast(dict[str, Any], json.loads(baseline_path.read_text(encoding="utf-8")))
    ragas_gate = cast(dict[str, Any], json.loads(ragas_gate_path.read_text(encoding="utf-8")))
    expanded_gate = cast(dict[str, Any], json.loads(expanded_gate_path.read_text(encoding="utf-8")))
    holdout_gate = cast(dict[str, Any], json.loads(holdout_gate_path.read_text(encoding="utf-8")))

    _require_passed(ragas_gate, "native judge quality gate")
    _require_passed(expanded_gate, "expanded retrieval quality gate")
    _require_passed(holdout_gate, "holdout quality gate")
    _require_judge_count(ragas_gate, 40)
    _require_count(expanded_gate, "full_suite_sample_count", 240, "expanded retrieval gate")
    representative_count = int(
        expanded_gate.get("representative_judge_sample_count")
        or expanded_gate.get("representative_ragas_sample_count")
        or 0
    )
    if representative_count != 40:
        raise ValueError(
            "expanded retrieval gate must contain 40 representative judge samples; "
            f"found {representative_count}"
        )
    _require_count(holdout_gate, "sample_count", 150, "holdout quality gate")
    retrieval_runtime_identity = _require_matching_retrieval_runtime_identity(
        expanded_gate,
        holdout_gate,
        production_identity=_production_runtime_identity(production_env_path),
        ragas_gate=ragas_gate,
    )
    retrieval_runtime_contract_sha256 = _require_matching_retrieval_runtime_contract(
        expanded_gate,
        holdout_gate,
        ragas_gate,
    )
    dataset_contract = _protected_dataset_contract(
        project_root,
        ragas_gate=ragas_gate,
        expanded_gate=expanded_gate,
        holdout_gate=holdout_gate,
    )

    implementation = baseline.setdefault("implementation", {})
    paths = implementation.get("paths") or []
    if not paths:
        raise ValueError("release baseline has no implementation paths")
    implementation["sha256"] = implementation_sha256(project_root, paths)

    evaluation_engine = str(ragas_gate.get("evaluation_engine") or "")
    evaluation_engine_version = str(ragas_gate.get("evaluation_engine_version") or "")
    judge_model = str(ragas_gate.get("judge_model") or "").strip()
    if evaluation_engine != EXPECTED_EVALUATION_ENGINE:
        raise ValueError("native judge quality gate has an unexpected evaluation engine")
    if evaluation_engine_version != EXPECTED_EVALUATION_ENGINE_VERSION:
        raise ValueError("native judge quality gate has an unexpected evaluation engine version")
    if not judge_model:
        raise ValueError("native judge quality gate is missing judge_model")
    answer_generation_policy_sha256 = normalize_sha256(
        ragas_gate.get("answer_generation_policy_sha256")
    )
    judge_request_policy_sha256 = normalize_sha256(
        ragas_gate.get("judge_request_policy_sha256")
    )
    if answer_generation_policy_sha256 is None:
        raise ValueError("native judge quality gate is missing answer generation policy")
    if judge_request_policy_sha256 is None:
        raise ValueError("native judge quality gate is missing judge request policy")
    if int(ragas_gate.get("top_k") or 0) != 5:
        raise ValueError("native judge quality gate must use retrieval top_k=5")

    baseline["schema_version"] = CURRENT_SCHEMA_VERSION
    baseline["hash_contract"] = HASH_CONTRACT
    baseline["datasets"] = dataset_contract
    contract = baseline.setdefault("evaluation_contract", {})
    contract.pop("ragas_version", None)
    contract.update(
        {
            "retrieval_top_k": 5,
            "evaluation_engine": evaluation_engine,
            "evaluation_engine_version": evaluation_engine_version,
            "judge_model": judge_model,
            "retrieval_llm_runtime_identity": retrieval_runtime_identity,
            "retrieval_runtime_contract_sha256": retrieval_runtime_contract_sha256,
            "answer_generation_policy_sha256": answer_generation_policy_sha256,
            "judge_request_policy_sha256": judge_request_policy_sha256,
            "evaluation_lock": "requirements-evaluation.lock.txt",
        }
    )

    checks: dict[str, dict[str, Any]] = {}
    _copy_numeric_checks(
        checks,
        ragas_gate.get("checks") or {},
        (
            "answer_accuracy",
            "faithfulness",
            "context_precision",
            "context_recall",
            "citation_recall",
            "citation_mrr",
        ),
    )
    _copy_numeric_checks(
        checks,
        expanded_gate.get("checks") or {},
        (
            "in_domain_citation_recall",
            "adversarial_citation_recall",
            "comparison_citation_recall",
            "no_answer_abstention_rate",
        ),
    )
    _copy_numeric_checks(
        checks,
        holdout_gate.get("checks") or {},
        ("citation_recall", "citation_mrr"),
        prefix="holdout_",
    )

    baseline["status"] = "approved"
    baseline["blocked_reasons"] = []
    baseline["evaluated_at"] = datetime.now(UTC).isoformat()
    baseline["checks"] = checks
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("eval/release_baseline.json"))
    parser.add_argument("--ragas-gate", type=Path, required=True)
    parser.add_argument("--expanded-gate", type=Path, required=True)
    parser.add_argument("--holdout-gate", type=Path, required=True)
    parser.add_argument("--production-env", type=Path, required=True)
    parser.add_argument(
        "--write", action="store_true", help="Atomically replace the baseline after validation"
    )
    args = parser.parse_args()

    baseline_path = args.baseline.resolve()
    ragas_gate_path = args.ragas_gate.resolve()
    expanded_gate_path = args.expanded_gate.resolve()
    holdout_gate_path = args.holdout_gate.resolve()
    production_env_path = args.production_env.resolve()
    project_root = baseline_path.parents[1]
    candidate = approve(
        baseline_path,
        project_root,
        ragas_gate_path=ragas_gate_path,
        expanded_gate_path=expanded_gate_path,
        holdout_gate_path=holdout_gate_path,
        production_env_path=production_env_path,
    )

    if not args.write:
        print(json.dumps({"approved_candidate": True, "status": candidate["status"]}, indent=2))
        return

    candidate_path = baseline_path.with_name(f"{baseline_path.name}.candidate")
    try:
        candidate_path.write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        validate(candidate_path, project_root)
        candidate_path.replace(baseline_path)
    finally:
        candidate_path.unlink(missing_ok=True)

    print(json.dumps({"approved": True, "baseline": str(baseline_path)}, indent=2))


if __name__ == "__main__":
    main()
