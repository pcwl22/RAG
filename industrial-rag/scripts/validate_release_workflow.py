"""Fail-closed semantic validation for the protected release workflow.

The workflow is security-sensitive configuration.  Textual review alone is
too easy to bypass when a job is renamed, a dependency edge is removed, or a
secret is accidentally made available to an evaluation dependency.  This
validator checks both the parsed job graph and a small set of required shell
invariants.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

CHECKOUT_ACTION = "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
REQUIRED_JOBS = ("preflight", "prepare-evaluation", "ragas-judge", "deploy")
EXPECTED_NEEDS = {
    "preflight": set(),
    "prepare-evaluation": {"preflight"},
    "ragas-judge": {"prepare-evaluation"},
    "deploy": {"preflight", "prepare-evaluation", "ragas-judge"},
}
EXPECTED_SECRETS = {
    "preflight": set(),
    "prepare-evaluation": {"RAG_EVALUATION_ENV_B64"},
    "ragas-judge": {"RAGAS_JUDGE_API_KEY"},
    "deploy": {"RAG_PRODUCTION_ENV_B64"},
}
EXPECTED_ENVIRONMENTS = {
    "prepare-evaluation": "production-evaluation-source",
    "ragas-judge": "ragas-judge",
    "deploy": "production",
}
EXPECTED_RUNNER_LABELS = {
    "preflight": {"ubuntu-latest"},
    "prepare-evaluation": {"self-hosted", "rag-evaluation-source"},
    "ragas-judge": {"ubuntu-latest"},
    "deploy": {"self-hosted", "rag-production"},
}
REGISTRY_VERIFICATION_STEPS = {
    "prepare-evaluation": "Verify candidate signatures and source binding before execution",
    "deploy": "Verify signed production images",
}
SECRET_REFERENCE = re.compile(
    r"\$\{\{\s*secrets\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}"
)
FULL_SHA_ACTION = re.compile(r"^[^./\s]+/[^@\s]+@[0-9a-f]{40}$")
APPROVED_JOB_INSTALLS = {
    "preflight": "python -m pip install --require-hashes -r requirements-ci.lock.txt",
    "prepare-evaluation": "python -m pip install --require-hashes -r requirements-ci.lock.txt",
    "ragas-judge": (
        "python -m pip install --require-hashes -r requirements-evaluation.lock.txt"
    ),
    "deploy": "python -m pip install --require-hashes -r requirements-ci.lock.txt",
}


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def _job_text(job: dict[str, Any]) -> str:
    return yaml.safe_dump(job, sort_keys=False)


def _needs(job: dict[str, Any]) -> set[str]:
    value = job.get("needs")
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)}


def _runner_labels(job: dict[str, Any]) -> set[str]:
    value = job.get("runs-on")
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)} if value is not None else set()


def _environment_name(job: dict[str, Any]) -> str:
    value = job.get("environment")
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return str(value or "")


def _step(job: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((item for item in _as_steps(job) if item.get("name") == name), None)


def _validate_permissions(
    scope: str,
    permissions: Any,
    errors: list[str],
    *,
    packages_read_allowed: bool = False,
) -> None:
    values = _as_mapping(permissions)
    if values.get("contents") != "read":
        errors.append(f"{scope} permissions must set contents: read")
    if values.get("id-token") != "none":
        errors.append(f"{scope} permissions must set id-token: none")
    allowed = {"contents", "id-token"}
    if packages_read_allowed:
        allowed.add("packages")
        if values.get("packages") != "read":
            errors.append(f"{scope} permissions must set packages: read")
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        errors.append(f"{scope} permissions contain unexpected grants: {', '.join(unexpected)}")


def _validate_checkout(job_name: str, job: dict[str, Any], errors: list[str]) -> None:
    checkout_steps = [
        step for step in _as_steps(job) if str(step.get("uses") or "").startswith("actions/checkout@")
    ]
    if len(checkout_steps) != 1:
        errors.append(f"{job_name} must contain exactly one checkout step")
        return
    checkout = checkout_steps[0]
    if checkout.get("uses") != CHECKOUT_ACTION:
        errors.append(f"{job_name} checkout must use the approved immutable action SHA")
    options = _as_mapping(checkout.get("with"))
    if str(options.get("persist-credentials", "")).lower() != "false":
        errors.append(f"{job_name} checkout must set persist-credentials: false")
    if str(options.get("fetch-depth", "")) != "0":
        errors.append(f"{job_name} checkout must set fetch-depth: 0")


def _validate_action_pins(job_name: str, job: dict[str, Any], errors: list[str]) -> None:
    for step in _as_steps(job):
        action = str(step.get("uses") or "")
        if action and not FULL_SHA_ACTION.fullmatch(action):
            errors.append(f"{job_name} action is not pinned to a full commit SHA: {action}")


def _require_tokens(
    label: str,
    text: str,
    tokens: tuple[str, ...],
    errors: list[str],
) -> None:
    for token in tokens:
        if token not in text:
            errors.append(f"{label} is missing required invariant: {token}")


def _validate_trusted_ref(job: dict[str, Any], errors: list[str]) -> None:
    step = _step(job, "Validate trusted release ref and ancestry")
    if step is None:
        errors.append("preflight is missing the trusted ref and ancestry gate")
        return
    run = str(step.get("run") or "")
    _require_tokens(
        "trusted ref gate",
        run,
        (
            '"refs/heads/main"',
            "refs/tags/v*",
            "refs/remotes/origin/main",
            'git merge-base --is-ancestor "$GITHUB_SHA" origin/main',
            'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"',
        ),
        errors,
    )


def _validate_job_boundaries(jobs: dict[str, Any], errors: list[str]) -> None:
    for name in REQUIRED_JOBS:
        job = _as_mapping(jobs.get(name))
        if not job:
            continue
        if _needs(job) != EXPECTED_NEEDS[name]:
            errors.append(
                f"{name} needs must be exactly: {', '.join(sorted(EXPECTED_NEEDS[name])) or '(none)'}"
            )
        labels = _runner_labels(job)
        if not EXPECTED_RUNNER_LABELS[name].issubset(labels):
            errors.append(f"{name} does not run on the required isolated runner boundary")
        _validate_permissions(
            name,
            job.get("permissions"),
            errors,
            packages_read_allowed=name in {"prepare-evaluation", "deploy"},
        )
        _validate_checkout(name, job, errors)
        _validate_action_pins(name, job, errors)

        secrets = set(SECRET_REFERENCE.findall(_job_text(job)))
        if secrets != EXPECTED_SECRETS[name]:
            errors.append(
                f"{name} secret boundary must be exactly: "
                f"{', '.join(sorted(EXPECTED_SECRETS[name])) or '(none)'}"
            )

    preflight = _as_mapping(jobs.get("preflight"))
    if "environment" in preflight:
        errors.append("preflight must not be attached to a protected secret environment")

    for name, expected in EXPECTED_ENVIRONMENTS.items():
        if _environment_name(_as_mapping(jobs.get(name))) != expected:
            errors.append(f"{name} must use environment {expected}")


def _validate_dependency_separation(jobs: dict[str, Any], errors: list[str]) -> None:
    preflight_text = _job_text(_as_mapping(jobs.get("preflight")))
    prepare_text = _job_text(_as_mapping(jobs.get("prepare-evaluation")))
    judge_job = _as_mapping(jobs.get("ragas-judge"))
    judge_text = _job_text(judge_job)
    deploy_text = _job_text(_as_mapping(jobs.get("deploy")))

    evaluation_execution = re.compile(
        r"(?:pip\s+install[^\n]*requirements-evaluation\.lock\.txt|"
        r"python\s+scripts/evaluate_ragas\.py)"
    )
    if evaluation_execution.search(preflight_text):
        errors.append("preflight must not install or execute evaluation dependencies")
    if evaluation_execution.search(prepare_text):
        errors.append("prepare-evaluation must execute the candidate image without Ragas dependencies")
    for job_name, approved_install in APPROVED_JOB_INSTALLS.items():
        run_text = "\n".join(
            str(step.get("run") or "")
            for step in _as_steps(_as_mapping(jobs.get(job_name)))
        )
        pip_occurrences = re.findall(r"(?i)\bpip\s+install\b", run_text)
        install_lines = [
            line.strip()
            for line in run_text.splitlines()
            if re.search(r"(?i)\bpip\s+install\b", line)
        ]
        if len(pip_occurrences) != 1 or install_lines != [approved_install]:
            errors.append(
                f"{job_name} must contain exactly one pip install: the approved "
                "hash-locked dependency command"
            )
    if evaluation_execution.search(deploy_text):
        errors.append("deploy must not install or execute evaluation dependencies")

    for forbidden in (
        "RAG_PRODUCTION_ENV_B64",
        "RAG_EVALUATION_ENV_B64",
        "RAG_ENV_FILE",
        "POSTGRES_",
        "REDIS_URL",
        "S3_",
        "kubectl",
        "docker run",
    ):
        if forbidden in judge_text:
            errors.append(f"ragas-judge crosses its minimal boundary via {forbidden}")


def _validate_registry_authentication(jobs: dict[str, Any], errors: list[str]) -> None:
    for job_name, verification_step_name in REGISTRY_VERIFICATION_STEPS.items():
        job = _as_mapping(jobs.get(job_name))
        steps = _as_steps(job)
        authentication_steps = [
            step
            for step in steps
            if step.get("name") == "Authenticate to repository package namespace"
        ]
        if len(authentication_steps) != 1:
            errors.append(
                f"{job_name} must contain exactly one repository package authentication step"
            )
            continue

        authentication = authentication_steps[0]
        environment = _as_mapping(authentication.get("env"))
        if environment.get("REGISTRY_TOKEN") != "${{ github.token }}":
            errors.append(
                f"{job_name} registry authentication must source REGISTRY_TOKEN from github.token"
            )
        authentication_run = str(authentication.get("run") or "")
        _require_tokens(
            f"{job_name} registry authentication",
            authentication_run,
            (
                'test -n "$REGISTRY_TOKEN"',
                "printf '%s' \"$REGISTRY_TOKEN\"",
                "docker login ghcr.io",
                '--username "$GITHUB_ACTOR"',
                "--password-stdin",
            ),
            errors,
        )

        verification = _step(job, verification_step_name)
        if verification is None:
            errors.append(
                f"{job_name} is missing the image verification step required after registry login"
            )
        elif steps.index(authentication) > steps.index(verification):
            errors.append(f"{job_name} authenticates to the registry after image verification")

        logout_steps = [
            step
            for step in steps
            if step.get("name") == "Remove repository package credentials"
        ]
        if len(logout_steps) != 1:
            errors.append(
                f"{job_name} must contain exactly one repository package credential cleanup step"
            )
            continue
        logout = logout_steps[0]
        if "always()" not in str(logout.get("if") or ""):
            errors.append(f"{job_name} registry credential cleanup must run with always()")
        if "docker logout ghcr.io" not in str(logout.get("run") or ""):
            errors.append(f"{job_name} registry credential cleanup must log out of ghcr.io")
        if steps[-1] is not logout:
            errors.append(f"{job_name} registry credential cleanup must be the final step")


def _validate_artifact_chain(jobs: dict[str, Any], errors: list[str]) -> None:
    preflight = _job_text(_as_mapping(jobs.get("preflight")))
    prepare = _job_text(_as_mapping(jobs.get("prepare-evaluation")))
    judge = _job_text(_as_mapping(jobs.get("ragas-judge")))
    deploy = _job_text(_as_mapping(jobs.get("deploy")))

    _require_tokens(
        "preflight manifest",
        preflight,
        (
            "preflight-manifest.json",
            "industrial-rag-release-preflight",
            "manifest_sha256",
            "requirements-runtime.lock.txt",
            "requirements-evaluation.lock.txt",
            "legal_expanded_ragas_40.jsonl",
            "legal_expanded_240.jsonl",
            "legal_holdout_150.jsonl",
            "app/evaluation/retrieval_contract.py",
            "scripts/approve_release_baseline.py",
            "scripts/score_retrieval.py",
            "scripts/validate_release_baseline.py",
        ),
        errors,
    )
    _require_tokens(
        "prepared evaluation manifest",
        prepare,
        (
            "needs.preflight.outputs.manifest_sha256",
            "industrial-rag-release-preflight",
            "prepared-evaluation-manifest.json",
            "industrial-rag-release-prepared-evaluation",
            "source_manifest_sha256",
            "candidate_images",
            "docker run --rm",
            "dst=/evaluation/scripts,readonly",
            "dst=/evaluation/eval,readonly",
        ),
        errors,
    )
    _require_tokens(
        "judge manifest",
        judge,
        (
            "needs.prepare-evaluation.outputs.manifest_sha256",
            "industrial-rag-release-prepared-evaluation",
            "judged-evaluation-manifest.json",
            "industrial-rag-release-judged-evaluation",
            "prepared_manifest_sha256",
            "RAGAS_JUDGE_API_KEY",
            "--api-key-env RAGAS_JUDGE_API_KEY",
        ),
        errors,
    )
    _require_tokens(
        "deploy artifact verification",
        deploy,
        (
            "needs.preflight.outputs.manifest_sha256",
            "needs.prepare-evaluation.outputs.manifest_sha256",
            "needs.ragas-judge.outputs.manifest_sha256",
            "industrial-rag-release-preflight",
            "industrial-rag-release-prepared-evaluation",
            "industrial-rag-release-judged-evaluation",
            "Verify immutable release evidence chain",
            "candidate_images",
            "source_manifest_sha256",
            "prepared_manifest_sha256",
        ),
        errors,
    )


def _validate_retrieval_evaluation_contract(
    jobs: dict[str, Any],
    errors: list[str],
) -> None:
    prepare = _as_mapping(jobs.get("prepare-evaluation"))
    step = _step(prepare, "Execute deterministic evaluations in the verified candidate image")
    if step is None:
        errors.append("prepare-evaluation is missing deterministic retrieval evaluation")
        return

    run = str(step.get("run") or "").replace("\\\n", " ")
    commands = [
        line.strip()
        for line in run.splitlines()
        if "python /evaluation/scripts/evaluate_retrieval_suite.py" in line
    ]
    expected = {
        "/evaluation/eval/legal_expanded_240.jsonl": (
            "/evaluation/output/retrieval-240.json"
        ),
        "/evaluation/eval/legal_holdout_150.jsonl": (
            "/evaluation/output/retrieval-holdout-150.json"
        ),
    }
    if len(commands) != len(expected):
        errors.append("prepare-evaluation must run exactly two protected retrieval suites")
        return

    for input_path, output_path in expected.items():
        command = next((item for item in commands if input_path in item), "")
        if not command or output_path not in command:
            errors.append(f"protected retrieval suite is missing or misrouted: {input_path}")
            continue
        _require_tokens(
            f"protected retrieval suite {input_path}",
            command,
            (
                "--top-k 5",
                "--use-production-pipeline",
                "--production-decomposition",
                "--minimum-citation-recall 0.95",
            ),
            errors,
        )


def _validate_atomic_image_set(jobs: dict[str, Any], errors: list[str]) -> None:
    expected_steps = {
        "prepare-evaluation": "Verify atomic image-set attestation before candidate execution",
        "deploy": "Verify atomic production image-set attestation",
    }
    for job_name, step_name in expected_steps.items():
        job = _as_mapping(jobs.get(job_name))
        step = _step(job, step_name)
        if step is None:
            errors.append(f"{job_name} is missing atomic image-set attestation verification")
            continue
        step_env = _as_mapping(step.get("env"))
        if step_env.get("PUBLISHER_IDENTITY") != "${{ vars.COSIGN_CERTIFICATE_IDENTITY }}":
            errors.append(
                f"{job_name} atomic image-set verification must use the protected "
                "COSIGN_CERTIFICATE_IDENTITY variable"
            )
        run = str(step.get("run") or "")
        _require_tokens(
            f"{job_name} atomic image-set verification",
            run,
            (
                "release_set_attestation.py predicate-type",
                "${PUBLISHER_IDENTITY:?Set the exact protected image-publisher identity}",
                "cosign verify-attestation",
                '--certificate-github-workflow-sha "$GITHUB_SHA"',
                'VERIFIED_DSSE="$verification_output"',
                "base64.b64decode(payload, validate=True)",
                'source.get("workflowRun")',
                "release_set_attestation.py verify-attestation",
                '--verification-output "$verification_output"',
                '--source-revision "$GITHUB_SHA"',
                '--api-image "$API_IMAGE"',
                '--worker-image "$WORKER_IMAGE"',
                '--frontend-image "$FRONTEND_IMAGE"',
                '--model-bundle-image "$MODEL_BUNDLE_IMAGE"',
            ),
            errors,
        )
        if "cosign attest" in run:
            errors.append(f"{job_name} must verify, never issue, the atomic image-set attestation")

    prepare_names = [str(step.get("name") or "") for step in _as_steps(_as_mapping(jobs.get("prepare-evaluation")))]
    deploy_names = [str(step.get("name") or "") for step in _as_steps(_as_mapping(jobs.get("deploy")))]
    if all(name in prepare_names for name in (expected_steps["prepare-evaluation"], "Execute deterministic evaluations in the verified candidate image")):
        if prepare_names.index(expected_steps["prepare-evaluation"]) > prepare_names.index(
            "Execute deterministic evaluations in the verified candidate image"
        ):
            errors.append("prepare-evaluation executes the candidate before atomic verification")
    if all(name in deploy_names for name in (expected_steps["deploy"], "Roll out stable Deployments")):
        if deploy_names.index(expected_steps["deploy"]) > deploy_names.index("Roll out stable Deployments"):
            errors.append("deploy rolls out images before atomic image-set verification")


def _validate_cleanup_and_smoke(jobs: dict[str, Any], errors: list[str]) -> None:
    deploy = _as_mapping(jobs.get("deploy"))
    cleanup = _step(deploy, "Cleanup temporary Kubernetes release resources")
    if cleanup is None:
        errors.append("deploy is missing always-run Kubernetes temporary resource cleanup")
    else:
        if "always()" not in str(cleanup.get("if") or ""):
            errors.append("temporary Kubernetes resource cleanup must run with always()")
        run = str(cleanup.get("run") or "")
        _require_tokens(
            "temporary Kubernetes resource cleanup",
            run,
            (
                "delete_owned",
                "rag-postgres-migration",
                "rag-canary",
                "migration-secret-name",
                "canary-secret-name",
                "metadata.uid",
            ),
            errors,
        )

    smoke = _step(deploy, "Run read-only production ingress smoke")
    if smoke is None:
        errors.append("deploy is missing the read-only ingress smoke")
        return
    run = str(smoke.get("run") or "")
    _require_tokens(
        "read-only ingress smoke",
        run,
        ("--request GET", 'https://${INGRESS_HOST}/health/live', 'https://${INGRESS_HOST}/health/ready'),
        errors,
    )
    if re.search(r"(?:--request|-X)\s+(?:POST|PUT|PATCH|DELETE)", run, re.IGNORECASE):
        errors.append("production ingress smoke must be read-only")


def validate_workflow(path: Path) -> list[str]:
    """Return every release workflow policy violation."""
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return [f"unable to parse release workflow: {exc}"]
    root = _as_mapping(document)
    jobs = _as_mapping(root.get("jobs"))
    errors: list[str] = []

    _validate_permissions("global", root.get("permissions"), errors)
    missing = [name for name in REQUIRED_JOBS if name not in jobs]
    if missing:
        errors.append(f"release workflow is missing required jobs: {', '.join(missing)}")
    unexpected = sorted(set(jobs) - set(REQUIRED_JOBS))
    if unexpected:
        errors.append(f"release workflow contains unreviewed jobs: {', '.join(unexpected)}")

    _validate_job_boundaries(jobs, errors)
    if "preflight" in jobs:
        _validate_trusted_ref(_as_mapping(jobs["preflight"]), errors)
    _validate_dependency_separation(jobs, errors)
    _validate_registry_authentication(jobs, errors)
    _validate_artifact_chain(jobs, errors)
    _validate_retrieval_evaluation_contract(jobs, errors)
    _validate_atomic_image_set(jobs, errors)
    _validate_cleanup_and_smoke(jobs, errors)
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".github/workflows/release.yml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_workflow(args.workflow)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print("Release workflow semantic policy passed.")


if __name__ == "__main__":
    main()
