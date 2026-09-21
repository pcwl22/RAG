from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_release_workflow import validate_workflow

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPOSITORY_ROOT / ".github/workflows/release.yml"


def _mutated_workflow(tmp_path: Path, old: str, new: str) -> Path:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert old in source, f"checked-in workflow no longer contains mutation target: {old}"
    path = tmp_path / "release.yml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return path


def test_checked_in_release_workflow_satisfies_semantic_policy():
    assert validate_workflow(WORKFLOW) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            "id-token: none",
            "id-token: write",
            "global permissions must set id-token: none",
        ),
        (
            "persist-credentials: false",
            "persist-credentials: true",
            "preflight checkout must set persist-credentials: false",
        ),
        (
            'git merge-base --is-ancestor "$GITHUB_SHA" origin/main',
            "git merge-base --is-ancestor origin/main HEAD~1",
            "trusted ref gate is missing required invariant",
        ),
        (
            "needs: preflight",
            "needs: ragas-judge",
            "prepare-evaluation needs must be exactly: preflight",
        ),
        (
            "RAGAS_JUDGE_API_KEY: ${{ secrets.RAGAS_JUDGE_API_KEY }}",
            "RAGAS_JUDGE_API_KEY: ${{ secrets.RAG_PRODUCTION_ENV_B64 }}",
            "ragas-judge secret boundary must be exactly: RAGAS_JUDGE_API_KEY",
        ),
        (
            "if: always()\n        shell: bash\n        run: |\n          set +e\n          cleanup_status=0",
            "if: success()\n        shell: bash\n        run: |\n          set +e\n          cleanup_status=0",
            "temporary Kubernetes resource cleanup must run with always()",
        ),
        (
            'curl --request GET --fail --silent --show-error --output /dev/null "https://${INGRESS_HOST}/health/live"',
            'curl --request POST --fail --silent --show-error --output /dev/null "https://${INGRESS_HOST}/health/live"',
            "production ingress smoke must be read-only",
        ),
    ],
)
def test_release_workflow_policy_rejects_security_regressions(
    tmp_path: Path,
    old: str,
    new: str,
    expected: str,
):
    path = _mutated_workflow(tmp_path, old, new)

    assert any(expected in error for error in validate_workflow(path))


def test_release_workflow_registry_authentication_uses_ephemeral_job_token(tmp_path: Path):
    path = _mutated_workflow(
        tmp_path,
        "REGISTRY_TOKEN: ${{ github.token }}",
        "REGISTRY_TOKEN: ${{ secrets.RAG_EVALUATION_ENV_B64 }}",
    )

    assert any(
        "prepare-evaluation registry authentication must source REGISTRY_TOKEN from github.token"
        in error
        for error in validate_workflow(path)
    )


def test_release_workflow_registry_login_requires_password_stdin(tmp_path: Path):
    path = _mutated_workflow(tmp_path, "--password-stdin", "--password unsafe-command-argument")

    assert any(
        "prepare-evaluation registry authentication is missing required invariant: --password-stdin"
        in error
        for error in validate_workflow(path)
    )


def test_release_workflow_registry_logout_is_always_run_and_final(tmp_path: Path):
    path = _mutated_workflow(
        tmp_path,
        "- name: Remove repository package credentials\n        if: always()",
        "- name: Remove repository package credentials\n        if: success()",
    )

    assert (
        "prepare-evaluation registry credential cleanup must run with always()"
        in validate_workflow(path)
    )


def test_release_workflow_rejects_evaluation_dependencies_in_deploy(tmp_path: Path):
    path = _mutated_workflow(
        tmp_path,
        "python -m pip install --require-hashes -r requirements-ci.lock.txt\n",
        "python -m pip install --require-hashes -r requirements-ci.lock.txt\n"
        "          python -m pip install --require-hashes -r requirements-evaluation.lock.txt\n",
    )
    # The first occurrence belongs to preflight. Mutate the deploy occurrence explicitly.
    text = path.read_text(encoding="utf-8")
    first = text.find("requirements-evaluation.lock.txt", text.find("deploy:"))
    if first < 0:
        deploy_marker = "  deploy:\n"
        insertion = text.index("    steps:\n", text.index(deploy_marker)) + len("    steps:\n")
        text = (
            text[:insertion]
            + "      - name: Unsafe evaluation install\n"
            + "        run: python -m pip install -r requirements-evaluation.lock.txt\n"
            + text[insertion:]
        )
        path.write_text(text, encoding="utf-8")

    assert "deploy must not install or execute evaluation dependencies" in validate_workflow(path)


def test_release_workflow_rejects_extra_unlocked_judge_install(tmp_path: Path):
    approved = "python -m pip install --require-hashes -r requirements-evaluation.lock.txt"
    path = _mutated_workflow(
        tmp_path,
        approved,
        approved + "\n          python -m pip install -e . --no-deps",
    )

    assert any(
        "ragas-judge must contain exactly one pip install" in error
        for error in validate_workflow(path)
    )


def test_release_workflow_requires_retrieval_fail_fast_threshold_on_each_suite(tmp_path: Path):
    path = _mutated_workflow(
        tmp_path,
        "--minimum-citation-recall 0.95",
        "--minimum-citation-recall 0.90",
    )

    assert any(
        "protected retrieval suite /evaluation/eval/legal_expanded_240.jsonl "
        "is missing required invariant: --minimum-citation-recall 0.95" in error
        for error in validate_workflow(path)
    )


@pytest.mark.parametrize(
    "required_path",
    [
        "app/evaluation/retrieval_contract.py",
        "scripts/approve_release_baseline.py",
        "scripts/score_retrieval.py",
        "scripts/validate_release_baseline.py",
    ],
)
def test_release_workflow_source_contract_binds_release_gate_implementation(
    tmp_path: Path,
    required_path: str,
):
    path = _mutated_workflow(tmp_path, f'              "{required_path}",\n', "")

    assert any(
        f"preflight manifest is missing required invariant: {required_path}" in error
        for error in validate_workflow(path)
    )
