"""Regression tests for zero-exception dependency and judge boundaries."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_dependency_locks.py"
SPEC = importlib.util.spec_from_file_location("audit_dependency_locks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dependency_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dependency_audit)


@pytest.fixture
def policy_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo" / "industrial-rag"
    for relative in (
        "app/evaluation/__init__.py",
        "app/evaluation/native_judge.py",
        "app/evaluation/quality_gate.py",
        "app/evaluation/ragas_adapter.py",
        "scripts/evaluate_ragas.py",
        "requirements-evaluation.txt",
        "requirements-evaluation.lock.txt",
        "security/evaluation_dependency_exceptions.toml",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative).read_bytes())
    workflow = root.parent / ".github/workflows/release.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("jobs:\n  ragas-judge:\n    runs-on: ubuntu-latest\n", encoding="utf-8")
    return root


def test_checked_in_zero_exception_policy_is_valid(policy_root: Path) -> None:
    errors, entries = dependency_audit.validate_exception_policy(policy_root)

    assert errors == []
    assert entries == []
    assert dependency_audit.EXPECTED_EXCEPTIONS == {}


def test_policy_rejects_any_vulnerability_exception(policy_root: Path) -> None:
    policy = policy_root / dependency_audit.POLICY_PATH
    policy.write_text(
        'schema_version = 2\nexceptions = [{ advisory_id = "PYSEC-2099-1" }]\n',
        encoding="utf-8",
    )

    errors, entries = dependency_audit.validate_exception_policy(policy_root)

    assert "dependency exception policy must contain exactly exceptions = []" in errors
    assert entries


@pytest.mark.parametrize("target", ["input", "lock"])
def test_evaluation_dependencies_reject_forbidden_stack(
    policy_root: Path,
    target: str,
) -> None:
    filename = (
        dependency_audit.EVALUATION_INPUT
        if target == "input"
        else dependency_audit.EVALUATION_LOCK
    )
    path = policy_root / filename
    path.write_text(path.read_text(encoding="utf-8") + "\nragas==0.4.3\n", encoding="utf-8")

    errors = dependency_audit.validate_evaluation_boundary(policy_root)

    assert any("evaluation dependency" in error for error in errors)


@pytest.mark.parametrize(
    ("relative_path", "injected", "message"),
    [
        (
            "app/evaluation/native_judge.py",
            "\nimport ragas\n",
            "must not import forbidden module ragas",
        ),
        (
            "scripts/evaluate_ragas.py",
            "\nfrom langchain.prompts.loading import load_prompt\n",
            "must not import forbidden module langchain.prompts.loading",
        ),
        (
            "scripts/evaluate_ragas.py",
            "\nfrom app.utils.config import normalize_openai_base_url\n",
            "must not import runtime-only project module app.utils.config",
        ),
        (
            "app/evaluation/native_judge.py",
            "\ndef unsafe(client):\n    return client.chat.completions.create(tools=[])\n",
            "must not expose tool/function call keywords: tools",
        ),
    ],
)
def test_evaluation_boundary_rejects_vulnerable_sinks(
    policy_root: Path,
    relative_path: str,
    injected: str,
    message: str,
) -> None:
    path = policy_root / relative_path
    path.write_text(path.read_text(encoding="utf-8") + injected, encoding="utf-8")

    errors = dependency_audit.validate_evaluation_boundary(policy_root)

    assert any(message in error for error in errors)


def test_evaluation_boundary_requires_isolated_workflow_job(policy_root: Path) -> None:
    workflow = policy_root.parent / ".github/workflows/release.yml"
    workflow.write_text("jobs:\n  deploy:\n    runs-on: ubuntu-latest\n", encoding="utf-8")

    errors = dependency_audit.validate_evaluation_boundary(policy_root)

    assert "release workflow must provide the isolated ragas-judge job" in errors


def test_all_lock_audits_run_without_ignored_vulnerabilities(
    policy_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dependency_audit.subprocess, "run", fake_run)

    assert dependency_audit.audit_lock_files(policy_root, []) == []
    assert len(commands) == len(dependency_audit.AUDIT_LOCKS)
    assert all("--ignore-vuln" not in command for command in commands)


def test_audit_refuses_nonempty_exception_list(policy_root: Path) -> None:
    assert dependency_audit.audit_lock_files(
        policy_root,
        [{"advisory_id": "PYSEC-2099-1"}],
    ) == ["dependency vulnerability exceptions are no longer supported"]


def test_ci_enforces_all_lock_audits_and_release_policy() -> None:
    workflow_path = PROJECT_ROOT.parent / ".github/workflows/ci.yml"
    source = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.load(source, Loader=yaml.BaseLoader)
    job = workflow["jobs"]["dependency-security"]
    commands = "\n".join(
        str(step.get("run") or "")
        for step in job["steps"]
        if isinstance(step, dict)
    )

    assert "scripts/validate_dependency_lock.py" in commands
    assert "scripts/validate_release_workflow.py ../.github/workflows/release.yml" in commands
    assert "scripts/audit_dependency_locks.py" in commands
    assert "--check-upstream" in commands
    assert "--output-dir" in commands
    assert "pip install -e ." not in commands
    assert "secrets." not in yaml.safe_dump(job)

    for configured_job in workflow["jobs"].values():
        for step in configured_job.get("steps", []):
            if str(step.get("uses") or "").startswith("actions/checkout@"):
                assert step.get("with", {}).get("persist-credentials") == "false"


def test_workflows_and_images_do_not_install_project_through_pep517() -> None:
    dockerfiles = [
        PROJECT_ROOT / "docker/api/Dockerfile",
        PROJECT_ROOT / "docker/api/Dockerfile.production",
        PROJECT_ROOT / "docker/worker/Dockerfile",
        PROJECT_ROOT / "docker/worker/Dockerfile.production",
    ]
    paths = [
        PROJECT_ROOT.parent / ".github/workflows/ci.yml",
        PROJECT_ROOT.parent / ".github/workflows/release.yml",
        *dockerfiles,
    ]

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "pip install -e ." not in source, path
        assert "pip install --no-cache-dir --no-deps ." not in source, path
    for path in dockerfiles:
        assert "PYTHONPATH=/app" in path.read_text(encoding="utf-8"), path
