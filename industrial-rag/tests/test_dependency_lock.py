"""Regression tests for production dependency carrier boundaries."""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "validate_dependency_lock.py"
SPEC = importlib.util.spec_from_file_location("validate_dependency_lock", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dependency_lock = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dependency_lock)


def test_checked_in_runtime_dependency_matrix_is_consistent() -> None:
    assert dependency_lock.validate_runtime_matrix(PROJECT_ROOT) == []


def test_runtime_lock_cannot_silently_reintroduce_torch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for relative in (
        "constraints-docker.txt",
        "requirements-runtime.lock.txt",
        "requirements-gpu.lock.txt",
        "requirements-torch-cpu.lock.txt",
        "requirements-torch-cu126.lock.txt",
        "constraints-torch-cpu-linux-amd64-py312.txt",
        "constraints-torch-cpu-windows-amd64-py312.txt",
        "requirements-gpu-verified.txt",
        "pyproject.toml",
        ".dockerignore",
        "scripts/setup_gpu_env.ps1",
        "docker/api/Dockerfile",
        "docker/api/Dockerfile.production",
        "docker/worker/Dockerfile",
        "docker/worker/Dockerfile.production",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative).read_bytes())

    runtime_lock = root / "requirements-runtime.lock.txt"
    runtime_lock.write_text(
        runtime_lock.read_text(encoding="utf-8")
        + "\ntorch==2.13.0 \\\n    --hash=sha256:"
        + "0" * 64
        + "\n",
        encoding="utf-8",
    )

    errors = dependency_lock.validate_runtime_matrix(root)
    assert any("torch must be supplied only by a controlled carrier" in error for error in errors)


def test_gpu_source_must_cover_windows_project_runtime(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for relative in (
        "constraints-docker.txt",
        "requirements-runtime.lock.txt",
        "requirements-gpu.lock.txt",
        "requirements-gpu-verified.txt",
        "requirements-torch-cpu.lock.txt",
        "requirements-torch-cu126.lock.txt",
        "constraints-torch-cpu-linux-amd64-py312.txt",
        "constraints-torch-cpu-windows-amd64-py312.txt",
        "pyproject.toml",
        ".dockerignore",
        "scripts/setup_gpu_env.ps1",
        "docker/api/Dockerfile",
        "docker/api/Dockerfile.production",
        "docker/worker/Dockerfile",
        "docker/worker/Dockerfile.production",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative).read_bytes())

    gpu_source = root / "requirements-gpu-verified.txt"
    gpu_source.write_text(
        "\n".join(
            line
            for line in gpu_source.read_text(encoding="utf-8").splitlines()
            if not line.startswith("boto3==")
        )
        + "\n",
        encoding="utf-8",
    )

    errors = dependency_lock.validate_runtime_matrix(root)
    assert any("Windows project dependency boto3 is missing" in error for error in errors)


def test_worker_torch_lock_cannot_be_removed_from_build_context(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for relative in (
        "constraints-docker.txt",
        "requirements-runtime.lock.txt",
        "requirements-gpu.lock.txt",
        "requirements-gpu-verified.txt",
        "requirements-torch-cpu.lock.txt",
        "requirements-torch-cu126.lock.txt",
        "constraints-torch-cpu-linux-amd64-py312.txt",
        "constraints-torch-cpu-windows-amd64-py312.txt",
        "pyproject.toml",
        ".dockerignore",
        "scripts/setup_gpu_env.ps1",
        "docker/api/Dockerfile",
        "docker/api/Dockerfile.production",
        "docker/worker/Dockerfile",
        "docker/worker/Dockerfile.production",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative).read_bytes())

    dockerignore = root / ".dockerignore"
    dockerignore.write_text(
        "\n".join(
            line
            for line in dockerignore.read_text(encoding="utf-8").splitlines()
            if line.strip() != "!requirements-torch-cpu.lock.txt"
        )
        + "\n",
        encoding="utf-8",
    )

    errors = dependency_lock.validate_runtime_matrix(root)
    assert any("must be available to worker builds" in error for error in errors)
