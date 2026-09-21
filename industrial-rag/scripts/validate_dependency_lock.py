"""Validate that checked-in pip-tools locks are pinned and hash complete."""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement

PACKAGE_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\]+)")
HASH_LINE = re.compile(r"^--hash=sha256:[0-9a-f]{64}(?:\s+\\)?$")
ALLOWED_DIRECTIVES = ("--index-url ", "--extra-index-url ", "--trusted-host ")
EXPECTED_ML_VERSIONS = {
    "torch": "2.13.0",
    "transformers": "5.16.1",
    "sentence-transformers": "6.0.0",
    "tokenizers": "0.23.1",
    "huggingface-hub": "1.29.0",
}
EXPECTED_CPU_TORCH = "2.13.0+cpu"
EXPECTED_CUDA_TORCH = "2.13.0+cu126"
EXPECTED_CPU_TORCH_HASH = "4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2"
EXPECTED_CUDA_TORCH_HASH = "380081ea098bf2b9e727aa85205d94790d884d17c62df3bb00a4f6a1047010a2"
EXPECTED_WINDOWS_RESOLVER_TORCH_HASH = (
    "a8b450c1e58e5800e5b4691dac412f8d2d65a1dc3298166f91596603a3531e6f"
)
EXPECTED_API_BASE = (
    "FROM ghcr.io/pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@"
    "sha256:b2e6463673129d979402f11083769c9eaad9211487b6260f52525f55a0a82dac"
)
EXPECTED_WORKER_BASE = (
    "FROM python:3.12.14-slim@"
    "sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217"
)
EXPECTED_RESOLVER_CONSTRAINTS = {
    "constraints-torch-cpu-linux-amd64-py312.txt": (
        "torch @ https://download-r2.pytorch.org/whl/cpu/"
        "torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl#"
        "sha256=4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2"
    ),
    "constraints-torch-cpu-windows-amd64-py312.txt": (
        "torch @ https://download-r2.pytorch.org/whl/cpu/"
        "torch-2.13.0%2Bcpu-cp312-cp312-win_amd64.whl#"
        f"sha256={EXPECTED_WINDOWS_RESOLVER_TORCH_HASH}"
    ),
}


def _direct_requirement_names(path: Path) -> set[str]:
    if path.suffix == ".toml":
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        requirements = (document.get("project") or {}).get("dependencies") or []
        return {
            Requirement(requirement).name.lower().replace("_", "-")
            for requirement in requirements
        }

    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        names.add(Requirement(line).name.lower().replace("_", "-"))
    return names


def _normal_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _project_dependency_names(path: Path, *, platform_system: str) -> set[str]:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    requirements = (document.get("project") or {}).get("dependencies") or []
    environment: dict[str, str] = {
        name: str(value) for name, value in default_environment().items()
    }
    environment.update(
        {
            "platform_system": platform_system,
            "sys_platform": "win32" if platform_system == "Windows" else "linux",
        }
    )
    names: set[str] = set()
    for value in requirements:
        requirement = Requirement(value)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            names.add(_normal_name(requirement.name))
    return names


def validate_lock(
    lock_path: Path,
    source_path: Path,
    *,
    ignored_direct: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    entries: dict[str, bool] = {}
    current: str | None = None

    for line_number, raw_line in enumerate(lock_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(ALLOWED_DIRECTIVES):
            continue
        if stripped.startswith("--hash="):
            if current is None:
                errors.append(f"{lock_path}:{line_number}: hash without a package entry")
            elif not HASH_LINE.fullmatch(stripped):
                errors.append(f"{lock_path}:{line_number}: invalid SHA-256 hash")
            else:
                entries[current] = True
            continue
        match = PACKAGE_LINE.fullmatch(stripped.removesuffix("\\").rstrip())
        if match:
            current = _normal_name(match.group(1))
            entries.setdefault(current, False)
            continue
        if stripped.startswith("# via"):
            continue
        errors.append(f"{lock_path}:{line_number}: unsupported or unhashed lock line")

    missing_hashes = sorted(name for name, has_hash in entries.items() if not has_hash)
    errors.extend(f"{lock_path}: package {name} has no SHA-256 hash" for name in missing_hashes)

    source_names = _direct_requirement_names(source_path) - (ignored_direct or set())
    missing_direct = sorted(name for name in source_names if name not in entries)
    errors.extend(f"{lock_path}: direct requirement {name} is missing" for name in missing_direct)
    return errors


def _locked_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = PACKAGE_LINE.match(raw_line.strip())
        if match:
            versions[_normal_name(match.group(1))] = match.group(2)
    return versions


def _single_requirement_line(path: Path) -> str:
    lines = [
        raw_line.strip()
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if raw_line.strip() and not raw_line.lstrip().startswith("#")
    ]
    return "\n".join(lines)


def _lock_hashes(path: Path) -> set[str]:
    return {
        stripped.removeprefix("--hash=sha256:").removesuffix("\\").strip()
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (stripped := raw_line.strip()).startswith("--hash=sha256:")
    }


def validate_runtime_matrix(root: Path = Path(".")) -> list[str]:
    """Validate the controlled Torch carriers and shared ML application locks."""
    errors: list[str] = []
    constraints = _locked_versions(root / "constraints-docker.txt")
    runtime = _locked_versions(root / "requirements-runtime.lock.txt")
    gpu = _locked_versions(root / "requirements-gpu.lock.txt")
    gpu_source = _direct_requirement_names(root / "requirements-gpu-verified.txt")
    cpu_torch = _locked_versions(root / "requirements-torch-cpu.lock.txt")
    cuda_torch = _locked_versions(root / "requirements-torch-cu126.lock.txt")

    for package, expected in EXPECTED_ML_VERSIONS.items():
        if constraints.get(package) != expected:
            errors.append(
                f"constraints-docker.txt: {package} must be pinned to {expected}"
            )
        if package != "torch" and runtime.get(package) != expected:
            errors.append(
                f"requirements-runtime.lock.txt: {package} must be pinned to {expected}"
            )
        if package != "torch" and gpu.get(package) != expected:
            errors.append(
                f"requirements-gpu.lock.txt: {package} must be pinned to {expected}"
            )

    for lock_name, versions in (
        ("requirements-runtime.lock.txt", runtime),
        ("requirements-gpu.lock.txt", gpu),
    ):
        if "torch" in versions:
            errors.append(
                f"{lock_name}: torch must be supplied only by a controlled carrier"
            )
        for unused in ("torchvision", "torchaudio", "datasets", "pyarrow", "pillow"):
            if unused in versions:
                errors.append(f"{lock_name}: unused package {unused} must not be deployed")

    windows_project = _project_dependency_names(
        root / "pyproject.toml", platform_system="Windows"
    ) - {"torch"}
    for missing in sorted(windows_project - gpu_source):
        errors.append(
            "requirements-gpu-verified.txt: Windows project dependency "
            f"{missing} is missing"
        )
    for package in sorted(gpu_source & runtime.keys() & gpu.keys()):
        if gpu[package] != runtime[package]:
            errors.append(
                f"requirements-gpu.lock.txt: {package}=={gpu[package]} does not match "
                f"Linux runtime {runtime[package]}"
            )

    if cpu_torch != {"torch": EXPECTED_CPU_TORCH}:
        errors.append(
            "requirements-torch-cpu.lock.txt: expected only torch=="
            f"{EXPECTED_CPU_TORCH}"
        )
    if _lock_hashes(root / "requirements-torch-cpu.lock.txt") != {
        EXPECTED_CPU_TORCH_HASH
    }:
        errors.append("requirements-torch-cpu.lock.txt: CPU wheel SHA-256 does not match")
    if cuda_torch != {"torch": EXPECTED_CUDA_TORCH}:
        errors.append(
            "requirements-torch-cu126.lock.txt: expected only torch=="
            f"{EXPECTED_CUDA_TORCH}"
        )
    if _lock_hashes(root / "requirements-torch-cu126.lock.txt") != {
        EXPECTED_CUDA_TORCH_HASH
    }:
        errors.append("requirements-torch-cu126.lock.txt: CUDA wheel SHA-256 does not match")

    for name, expected in EXPECTED_RESOLVER_CONSTRAINTS.items():
        actual = _single_requirement_line(root / name)
        if actual != expected:
            errors.append(f"{name}: resolver artifact URL or SHA-256 does not match")

    for relative in ("docker/api/Dockerfile", "docker/api/Dockerfile.production"):
        content = (root / relative).read_text(encoding="utf-8")
        if EXPECTED_API_BASE not in content:
            errors.append(f"{relative}: API base image is not the approved immutable digest")
        if "requirements-torch-cpu.lock.txt" in content:
            errors.append(f"{relative}: CUDA API must not install the CPU Torch carrier")

    for relative in ("docker/worker/Dockerfile", "docker/worker/Dockerfile.production"):
        content = (root / relative).read_text(encoding="utf-8")
        if EXPECTED_WORKER_BASE not in content:
            errors.append(f"{relative}: worker base image is not the approved immutable digest")
        for required_text in (
            "requirements-torch-cpu.lock.txt",
            "https://download.pytorch.org/whl/cpu",
            "--no-deps",
            "--require-hashes",
            "torch.version.cuda is None",
        ):
            if required_text not in content:
                errors.append(f"{relative}: missing controlled CPU Torch check {required_text!r}")

    dockerignore = {
        line.strip()
        for line in (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if "!requirements-torch-cpu.lock.txt" not in dockerignore:
        errors.append(
            ".dockerignore: requirements-torch-cpu.lock.txt must be available to worker builds"
        )

    setup_script = (root / "scripts/setup_gpu_env.ps1").read_text(encoding="utf-8")
    for required_text in (
        "requirements-torch-cu126.lock.txt",
        "--require-hashes",
        "--no-deps -e $projectRoot",
        "python -m pip check",
    ):
        if required_text not in setup_script:
            errors.append(
                "scripts/setup_gpu_env.ps1: missing locked installation control "
                f"{required_text!r}"
            )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock-source",
        nargs=2,
        action="append",
        metavar=("LOCK", "SOURCE"),
        default=[
            ("requirements-ci.lock.txt", "requirements-ci.txt"),
            ("requirements-gpu.lock.txt", "requirements-gpu-verified.txt"),
            ("requirements-runtime.lock.txt", "pyproject.toml"),
            ("requirements-torch-cpu.lock.txt", "requirements-torch-cpu.in"),
            ("requirements-torch-cu126.lock.txt", "requirements-torch-cu126.in"),
            ("requirements-evaluation.lock.txt", "requirements-evaluation.txt"),
        ],
        help="lock file and its direct requirement source (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    for lock_name, source_name in args.lock_source:
        lock_path = Path(lock_name)
        source_path = Path(source_name)
        if not lock_path.is_file():
            errors.append(f"missing lock file: {lock_path}")
            continue
        if not source_path.is_file():
            errors.append(f"missing lock source: {source_path}")
            continue
        ignored_direct = {"torch"} if lock_path.name == "requirements-runtime.lock.txt" else set()
        errors.extend(validate_lock(lock_path, source_path, ignored_direct=ignored_direct))

    errors.extend(validate_runtime_matrix())

    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print("Dependency locks are pinned and hash-complete.")


if __name__ == "__main__":
    main()
