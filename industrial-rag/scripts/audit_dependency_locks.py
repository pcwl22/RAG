"""Audit every checked-in dependency lock with zero vulnerability exceptions."""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

POLICY_PATH = Path("security/evaluation_dependency_exceptions.toml")
AUDIT_LOCKS = (
    "requirements-ci.lock.txt",
    "requirements-runtime.lock.txt",
    "requirements-gpu.lock.txt",
    "requirements-torch-cpu.lock.txt",
    "requirements-torch-cu126.lock.txt",
    "requirements-evaluation.lock.txt",
)
EVALUATION_INPUT = "requirements-evaluation.txt"
EVALUATION_LOCK = "requirements-evaluation.lock.txt"
WORKFLOW_JOB = ".github/workflows/release.yml#ragas-judge"
ALLOWED_METRICS = (
    "answer_accuracy",
    "faithfulness",
    "context_precision",
    "context_recall",
)
EXPECTED_OPENAI_VERSION = "3.6.0"
EXPECTED_EXCEPTIONS: dict[str, tuple[str, bool]] = {}
FORBIDDEN_EVALUATION_PACKAGES = frozenset(
    {
        "diskcache",
        "instructor",
        "langchain",
        "langchain-community",
        "langchain-core",
        "langchain-openai",
        "langchain-text-splitters",
        "ragas",
    }
)
FORBIDDEN_IMPORT_ROOTS = frozenset({"diskcache", "instructor", "langchain", "ragas"})
ALLOWED_EVALUATION_APP_IMPORTS = frozenset(
    {
        "app.evaluation.native_judge",
        "app.evaluation.ragas_adapter",
        "app.utils.strict_dotenv",
    }
)
FORBIDDEN_SOURCE_SINKS = {
    "ChatOpenAI": "image URL token-counting client",
    "DiskCacheBackend": "unsafe cache deserializer",
    "MultiModal": "multimodal evaluator",
    "load_prompt": "filesystem prompt loader",
    "load_prompt_from_config": "filesystem prompt loader",
    "split_text_from_url": "redirect-following URL splitter",
}
FORBIDDEN_OPENAI_KEYWORDS = frozenset({"function_call", "functions", "tool_choice", "tools"})
PACKAGE_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\]+)")


def _normal_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _locked_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = PACKAGE_LINE.match(raw_line.strip())
        if match:
            versions[_normal_name(match.group(1))] = match.group(2)
    return versions


def _direct_requirements(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PACKAGE_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"evaluation dependency must use an exact pin: {line}")
        requirements[_normal_name(match.group(1))] = match.group(2)
    return requirements


def _metric_defaults(module: ast.Module) -> tuple[str, ...] | None:
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_METRICS"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            return None
        if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
            return value
    return None


def _source_modules(module: ast.Module) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def validate_evaluation_boundary(root: Path) -> list[str]:
    """Require a dependency-minimal, text-only judge with no vulnerable sinks."""
    errors: list[str] = []
    try:
        direct = _direct_requirements(root / EVALUATION_INPUT)
    except (OSError, ValueError) as exc:
        errors.append(f"evaluation dependency input is invalid: {exc}")
        direct = {}
    if direct != {"openai": EXPECTED_OPENAI_VERSION}:
        errors.append(
            "evaluation dependency input must contain only "
            f"openai=={EXPECTED_OPENAI_VERSION}"
        )

    try:
        locked = _locked_versions(root / EVALUATION_LOCK)
    except OSError as exc:
        errors.append(f"evaluation dependency lock is unreadable: {exc}")
        locked = {}
    present_forbidden = sorted(FORBIDDEN_EVALUATION_PACKAGES.intersection(locked))
    if present_forbidden:
        errors.append(
            "evaluation dependency lock contains forbidden packages: "
            + ", ".join(present_forbidden)
        )
    if locked.get("openai") != EXPECTED_OPENAI_VERSION:
        errors.append(f"evaluation dependency lock must pin openai=={EXPECTED_OPENAI_VERSION}")

    source_paths = [
        *sorted((root / "app/evaluation").glob("*.py")),
        root / "scripts/evaluate_ragas.py",
    ]
    parsed: dict[Path, tuple[ast.Module, str]] = {}
    for path in source_paths:
        try:
            source = path.read_text(encoding="utf-8")
            parsed[path] = (ast.parse(source, filename=str(path)), source)
        except (OSError, SyntaxError) as exc:
            errors.append(f"evaluation source is invalid: {path}: {exc}")

    adapter_path = root / "app/evaluation/ragas_adapter.py"
    if adapter_path in parsed and _metric_defaults(parsed[adapter_path][0]) != ALLOWED_METRICS:
        errors.append("evaluation DEFAULT_METRICS must remain the approved text-only tuple")

    for path, (module, source) in parsed.items():
        relative = path.relative_to(root).as_posix()
        for imported in _source_modules(module):
            import_root = imported.split(".", 1)[0].lower()
            if import_root in FORBIDDEN_IMPORT_ROOTS:
                errors.append(f"{relative} must not import forbidden module {imported}")
            if (
                relative == "scripts/evaluate_ragas.py"
                and imported.startswith("app.")
                and imported not in ALLOWED_EVALUATION_APP_IMPORTS
            ):
                errors.append(
                    f"{relative} must not import runtime-only project module {imported}"
                )
        for marker, description in FORBIDDEN_SOURCE_SINKS.items():
            if marker in source:
                errors.append(f"{relative} must not expose {description} ({marker})")
        for call in (node for node in ast.walk(module) if isinstance(node, ast.Call)):
            forbidden_keywords = sorted(
                keyword.arg
                for keyword in call.keywords
                if keyword.arg in FORBIDDEN_OPENAI_KEYWORDS
            )
            if forbidden_keywords:
                errors.append(
                    f"{relative} must not expose tool/function call keywords: "
                    + ", ".join(forbidden_keywords)
                )

    native_path = root / "app/evaluation/native_judge.py"
    if native_path in parsed:
        native_source = parsed[native_path][1]
        for marker in (
            '"role": "system"',
            '"role": "user"',
            "self.client.chat.completions.create",
            "json.dumps(payload",
        ):
            if marker not in native_source:
                errors.append(f"native text judge is missing required boundary marker: {marker}")

    try:
        workflow = (root.parent / ".github/workflows/release.yml").read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"release workflow is unreadable: {exc}")
    else:
        if re.search(r"(?m)^  ragas-judge:\s*$", workflow) is None:
            errors.append("release workflow must provide the isolated ragas-judge job")
    return errors


def validate_reachability_contract(root: Path) -> list[str]:
    """Backward-compatible name for the evaluation boundary validation."""
    return validate_evaluation_boundary(root)


def validate_exception_policy(
    root: Path,
    *,
    now: object | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    del now
    errors = validate_evaluation_boundary(root)
    policy_path = root / POLICY_PATH
    try:
        document = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [*errors, f"dependency exception policy is unreadable: {exc}"], []
    if document.get("schema_version") != 2:
        errors.append("dependency exception policy schema_version must be 2")
    entries = document.get("exceptions")
    if entries != []:
        errors.append("dependency exception policy must contain exactly exceptions = []")
        return errors, entries if isinstance(entries, list) else []
    return errors, []


def validate_upstream_no_fix(
    entries: list[dict[str, Any]],
    **_kwargs: object,
) -> list[str]:
    if entries:
        return ["dependency vulnerability exceptions are no longer supported"]
    return []


def audit_lock_files(
    root: Path,
    entries: list[dict[str, Any]] | None = None,
    *,
    output_dir: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    if entries:
        return ["dependency vulnerability exceptions are no longer supported"]
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for lock_name in AUDIT_LOCKS:
        command = [
            sys.executable,
            "-m",
            "pip_audit",
            "--strict",
            "--no-deps",
            "--disable-pip",
            "--progress-spinner=off",
            "--vulnerability-service",
            "osv",
            "-r",
            lock_name,
        ]
        if output_dir is not None:
            command.extend(
                ("--format", "json", "--output", str(output_dir / f"{lock_name}.audit.json"))
            )
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode:
            errors.append(f"pip-audit rejected {lock_name} with exit code {result.returncode}")
        else:
            print(f"Dependency audit passed: {lock_name}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-only", action="store_true")
    parser.add_argument("--check-upstream", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    errors, entries = validate_exception_policy(root)
    if args.check_upstream:
        errors.extend(validate_upstream_no_fix(entries))
    if not args.policy_only:
        errors.extend(audit_lock_files(root, entries, output_dir=args.output_dir))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print("Dependency audit policy is valid with no vulnerability exceptions.")


if __name__ == "__main__":
    main()
