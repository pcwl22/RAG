"""Regression tests for the dependency-minimal evaluation command."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluate_ragas import _normalize_openai_base_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_ragas.py"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://judge.example", "https://judge.example/v1"),
        ("https://judge.example/compatible/v1/", "https://judge.example/compatible/v1"),
    ],
)
def test_normalize_openai_base_url(value: str, expected: str) -> None:
    assert _normalize_openai_base_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "judge.example/v1",
        "ftp://judge.example/v1",
        "https://user:secret@judge.example/v1",
        "https://judge.example/v1?tenant=unsafe",
        "https://judge.example/v1#unsafe",
    ],
)
def test_normalize_openai_base_url_rejects_unsafe_value(value: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        _normalize_openai_base_url(value)


def test_evaluation_cli_help_runs_without_site_packages(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "protected text-only judge" in result.stdout
