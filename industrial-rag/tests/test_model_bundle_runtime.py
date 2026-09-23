"""Tests for the runtime model-bundle integrity boundary."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from app.embedding.model_bundle import (
    _tree_sha256,
    prepare_runtime_model,
    validate_runtime_model_manifest,
)


def _config(model_path: str) -> dict[str, object]:
    return {"embedding": {"model_path": model_path}}


def test_local_profile_may_omit_model_manifest_digest(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_ENV", "laptop")
    monkeypatch.delenv("MODEL_MANIFEST_SHA256", raising=False)
    assert validate_runtime_model_manifest(_config(str(tmp_path / "bge-m3"))) is None


def test_base_profile_requires_model_manifest_digest(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_ENV", "base")
    monkeypatch.delenv("MODEL_MANIFEST_SHA256", raising=False)
    with pytest.raises(RuntimeError, match="is required"):
        validate_runtime_model_manifest(_config(str(tmp_path / "bge-m3")))


def test_runtime_manifest_digest_must_match(monkeypatch, tmp_path):
    models = tmp_path / "models"
    model = models / "bge-m3"
    model.mkdir(parents=True)
    manifest = models / "model-manifest.json"
    manifest.write_bytes(b'{"schema_version":1}\n')
    expected = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()

    monkeypatch.setenv("RAG_ENV", "base")
    monkeypatch.setenv("MODEL_MANIFEST_SHA256", expected)
    assert validate_runtime_model_manifest(_config(str(model))) == manifest.resolve()

    monkeypatch.setenv("MODEL_MANIFEST_SHA256", "sha256:" + "0" * 64)
    with pytest.raises(RuntimeError, match="does not match"):
        validate_runtime_model_manifest(_config(str(model)))


def test_explicit_invalid_digest_is_rejected_in_local_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_ENV", "laptop")
    monkeypatch.setenv("MODEL_MANIFEST_SHA256", "not-a-digest")
    with pytest.raises(RuntimeError, match="64 hex"):
        validate_runtime_model_manifest(_config(str(tmp_path / "bge-m3")))


def _download_contract(tmp_path: Path, monkeypatch) -> tuple[dict[str, object], Path]:
    target = tmp_path / "models" / "bge-m3"
    expected_tree = tmp_path / "expected"
    expected_tree.mkdir()
    (expected_tree / "config.json").write_text('{"model": true}\n', encoding="utf-8")

    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "model-sources/model-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["models"]["bge-m3"]["sha256"] = _tree_sha256(expected_tree)
    manifest_path = tmp_path / "model-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_ENV", "base")
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv(
        "MODEL_MANIFEST_SHA256",
        "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    config: dict[str, object] = {
        "embedding": {
            "model_name": "BAAI/bge-m3",
            "model_revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "model_path": str(target),
        }
    }
    return config, target


def test_runtime_downloads_exact_revision_validates_tree_and_reuses_marker(
    monkeypatch, tmp_path
):
    config, target = _download_contract(tmp_path, monkeypatch)
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        (local_dir / ".cache/huggingface").mkdir(parents=True)
        (local_dir / ".cache/huggingface/download.json").write_text("{}")
        (local_dir / "config.json").write_text('{"model": true}\n', encoding="utf-8")
        return str(local_dir)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )

    assert prepare_runtime_model(config, "bge-m3", force_download=True) == target.resolve()
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "BAAI/bge-m3"
    assert calls[0]["revision"] == "5617a9f61b028005a4858fdac845db406aefb181"
    assert calls[0]["endpoint"] == "https://huggingface.co"
    assert calls[0]["token"] is False
    assert not (target / ".cache").exists()
    marker = json.loads((target / ".industrial-rag-model.json").read_text())
    assert marker["download_url"].endswith(
        "/tree/5617a9f61b028005a4858fdac845db406aefb181"
    )

    assert prepare_runtime_model(config, "bge-m3") == target.resolve()
    assert len(calls) == 1


def test_runtime_rejects_changed_download_source_before_network(monkeypatch, tmp_path):
    config, _target = _download_contract(tmp_path, monkeypatch)
    manifest_path = Path(os.environ["MODEL_MANIFEST_PATH"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"]["bge-m3"]["source_url"] = "https://attacker.invalid/BAAI/bge-m3"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    monkeypatch.setenv(
        "MODEL_MANIFEST_SHA256",
        "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(RuntimeError, match="source_url"):
        prepare_runtime_model(config, "bge-m3", force_download=True)
