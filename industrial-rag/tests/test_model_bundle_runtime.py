"""Tests for the runtime model-bundle integrity boundary."""

from __future__ import annotations

import hashlib

import pytest

from app.embedding.model_bundle import validate_runtime_model_manifest


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
