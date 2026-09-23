"""Security contract for the protected Hugging Face source publisher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT.parent / ".github/workflows/publish-model-bundle.yml"
MANIFEST_PATH = PROJECT_ROOT / "model-sources/model-manifest.json"


def _workflow() -> tuple[str, dict[str, Any]]:
    source = WORKFLOW_PATH.read_text(encoding="utf-8")
    payload = yaml.load(source, Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return source, payload


def test_model_bundle_publisher_is_manual_and_least_privilege() -> None:
    _source, workflow = _workflow()

    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {
        "contents": "read",
        "packages": "write",
        "id-token": "write",
    }
    assert workflow["on"]["workflow_dispatch"] == ""


def test_model_source_publisher_uses_a_protected_hosted_runner() -> None:
    source, workflow = _workflow()
    job = workflow["jobs"]["publish"]

    assert job["environment"]["name"] == "production-model-bundle"
    assert job["runs-on"] == "ubuntu-24.04"
    assert 'test "$GITHUB_REF" = "refs/heads/main"' in source
    assert "pull_request" not in workflow["on"]
    assert "push" not in workflow["on"]
    assert "self-hosted" not in source
    assert "/opt/rag-models" not in source


def test_model_source_publisher_pins_huggingface_revisions_without_weights() -> None:
    source, workflow = _workflow()
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["hub"] == "https://huggingface.co"
    assert manifest["models"]["bge-m3"]["revision"] == (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert manifest["models"]["bge-reranker-v2-m3"]["revision"] == (
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    )
    assert "build_model_bundle_manifest.py" in source
    assert "validate_model_bundle_manifest.py" in source
    assert "approved-model-source" in source
    assert "io.industrial-rag.model-weights" in source
    assert "contains_model_weights" in source
    assert "unexpectedly contains model weights" in source

    dockerfile = (PROJECT_ROOT / "docker/model-bundle/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "COPY model-manifest.json /models/model-manifest.json" in dockerfile
    assert "COPY models/bge-m3" not in dockerfile
    assert "COPY models/bge-reranker-v2-m3" not in dockerfile


def test_model_bundle_publisher_never_uses_mutable_release_references() -> None:
    source, workflow = _workflow()
    steps = workflow["jobs"]["publish"]["steps"]
    commands = "\n".join(str(step.get("run") or "") for step in steps)

    checkout = next(step for step in steps if str(step.get("uses") or "").startswith("actions/checkout@"))
    assert checkout["with"]["persist-credentials"] == "false"
    assert "docker login ghcr.io" in commands
    assert "--password-stdin" in commands
    assert "docker buildx build" in commands
    assert "--push" in commands
    assert 'image_ref="$image_repository@$image_digest"' in commands
    assert "cosign sign --yes \"$MODEL_BUNDLE_IMAGE\"" in commands
    assert "--certificate-github-workflow-sha \"$GITHUB_SHA\"" in commands
    assert "cosign verify" in commands
    assert "docker logout ghcr.io" in commands
    assert ":latest" not in source


def test_model_bundle_publisher_installs_checksum_pinned_cosign() -> None:
    source, workflow = _workflow()
    job = workflow["jobs"]["publish"]
    env = job["env"]
    steps = job["steps"]
    install = next(step for step in steps if "checksum-pinned Cosign" in step["name"])
    command = str(install["run"])

    assert env["COSIGN_VERSION"] == "v3.1.2"
    assert (
        env["COSIGN_LINUX_AMD64_SHA256"]
        == "f7622ed3cf22e55e1ae6377c080979ff77a22da9981c11df222a2e444991e7cf"
    )
    assert "releases/download/$COSIGN_VERSION/cosign-linux-amd64" in command
    assert "sha256sum --check --strict" in command
    assert "--proto '=https'" in command
    assert "sigstore/cosign-installer" not in source
    assert "+            --" not in source


def test_model_source_publisher_needs_no_local_runner_proxy() -> None:
    source, workflow = _workflow()
    job = workflow["jobs"]["publish"]
    buildx = next(
        step
        for step in job["steps"]
        if str(step.get("uses") or "").startswith("docker/setup-buildx-action@")
    )
    assert "with" not in buildx
    assert "BUILDKIT_PROXY_URL" not in job["env"]
    assert "host.docker.internal" not in source


def test_model_bundle_publisher_keeps_credentials_out_of_evidence() -> None:
    source, workflow = _workflow()
    steps = workflow["jobs"]["publish"]["steps"]
    upload = next(
        step
        for step in steps
        if str(step.get("uses") or "").startswith("actions/upload-artifact@")
    )

    assert upload["with"]["path"] == "${{ runner.temp }}/approved-model-source-evidence"
    assert "REGISTRY_TOKEN" not in upload["with"]["path"]
    assert "model-source.env" in source
    assert "RAGAS_JUDGE_API_KEY" not in source
    assert "DEEPSEEK_API_KEY" not in source
