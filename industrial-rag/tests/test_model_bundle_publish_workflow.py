"""Security contract for the protected Hugging Face model-bundle publisher."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT.parent / ".github/workflows/publish-model-bundle.yml"
ACTIONLINT_CONFIG_PATH = PROJECT_ROOT.parent / ".github/actionlint.yaml"


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
    dispatch = workflow["on"]["workflow_dispatch"]
    assert set(dispatch["inputs"]) == {"runner_label", "model_manifest_sha256"}
    assert all(value["required"] == "true" for value in dispatch["inputs"].values())


def test_model_bundle_publisher_uses_one_time_protected_runner() -> None:
    source, workflow = _workflow()
    job = workflow["jobs"]["publish"]

    assert job["environment"]["name"] == "production-model-bundle"
    assert set(job["runs-on"][:4]) == {"self-hosted", "Linux", "X64", "rag-model-bundle"}
    assert job["runs-on"][4] == "${{ inputs.runner_label }}"
    assert job["env"]["MODEL_SOURCE_ROOT"] == "/opt/rag-models"
    assert re.search(r"\^rag-model-bundle-\[0-9a-f\]\{32\}\$", source)
    assert 'test "$GITHUB_REF" = "refs/heads/main"' in source
    assert "pull_request" not in workflow["on"]
    assert "push" not in workflow["on"]
    actionlint = yaml.safe_load(ACTIONLINT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert "rag-model-bundle" in actionlint["self-hosted-runner"]["labels"]


def test_model_bundle_publisher_pins_huggingface_revisions_and_tree_manifest() -> None:
    source, workflow = _workflow()
    env = workflow["jobs"]["publish"]["env"]

    assert env["BGE_M3_MODEL_ID"] == "BAAI/bge-m3"
    assert env["BGE_M3_REVISION"] == "5617a9f61b028005a4858fdac845db406aefb181"
    assert env["BGE_RERANKER_MODEL_ID"] == "BAAI/bge-reranker-v2-m3"
    assert env["BGE_RERANKER_REVISION"] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert "build_model_bundle_manifest.py" in source
    assert "validate_model_bundle_manifest.py" in source
    assert "refs/remotes/origin/main" in source
    assert "--exclude='.git'" in source
    assert "--exclude='*/.git'" in source


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


def test_model_bundle_publisher_routes_buildkit_through_protected_local_proxy() -> None:
    source, workflow = _workflow()
    job = workflow["jobs"]["publish"]
    buildx = next(
        step
        for step in job["steps"]
        if str(step.get("uses") or "").startswith("docker/setup-buildx-action@")
    )
    driver_opts = set(str(buildx["with"]["driver-opts"]).splitlines())

    assert job["env"]["BUILDKIT_PROXY_URL"] == "${{ vars.BUILDKIT_PROXY_URL }}"
    assert driver_opts == {
        "env.http_proxy=${{ env.BUILDKIT_PROXY_URL }}",
        "env.https_proxy=${{ env.BUILDKIT_PROXY_URL }}",
        "env.HTTP_PROXY=${{ env.BUILDKIT_PROXY_URL }}",
        "env.HTTPS_PROXY=${{ env.BUILDKIT_PROXY_URL }}",
        "env.no_proxy=localhost,127.0.0.1",
        "env.NO_PROXY=localhost,127.0.0.1",
    }
    assert "^http://host\\.docker\\.internal:([0-9]{1,5})$" in source


def test_model_bundle_publisher_keeps_credentials_out_of_evidence() -> None:
    source, workflow = _workflow()
    steps = workflow["jobs"]["publish"]["steps"]
    upload = next(
        step
        for step in steps
        if str(step.get("uses") or "").startswith("actions/upload-artifact@")
    )

    assert upload["with"]["path"] == "${{ runner.temp }}/approved-model-bundle-evidence"
    assert "REGISTRY_TOKEN" not in upload["with"]["path"]
    assert "model-bundle.env" in source
    assert "RAGAS_JUDGE_API_KEY" not in source
    assert "DEEPSEEK_API_KEY" not in source
