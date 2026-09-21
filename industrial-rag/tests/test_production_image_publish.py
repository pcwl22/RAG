import base64
import json
import re
from pathlib import Path

import pytest
import yaml

from scripts.production_image_publish import (
    finalize_publish,
    prepare_publish,
    verify_attestation_output,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "Example/Industrial-RAG"
REVISION = "1" * 40
MODEL_DIGEST = "sha256:" + "a" * 64
MANIFEST_DIGEST = "sha256:" + "b" * 64
APPROVED_MODEL_IMAGE = f"ghcr.io/{REPOSITORY.lower()}/approved/model-bundle@{MODEL_DIGEST}"
MODEL_IMAGE = f"ghcr.io/{REPOSITORY.lower()}/model-bundle@sha256:" + "f" * 64
MODEL_SOURCE = "https://github.com/Model-Owner/Model-Builds"
MODEL_SOURCE_REVISION = "2" * 40
MODEL_IDENTITY = MODEL_SOURCE + "/.github/workflows/model.yml@refs/tags/model-v1"


def model_contract():
    return {
        "approved_model_bundle_image": APPROVED_MODEL_IMAGE,
        "model_bundle_certificate_identity": MODEL_IDENTITY,
        "model_bundle_source_repository": MODEL_SOURCE,
        "model_bundle_source_revision": MODEL_SOURCE_REVISION,
    }


def test_prepare_publish_accepts_only_main_and_returns_safe_exact_values():
    outputs = prepare_publish(
        **model_contract(),
        model_manifest_sha256=MANIFEST_DIGEST,
        github_repository=REPOSITORY,
        github_sha=REVISION,
        github_ref="refs/heads/main",
    )

    assert outputs["api_tag"] == f"ghcr.io/{REPOSITORY.lower()}/api:git-{REVISION}"
    assert outputs["api_repository"] == f"ghcr.io/{REPOSITORY.lower()}/api"
    assert outputs["approved_model_bundle_digest"] == MODEL_DIGEST
    assert outputs["source_revision"] == REVISION
    assert outputs["publisher_identity"] == (
        f"https://github.com/{REPOSITORY}/.github/workflows/"
        "publish-production-images.yml@refs/heads/main"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"github_ref": "refs/heads/feature"}, "only be published"),
        (
            {"approved_model_bundle_image": "registry.models.test/model:latest"},
            "safe immutable OCI",
        ),
        (
            {"approved_model_bundle_image": ("ghcr.io/another-owner/approved/model-bundle:latest")},
            "safe immutable OCI",
        ),
        (
            {"model_manifest_sha256": "sha256:" + "0" * 64},
            "zero placeholder",
        ),
        (
            {"model_manifest_sha256": "sha256:" + "A" * 64},
            "lowercase hexadecimal",
        ),
        (
            {"github_repository": "example/repo\nINJECTED=value"},
            "single-line",
        ),
    ],
)
def test_prepare_publish_rejects_mutable_or_untrusted_input(overrides, message):
    arguments = {
        **model_contract(),
        "model_manifest_sha256": MANIFEST_DIGEST,
        "github_repository": REPOSITORY,
        "github_sha": REVISION,
        "github_ref": "refs/heads/main",
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        prepare_publish(**arguments)


def test_prepare_publish_accepts_exact_internal_model_workflow_identity():
    source = f"https://github.com/{REPOSITORY}"
    outputs = prepare_publish(
        approved_model_bundle_image=APPROVED_MODEL_IMAGE,
        model_manifest_sha256=MANIFEST_DIGEST,
        model_bundle_certificate_identity=(
            source + "/.github/workflows/model-build.yml@refs/heads/model-release"
        ),
        model_bundle_source_repository=source.lower(),
        model_bundle_source_revision=MODEL_SOURCE_REVISION,
        github_repository=REPOSITORY,
        github_sha=REVISION,
        github_ref="refs/heads/main",
    )

    assert outputs["model_bundle_source_repository"] == source.lower()


def test_prepare_publish_accepts_digest_pinned_external_model_registry():
    external_image = "registry.models.corp/approved/model-bundle@" + MODEL_DIGEST
    outputs = prepare_publish(
        **{
            **model_contract(),
            "approved_model_bundle_image": external_image,
        },
        model_manifest_sha256=MANIFEST_DIGEST,
        github_repository=REPOSITORY,
        github_sha=REVISION,
        github_ref="refs/heads/main",
    )

    assert outputs["approved_model_bundle_image"] == external_image
    assert outputs["model_bundle_repository"] == (f"ghcr.io/{REPOSITORY.lower()}/model-bundle")


@pytest.mark.parametrize(
    "identity",
    [
        MODEL_SOURCE + "/.github/workflows/*.yml@refs/tags/model-v1",
        MODEL_SOURCE + "/.github/workflows/model.yml@refs/tags/../main",
    ],
)
def test_prepare_publish_rejects_non_exact_model_workflow_identity(identity):
    with pytest.raises(ValueError, match="workflow identity|invalid Git ref"):
        prepare_publish(
            **{
                **model_contract(),
                "model_bundle_certificate_identity": identity,
            },
            model_manifest_sha256=MANIFEST_DIGEST,
            github_repository=REPOSITORY,
            github_sha=REVISION,
            github_ref="refs/heads/main",
        )


def test_finalize_publish_emits_digest_release_values_and_slsa_v1_predicates(tmp_path):
    images = {
        "api_image": f"ghcr.io/{REPOSITORY.lower()}/api@sha256:" + "c" * 64,
        "worker_image": f"ghcr.io/{REPOSITORY.lower()}/worker@sha256:" + "d" * 64,
        "frontend_image": f"ghcr.io/{REPOSITORY.lower()}/frontend@sha256:" + "e" * 64,
    }
    outputs = finalize_publish(
        **images,
        **model_contract(),
        model_bundle_image=MODEL_IMAGE,
        model_manifest_sha256=MANIFEST_DIGEST,
        github_repository=REPOSITORY,
        github_sha=REVISION,
        github_ref="refs/heads/main",
        github_run_url=f"https://github.com/{REPOSITORY}/actions/runs/123/attempts/1",
        output_dir=tmp_path,
    )

    release_env = (tmp_path / "release-images.env").read_text(encoding="utf-8")
    assert f"API_IMAGE={images['api_image']}\n" in release_env
    assert f"MODEL_BUNDLE_IMAGE={MODEL_IMAGE}\n" in release_env
    assert ":git-" not in release_env
    assert outputs["api"] == images["api_image"]

    release_record = json.loads((tmp_path / "release-images.json").read_text())
    assert release_record["source_revision"] == REVISION
    assert release_record["model_bundle"]["manifest_sha256"] == MANIFEST_DIGEST
    assert release_record["model_bundle"]["approved_source"]["image"] == (APPROVED_MODEL_IMAGE)
    assert release_record["images"]["worker"] == images["worker_image"]

    for product in ("model-bundle", "api", "worker", "frontend"):
        provenance = json.loads((tmp_path / f"{product}-slsa-provenance.json").read_text())
        definition = provenance["buildDefinition"]
        assert definition["externalParameters"]["product"] == product
        assert definition["externalParameters"]["sourceRevision"] == REVISION
        assert definition["externalParameters"]["modelSignerIdentity"] == MODEL_IDENTITY
        if product == "model-bundle":
            assert len(definition["resolvedDependencies"]) == 2
            assert definition["resolvedDependencies"][1]["uri"] == APPROVED_MODEL_IMAGE
        else:
            assert definition["resolvedDependencies"][1]["uri"] == MODEL_IMAGE
            assert definition["resolvedDependencies"][2]["uri"] == APPROVED_MODEL_IMAGE
        assert provenance["runDetails"]["builder"]["id"] == outputs["publisher_identity"]


def test_finalize_publish_rejects_a_cross_repository_application_digest(tmp_path):
    with pytest.raises(ValueError, match="immutable ghcr.io/example/industrial-rag/api"):
        finalize_publish(
            api_image="ghcr.io/attacker/repo/api@sha256:" + "c" * 64,
            worker_image=f"ghcr.io/{REPOSITORY.lower()}/worker@sha256:" + "d" * 64,
            frontend_image=f"ghcr.io/{REPOSITORY.lower()}/frontend@sha256:" + "e" * 64,
            **model_contract(),
            model_bundle_image=MODEL_IMAGE,
            model_manifest_sha256=MANIFEST_DIGEST,
            github_repository=REPOSITORY,
            github_sha=REVISION,
            github_ref="refs/heads/main",
            github_run_url=f"https://github.com/{REPOSITORY}/actions/runs/123",
            output_dir=tmp_path,
        )


def test_finalize_publish_rejects_a_non_exact_workflow_run_url(tmp_path):
    with pytest.raises(ValueError, match="github_run_url"):
        finalize_publish(
            api_image=f"ghcr.io/{REPOSITORY.lower()}/api@sha256:" + "c" * 64,
            worker_image=f"ghcr.io/{REPOSITORY.lower()}/worker@sha256:" + "d" * 64,
            frontend_image=f"ghcr.io/{REPOSITORY.lower()}/frontend@sha256:" + "e" * 64,
            **model_contract(),
            model_bundle_image=MODEL_IMAGE,
            model_manifest_sha256=MANIFEST_DIGEST,
            github_repository=REPOSITORY,
            github_sha=REVISION,
            github_ref="refs/heads/main",
            github_run_url=(f"https://github.com/{REPOSITORY}/actions/runs/123/attacker-suffix"),
            output_dir=tmp_path,
        )


def test_verify_attestation_output_binds_exact_digest_type_and_predicate(tmp_path):
    predicate = {"bomFormat": "CycloneDX", "specVersion": "1.6"}
    predicate_path = tmp_path / "sbom.json"
    predicate_path.write_text(json.dumps(predicate), encoding="utf-8")
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": "api", "digest": {"sha256": "c" * 64}}],
        "predicateType": "https://cyclonedx.org/bom",
        "predicate": predicate,
    }
    envelope = {
        "payloadType": "application/vnd.in-toto+json",
        "payload": base64.b64encode(json.dumps(statement).encode()).decode(),
        "signatures": [{"sig": "already-verified-by-cosign"}],
    }
    verification_path = tmp_path / "verification.json"
    verification_path.write_text(json.dumps(envelope), encoding="utf-8")

    verify_attestation_output(
        verification_output=verification_path,
        expected_predicate=predicate_path,
        expected_image=f"ghcr.io/{REPOSITORY.lower()}/api@sha256:" + "c" * 64,
        expected_predicate_type="https://cyclonedx.org/bom",
    )

    statement["subject"][0]["digest"]["sha256"] = "d" * 64
    envelope["payload"] = base64.b64encode(json.dumps(statement).encode()).decode()
    verification_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="no verified attestation binds"):
        verify_attestation_output(
            verification_output=verification_path,
            expected_predicate=predicate_path,
            expected_image=f"ghcr.io/{REPOSITORY.lower()}/api@sha256:" + "c" * 64,
            expected_predicate_type="https://cyclonedx.org/bom",
        )


def test_production_publish_workflow_keeps_build_scan_and_signing_contract():
    path = REPOSITORY_ROOT / ".github/workflows/publish-production-images.yml"
    workflow = path.read_text(encoding="utf-8")
    parsed = yaml.load(workflow, Loader=yaml.BaseLoader)

    assert parsed["on"]["workflow_dispatch"]["inputs"]
    assert parsed["permissions"] == {
        "contents": "read",
        "packages": "write",
        "id-token": "write",
    }
    assert "production-image-publish" in workflow
    assert "industrial-rag/docker/api/Dockerfile.production" in workflow
    assert "industrial-rag/docker/worker/Dockerfile.production" in workflow
    assert "industrial-rag/docker/model-bundle/Dockerfile.production" in workflow
    assert "rag-frontend/Dockerfile.production" in workflow
    assert "--certificate-identity-regexp" not in workflow
    assert '--certificate-identity "$PUBLISHER_IDENTITY"' in workflow
    assert 'cosign attest --yes --type "$CYCLONEDX_PREDICATE_TYPE"' in workflow
    assert 'cosign attest --yes --type "$SLSA_PROVENANCE_PREDICATE_TYPE"' in workflow
    assert "https://cyclonedx.org/bom" in workflow
    assert "https://slsa.dev/provenance/v1" in workflow
    assert 'cosign-release: "v3.1.2"' in workflow
    assert 'sha256sum "$PUBLISH_ARTIFACT_DIR/model/mirror-manifest.json"' not in workflow
    assert "MODEL_BUNDLE_SOURCE_REVISION" in workflow
    assert workflow.count('--certificate-github-workflow-sha "$EXPECTED_REVISION"') == 5
    assert "Scan repository-bound production model image" in workflow
    assert "Generate repository-bound production model SBOM" in workflow
    assert '"model-bundle=$MODEL_BUNDLE_IMAGE"' in workflow
    assert '"$MODEL_BUNDLE_IMAGE")" = "$EXPECTED_REVISION"' in workflow
    assert workflow.index("Scan production API image") < workflow.index(
        "Attest SBOM and SLSA provenance"
    )
    assert workflow.index("SLSA_PROVENANCE_PREDICATE_TYPE") < workflow.index("cosign sign --yes")
    assert "verify-attestation-output" in workflow
    atomic_step = workflow.index("Create, attest, and verify atomic production image set")
    per_image_verification = workflow.index(
        "Verify signatures and attestations against this exact workflow identity"
    )
    assert per_image_verification < atomic_step
    assert "release_set_attestation.py create" in workflow[atomic_step:]
    assert "release_set_attestation.py verify-attestation" in workflow[atomic_step:]
    assert 'cosign attest --yes --type "$image_set_type"' in workflow[atomic_step:]
    assert "API_IMAGE: ${{ steps.prepare.outputs.api_repository }}@" in workflow
    assert "SOURCE_REVISION=${{ steps.prepare.outputs.source_revision }}" in workflow
    assert "provenance: false" in workflow
    assert "sbom: false" in workflow
    publication_upload = workflow.index("Upload production image publication evidence")
    failed_upload = workflow.index("Upload failed-publication diagnostics")
    assert "if: success()" in workflow[publication_upload:failed_upload]
    assert "release-images.env" not in workflow[failed_upload:]

    action_uses = re.findall(r"^\s*uses:\s*([^\s#]+)", workflow, flags=re.MULTILINE)
    assert action_uses
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", value) for value in action_uses)

    model_dockerfile = (
        REPOSITORY_ROOT / "industrial-rag/docker/model-bundle/Dockerfile.production"
    ).read_text(encoding="utf-8")
    assert "ARG APPROVED_MODEL_BUNDLE_IMAGE" in model_dockerfile
    assert "org.opencontainers.image.revision=$SOURCE_REVISION" in model_dockerfile
    assert "org.opencontainers.image.source=$SOURCE_REPOSITORY" in model_dockerfile
    assert "COPY --from=approved_model /models/ /models/" in model_dockerfile


@pytest.mark.parametrize(
    "relative_path",
    [
        "rag-frontend/Dockerfile.production",
        "industrial-rag/docker/frontend.Dockerfile.production",
    ],
)
def test_production_frontend_uses_a_writable_non_root_pid_path(relative_path):
    dockerfile = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")

    assert "USER 101:101" in dockerfile
    assert "/tmp/nginx.pid" in dockerfile
