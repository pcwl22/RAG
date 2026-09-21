import base64
import copy
import json
from pathlib import Path

import pytest

from scripts import release_set_attestation
from scripts.production_image_publish import (
    _release_set_predicate_type as publication_release_set_predicate_type,
)
from scripts.release_set_attestation import (
    COMPONENTS,
    MATERIAL_PATHS,
    build_predicate,
    predicate_type,
    validate_predicate,
    verify_attestation,
    write_predicate,
)

REPOSITORY = "Example/Industrial-RAG"
NORMALIZED_REPOSITORY = REPOSITORY.lower()
REVISION = "1" * 40
RUN = f"https://github.com/{REPOSITORY}/actions/runs/123/attempts/2"
IDENTITY = (
    f"https://github.com/{REPOSITORY}/.github/workflows/"
    "publish-production-images.yml@refs/heads/main"
)
MODEL_MANIFEST = "sha256:" + "a" * 64


def _images() -> dict[str, str]:
    return {
        "model_bundle_image": (
            f"ghcr.io/{NORMALIZED_REPOSITORY}/model-bundle@sha256:" + "b" * 64
        ),
        "api_image": f"ghcr.io/{NORMALIZED_REPOSITORY}/api@sha256:" + "c" * 64,
        "worker_image": f"ghcr.io/{NORMALIZED_REPOSITORY}/worker@sha256:" + "d" * 64,
        "frontend_image": f"ghcr.io/{NORMALIZED_REPOSITORY}/frontend@sha256:" + "e" * 64,
    }


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for group, paths in MATERIAL_PATHS.items():
        for index, relative in enumerate(paths):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{group}:{index}:{relative}\n".encode())
    return root


def _arguments(root: Path) -> dict[str, object]:
    return {
        "project_root": root,
        "github_repository": REPOSITORY,
        "source_revision": REVISION,
        "workflow_run": RUN,
        "publisher_identity": IDENTITY,
        **_images(),
        "model_manifest_sha256": MODEL_MANIFEST,
    }


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _envelope(predicate: dict[str, object], *, subjects=None, predicate_uri=None):
    api = _images()["api_image"]
    image_name, digest = api.rsplit("@sha256:", 1)
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": (
            [{"name": image_name, "digest": {"sha256": digest}}]
            if subjects is None
            else subjects
        ),
        "predicateType": predicate_uri or predicate_type(REPOSITORY),
        "predicate": predicate,
    }
    return {
        "payloadType": "application/vnd.in-toto+json",
        "payload": base64.b64encode(_canonical(statement).rstrip(b"\n")).decode(),
        "signatures": [{"sig": "already-verified-by-cosign"}],
    }


def test_build_write_and_validate_canonical_release_set(tmp_path):
    root = _project(tmp_path)
    predicate = build_predicate(**_arguments(root))
    output = tmp_path / "evidence" / "release-set.json"
    write_predicate(output, predicate)

    assert output.read_bytes() == _canonical(predicate)
    assert predicate["schema"] == (
        "https://github.com/example/industrial-rag/attestations/image-set/v1"
    )
    assert [component["name"] for component in predicate["components"]] == list(COMPONENTS)
    assert predicate["source"] == {
        "repository": "https://github.com/example/industrial-rag",
        "revision": REVISION,
        "workflowRun": (
            "https://github.com/example/industrial-rag/actions/runs/123/attempts/2"
        ),
        "publisherIdentity": (
            f"https://github.com/{REPOSITORY}/.github/workflows/"
            "publish-production-images.yml@refs/heads/main"
        ),
    }
    assert set(predicate["materials"]) == set(MATERIAL_PATHS)
    assert validate_predicate(predicate_path=output, **_arguments(root)) == predicate


def test_atomic_image_set_schema_cannot_alias_publication_release_set_schema():
    assert predicate_type(REPOSITORY).endswith("/attestations/image-set/v1")
    assert publication_release_set_predicate_type(
        f"https://github.com/{REPOSITORY}"
    ).endswith("/attestations/release-set/v1")
    assert predicate_type(REPOSITORY) != publication_release_set_predicate_type(
        f"https://github.com/{REPOSITORY}"
    )


def test_atomic_writer_preserves_previous_predicate_if_replace_fails(tmp_path, monkeypatch):
    output = tmp_path / "image-set.json"
    output.write_bytes(b"previous\n")

    def fail_replace(_source, _destination):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(release_set_attestation.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        write_predicate(output, {"new": True})

    assert output.read_bytes() == b"previous\n"
    assert list(tmp_path.glob(".image-set.json.*.tmp")) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("api_image", "ghcr.io/example/industrial-rag/api:latest", "immutable"),
        (
            "worker_image",
            "ghcr.io/other/industrial-rag/worker@sha256:" + "d" * 64,
            "immutable",
        ),
        (
            "frontend_image",
            "docker.io/example/industrial-rag/frontend@sha256:" + "e" * 64,
            "immutable",
        ),
        (
            "model_bundle_image",
            f"ghcr.io/{NORMALIZED_REPOSITORY}/model-bundle@sha256:bad",
            "digest",
        ),
        ("source_revision", "A" * 40, "Git SHA"),
        ("model_manifest_sha256", "sha256:" + "0" * 64, "non-zero"),
    ],
)
def test_builder_rejects_mutable_cross_repository_or_malformed_inputs(
    tmp_path, field, value, message
):
    arguments = _arguments(_project(tmp_path))
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        build_predicate(**arguments)


def test_builder_accepts_only_protected_image_publisher_identity(tmp_path):
    identity = (
        f"https://github.com/{REPOSITORY}/.github/workflows/"
        "publish-production-images.yml@refs/heads/main"
    )
    arguments = _arguments(_project(tmp_path))
    arguments["publisher_identity"] = identity
    assert build_predicate(**arguments)["source"]["publisherIdentity"].endswith(
        identity.split(f"github.com/{REPOSITORY}", 1)[1]
    )


@pytest.mark.parametrize(
    "identity",
    [
        f"https://github.com/{REPOSITORY}/.github/workflows/evil.yml@refs/heads/main",
        f"https://github.com/{REPOSITORY}/.github/workflows/release.yml@refs/heads/main",
        f"https://github.com/{REPOSITORY}/.github/workflows/release.yml@refs/tags/v1.2.3",
        f"https://github.com/{REPOSITORY}/.github/workflows/release.yml@refs/heads/feature",
        f"https://github.com/{REPOSITORY}/.github/workflows/release.yml@refs/tags/test-1",
    ],
)
def test_builder_rejects_unapproved_workflow_identity_forms(tmp_path, identity):
    arguments = _arguments(_project(tmp_path))
    arguments["publisher_identity"] = identity
    with pytest.raises(ValueError, match="protected production image publisher"):
        build_predicate(**arguments)


def test_publication_image_set_does_not_bind_mutable_quality_approval(tmp_path):
    root = _project(tmp_path)
    baseline = root / "eval/release_baseline.json"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text('{"status":"blocked"}\n', encoding="utf-8")

    first = build_predicate(**_arguments(root))
    baseline.write_text('{"status":"approved"}\n', encoding="utf-8")
    second = build_predicate(**_arguments(root))

    assert "eval/release_baseline.json" not in MATERIAL_PATHS["evaluation"]
    assert first == second


def test_validator_rejects_duplicate_or_missing_components_and_unknown_fields(tmp_path):
    root = _project(tmp_path)
    arguments = _arguments(root)
    predicate = build_predicate(**arguments)
    output = tmp_path / "release-set.json"

    for mutate in (
        lambda value: value["components"].append(copy.deepcopy(value["components"][1])),
        lambda value: value["components"].pop(),
        lambda value: value.update({"untrusted": True}),
        lambda value: value["model"].update({"untrusted": True}),
        lambda value: value.update({"schemaVersion": True}),
    ):
        candidate = copy.deepcopy(predicate)
        mutate(candidate)
        write_predicate(output, candidate)
        with pytest.raises(ValueError, match="does not match"):
            validate_predicate(predicate_path=output, **arguments)


def test_validator_rejects_digest_sha_material_and_noncanonical_mismatches(tmp_path):
    root = _project(tmp_path)
    arguments = _arguments(root)
    predicate = build_predicate(**arguments)
    output = tmp_path / "release-set.json"

    candidate = copy.deepcopy(predicate)
    candidate["components"][1]["digest"] = "sha256:" + "f" * 64
    write_predicate(output, candidate)
    with pytest.raises(ValueError, match="does not match"):
        validate_predicate(predicate_path=output, **arguments)

    write_predicate(output, predicate)
    with pytest.raises(ValueError, match="does not match"):
        validate_predicate(
            predicate_path=output,
            **{**arguments, "source_revision": "2" * 40},
        )

    material = root / MATERIAL_PATHS["contracts"][0]
    material.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_predicate(predicate_path=output, **arguments)

    material.write_bytes(b"contracts:0:app/llm/answer_contract.py\n")
    output.write_text(json.dumps(predicate, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSON"):
        validate_predicate(predicate_path=output, **arguments)


def test_validator_rejects_duplicate_json_keys_and_missing_material(tmp_path):
    root = _project(tmp_path)
    arguments = _arguments(root)
    output = tmp_path / "release-set.json"
    output.write_text('{"schema":1,"schema":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate object key"):
        validate_predicate(predicate_path=output, **arguments)

    (root / MATERIAL_PATHS["evaluation"][0]).unlink()
    with pytest.raises(ValueError, match="missing"):
        build_predicate(**arguments)


def test_verify_attestation_accepts_one_exact_api_bound_statement(tmp_path):
    root = _project(tmp_path)
    arguments = _arguments(root)
    predicate = build_predicate(**arguments)
    verification = tmp_path / "cosign.json"
    verification.write_text(json.dumps(_envelope(predicate)), encoding="utf-8")

    statement = verify_attestation(verification_output=verification, **arguments)

    assert statement["predicate"] == predicate


@pytest.mark.parametrize(
    "mutation",
    ["subject", "missing-subject", "duplicate-subject", "predicate", "type"],
)
def test_verify_attestation_rejects_subject_or_predicate_substitution(tmp_path, mutation):
    root = _project(tmp_path)
    arguments = _arguments(root)
    predicate = build_predicate(**arguments)
    image_name = _images()["api_image"].split("@", 1)[0]
    subject = {"name": image_name, "digest": {"sha256": "9" * 64}}
    envelope = _envelope(predicate)
    if mutation == "subject":
        envelope = _envelope(predicate, subjects=[subject])
    elif mutation == "missing-subject":
        envelope = _envelope(predicate, subjects=[])
    elif mutation == "duplicate-subject":
        valid_subject = _envelope(predicate)
        statement = json.loads(base64.b64decode(valid_subject["payload"]))
        envelope = _envelope(predicate, subjects=statement["subject"] * 2)
    elif mutation == "predicate":
        altered = copy.deepcopy(predicate)
        altered["source"]["workflowRun"] = (
            "https://github.com/example/industrial-rag/actions/runs/999/attempts/1"
        )
        envelope = _envelope(altered)
    else:
        envelope = _envelope(predicate, predicate_uri="https://example.invalid/release-set/v1")
    verification = tmp_path / "cosign.json"
    verification.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one verified"):
        verify_attestation(verification_output=verification, **arguments)


def test_verify_attestation_rejects_duplicate_matching_envelopes(tmp_path):
    root = _project(tmp_path)
    arguments = _arguments(root)
    predicate = build_predicate(**arguments)
    envelope = _envelope(predicate)
    verification = tmp_path / "cosign.json"
    verification.write_text(json.dumps([envelope, envelope]), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one verified"):
        verify_attestation(verification_output=verification, **arguments)
