"""Validate and materialize the protected production-image publish contract.

The workflow deliberately delegates all values that later cross a shell/action
boundary to this module.  That keeps workflow-dispatch input validation in one
testable place and prevents mutable tags from becoming release inputs.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
from pathlib import Path

_GITHUB_REPOSITORY = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?"
)
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_OCI_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_IMMUTABLE_IMAGE = re.compile(
    r"[a-z0-9.-]+(?::[0-9]+)?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
    r"@(?P<digest>sha256:[0-9a-f]{64})"
)
_GITHUB_SOURCE_REPOSITORY = re.compile(
    r"https://github\.com/"
    r"(?P<repository>[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?)"
)
_GITHUB_WORKFLOW_IDENTITY = re.compile(
    r"https://github\.com/"
    r"(?P<repository>[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?)"
    r"/\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml"
    r"@refs/(?P<ref_kind>heads|tags)/(?P<ref>[A-Za-z0-9._/-]+)"
)
_GITHUB_RUN_URL = re.compile(
    r"https://github\.com/"
    r"(?P<repository>[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?)"
    r"/actions/runs/[1-9][0-9]*"
    r"(?:/attempts/[1-9][0-9]*)?"
)
_PRODUCTION_REF = "refs/heads/main"
_PUBLISH_WORKFLOW = ".github/workflows/publish-production-images.yml"
_PRODUCTS = ("api", "worker", "frontend")
_RELEASE_SET_PRODUCTS = ("model-bundle", "api", "worker", "frontend")
_RELEASE_SET_SCHEMA = "industrial-rag.release-set/v1"
_IN_TOTO_STATEMENT_TYPES = {
    "https://in-toto.io/Statement/v0.1",
    "https://in-toto.io/Statement/v1",
}
_PREDICATE_TYPES = {
    "https://cyclonedx.org/bom",
    "https://slsa.dev/provenance/v1",
}


def _release_set_predicate_type(source_repository: str) -> str:
    """Return the repository-owned URI for the release-set predicate."""
    normalized = _single_line("source_repository", source_repository).rstrip("/")
    match = _GITHUB_SOURCE_REPOSITORY.fullmatch(normalized)
    if match is None:
        raise ValueError("source_repository must be an exact GitHub repository URL")
    return normalized + "/attestations/release-set/v1"


def _single_line(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized or "\n" in normalized or "\r" in normalized:
        raise ValueError(f"{name} must be a non-empty single-line value")
    return normalized


def _repository_namespace(repository: str) -> str:
    normalized = _single_line("github_repository", repository)
    if _GITHUB_REPOSITORY.fullmatch(normalized) is None:
        raise ValueError("github_repository must be an owner/repository name")
    return normalized.lower()


def _git_sha(revision: str) -> str:
    normalized = _single_line("github_sha", revision)
    if _GIT_SHA.fullmatch(normalized) is None:
        raise ValueError("github_sha must be the full 40-character commit SHA")
    return normalized


def _digest(value: str, *, name: str) -> str:
    normalized = _single_line(name, value)
    if _OCI_DIGEST.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be sha256:<64 lowercase hexadecimal characters>")
    if normalized == "sha256:" + "0" * 64:
        raise ValueError(f"{name} must not use the zero placeholder digest")
    return normalized


def _expected_digest_reference(value: str, *, repository: str, product: str) -> str:
    normalized = _single_line(f"{product}_image", value)
    prefix = f"ghcr.io/{repository}/{product}@"
    if not normalized.startswith(prefix):
        raise ValueError(f"{product}_image must be an immutable {prefix}sha256:<digest> reference")
    digest = _digest(normalized.removeprefix(prefix), name=f"{product}_image digest")
    return prefix + digest


def _immutable_image_reference(value: str, *, name: str) -> str:
    normalized = _single_line(name, value)
    match = _IMMUTABLE_IMAGE.fullmatch(normalized)
    if match is None:
        raise ValueError(f"{name} must be a safe immutable OCI sha256 reference")
    _digest(match.group("digest"), name=f"{name} digest")
    return normalized


def _approved_model_reference(value: str) -> str:
    """Return a shell-safe upstream bundle reference pinned by OCI digest.

    The upstream model builder can live in a separately governed repository or
    registry.  Registry ownership is therefore not the trust boundary: the
    publish workflow verifies the exact keyless workflow identity, source
    repository/revision labels, manifest digest, and model-tree contents before
    copying the tree into this repository's own GHCR namespace.
    """
    return _immutable_image_reference(
        value,
        name="approved_model_bundle_image",
    )


def _model_source_contract(
    *,
    certificate_identity: str,
    source_repository: str,
    source_revision: str,
) -> dict[str, str]:
    source = _single_line("model_bundle_source_repository", source_repository)
    source_match = _GITHUB_SOURCE_REPOSITORY.fullmatch(source)
    if source_match is None:
        raise ValueError("model_bundle_source_repository must be an exact GitHub repository URL")
    identity = _single_line(
        "model_bundle_certificate_identity",
        certificate_identity,
    )
    identity_match = _GITHUB_WORKFLOW_IDENTITY.fullmatch(identity)
    if identity_match is None:
        raise ValueError(
            "model_bundle_certificate_identity must be one exact GitHub workflow identity"
        )
    if identity_match.group("repository").casefold() != source_match.group("repository").casefold():
        raise ValueError(
            "model bundle signer identity must belong to model_bundle_source_repository"
        )
    ref_parts = identity_match.group("ref").split("/")
    if any(part in {"", ".", ".."} for part in ref_parts):
        raise ValueError("model bundle signer identity contains an invalid Git ref")
    revision = _git_sha(source_revision)
    return {
        "model_bundle_certificate_identity": identity,
        "model_bundle_source_repository": source,
        "model_bundle_source_revision": revision,
    }


def prepare_publish(
    *,
    approved_model_bundle_image: str,
    model_manifest_sha256: str,
    model_bundle_certificate_identity: str,
    model_bundle_source_repository: str,
    model_bundle_source_revision: str,
    github_repository: str,
    github_sha: str,
    github_ref: str,
) -> dict[str, str]:
    """Validate dispatch inputs and return shell/action-safe build values."""
    repository_name = _single_line("github_repository", github_repository)
    repository = _repository_namespace(repository_name)
    revision = _git_sha(github_sha)
    ref = _single_line("github_ref", github_ref)
    if ref != _PRODUCTION_REF:
        raise ValueError(f"production images may only be published from {_PRODUCTION_REF}")

    approved_model_image = _approved_model_reference(approved_model_bundle_image)
    approved_model_digest = approved_model_image.rsplit("@", 1)[1]
    manifest_digest = _digest(
        model_manifest_sha256,
        name="model_manifest_sha256",
    )
    source_repository = f"https://github.com/{repository_name}"
    publisher_identity = f"{source_repository}/{_PUBLISH_WORKFLOW}@{_PRODUCTION_REF}"
    model_source = _model_source_contract(
        certificate_identity=model_bundle_certificate_identity,
        source_repository=model_bundle_source_repository,
        source_revision=model_bundle_source_revision,
    )
    model_repository = f"ghcr.io/{repository}/model-bundle"
    outputs = {
        "approved_model_bundle_image": approved_model_image,
        "approved_model_bundle_digest": approved_model_digest,
        "model_bundle_repository": model_repository,
        "model_bundle_tag": (
            f"{model_repository}:approved-{approved_model_digest.removeprefix('sha256:')}"
        ),
        "model_manifest_sha256": manifest_digest,
        "source_repository": source_repository,
        "source_revision": revision,
        "publisher_identity": publisher_identity,
        **model_source,
    }
    for product in _PRODUCTS:
        image_repository = f"ghcr.io/{repository}/{product}"
        outputs[f"{product}_repository"] = image_repository
        outputs[f"{product}_tag"] = f"{image_repository}:git-{revision}"
    return outputs


def validate_model_mirror(
    *,
    model_bundle_image: str,
    approved_model_bundle_image: str,
    model_manifest_sha256: str,
    model_bundle_certificate_identity: str,
    model_bundle_source_repository: str,
    model_bundle_source_revision: str,
    github_repository: str,
    github_sha: str,
    github_ref: str,
) -> dict[str, str]:
    """Validate the repository-owned mirror produced from an approved bundle."""
    prepare_publish(
        approved_model_bundle_image=approved_model_bundle_image,
        model_manifest_sha256=model_manifest_sha256,
        model_bundle_certificate_identity=model_bundle_certificate_identity,
        model_bundle_source_repository=model_bundle_source_repository,
        model_bundle_source_revision=model_bundle_source_revision,
        github_repository=github_repository,
        github_sha=github_sha,
        github_ref=github_ref,
    )
    repository = _repository_namespace(github_repository)
    mirror = _expected_digest_reference(
        model_bundle_image,
        repository=repository,
        product="model-bundle",
    )
    return {
        "model_bundle_image": mirror,
        "model_bundle_digest": mirror.rsplit("@", 1)[1],
    }


def _write_github_outputs(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for name, value in values.items():
            _single_line(name, value)
            stream.write(f"{name}={value}\n")


def _json_documents(raw: str) -> list[object]:
    """Decode one JSON value, a JSON array, or newline-delimited JSON values."""
    decoder = json.JSONDecoder()
    documents: list[object] = []
    position = 0
    while position < len(raw):
        while position < len(raw) and raw[position].isspace():
            position += 1
        if position == len(raw):
            break
        document, position = decoder.raw_decode(raw, position)
        if isinstance(document, list):
            documents.extend(document)
        else:
            documents.append(document)
    if not documents:
        raise ValueError("Cosign verification output contains no JSON documents")
    return documents


def _load_json_object(path: Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _verified_attestation_statement(
    *,
    verification_output: Path,
    expected_predicate: dict[str, object],
    expected_image: str,
    expected_predicate_type: str,
) -> dict[str, object]:
    """Return the exact matching statement from already verified Cosign output."""
    image = _immutable_image_reference(expected_image, name="expected_image")
    expected_digest = image.rsplit("@sha256:", 1)[1]
    predicate_type = _single_line(
        "expected_predicate_type",
        expected_predicate_type,
    )
    try:
        documents = _json_documents(verification_output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"attestation evidence is not readable JSON: {exc}") from exc

    for document in documents:
        if not isinstance(document, dict):
            continue
        payload = document.get("payload")
        if document.get("payloadType") != "application/vnd.in-toto+json" or not isinstance(
            payload, str
        ):
            continue
        try:
            statement = json.loads(base64.b64decode(payload, validate=True).decode("utf-8"))
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(statement, dict):
            continue
        if statement.get("_type") not in _IN_TOTO_STATEMENT_TYPES:
            continue
        if statement.get("predicateType") != predicate_type:
            continue
        subjects = statement.get("subject")
        if not isinstance(subjects, list):
            continue
        if not any(
            isinstance(subject, dict)
            and isinstance(subject.get("digest"), dict)
            and subject["digest"].get("sha256") == expected_digest
            for subject in subjects
        ):
            continue
        if statement.get("predicate") == expected_predicate:
            return statement
    raise ValueError(
        "no verified attestation binds the expected image digest, predicate URI, "
        "and exact predicate content"
    )


def verify_attestation_output(
    *,
    verification_output: Path,
    expected_predicate: Path,
    expected_image: str,
    expected_predicate_type: str,
) -> None:
    """Require one verified DSSE envelope to contain the exact local predicate."""
    predicate_type = _single_line(
        "expected_predicate_type",
        expected_predicate_type,
    )
    if predicate_type not in _PREDICATE_TYPES:
        raise ValueError("expected_predicate_type is not an approved predicate URI")
    expected = _load_json_object(expected_predicate, name="expected predicate")
    _verified_attestation_statement(
        verification_output=verification_output,
        expected_predicate=expected,
        expected_image=expected_image,
        expected_predicate_type=predicate_type,
    )


def _validate_release_set_predicate(
    record: dict[str, object],
    *,
    expected_api_image: str,
    expected_worker_image: str,
    expected_frontend_image: str,
    expected_model_bundle_image: str,
    expected_source_repository: str,
    expected_source_revision: str,
    expected_workflow_run: str,
) -> str:
    """Validate the closed release-set schema and all four immutable subjects."""
    expected_keys = {
        "schema",
        "schema_version",
        "source_repository",
        "source_revision",
        "publisher_identity",
        "workflow_run",
        "model_bundle",
        "images",
    }
    if set(record) != expected_keys:
        raise ValueError("release-set predicate has unexpected or missing top-level fields")
    if record.get("schema") != _RELEASE_SET_SCHEMA or record.get("schema_version") != 1:
        raise ValueError("release-set predicate schema must be industrial-rag.release-set/v1")

    source_repository = _single_line(
        "expected_source_repository",
        expected_source_repository,
    )
    source_match = _GITHUB_SOURCE_REPOSITORY.fullmatch(source_repository)
    if source_match is None:
        raise ValueError("expected_source_repository must be an exact GitHub repository URL")
    source_revision = _git_sha(expected_source_revision)
    workflow_run = _single_line("expected_workflow_run", expected_workflow_run)
    run_match = _GITHUB_RUN_URL.fullmatch(workflow_run)
    if (
        run_match is None
        or run_match.group("repository").casefold()
        != source_match.group("repository").casefold()
    ):
        raise ValueError("expected_workflow_run must identify a run in source repository")
    publisher_identity = f"{source_repository}/{_PUBLISH_WORKFLOW}@{_PRODUCTION_REF}"
    if record.get("source_repository") != source_repository:
        raise ValueError("release-set predicate source_repository does not match")
    if record.get("source_revision") != source_revision:
        raise ValueError("release-set predicate source_revision does not match")
    if record.get("publisher_identity") != publisher_identity:
        raise ValueError("release-set predicate publisher_identity does not match")
    if record.get("workflow_run") != workflow_run:
        raise ValueError("release-set predicate workflow_run does not match")

    expected_images = {
        "model-bundle": _immutable_image_reference(
            expected_model_bundle_image,
            name="expected_model_bundle_image",
        ),
        "api": _immutable_image_reference(expected_api_image, name="expected_api_image"),
        "worker": _immutable_image_reference(
            expected_worker_image,
            name="expected_worker_image",
        ),
        "frontend": _immutable_image_reference(
            expected_frontend_image,
            name="expected_frontend_image",
        ),
    }
    images = record.get("images")
    if not isinstance(images, dict) or set(images) != set(_RELEASE_SET_PRODUCTS):
        raise ValueError("release-set predicate must contain exactly four named images")
    if images != expected_images:
        raise ValueError("release-set predicate image digests do not match")

    model_bundle = record.get("model_bundle")
    if not isinstance(model_bundle, dict) or set(model_bundle) != {
        "image",
        "digest",
        "manifest_sha256",
        "approved_source",
    }:
        raise ValueError("release-set predicate model_bundle schema is invalid")
    if model_bundle.get("image") != expected_images["model-bundle"]:
        raise ValueError("release-set predicate model_bundle image does not match")
    model_digest = expected_images["model-bundle"].rsplit("@", 1)[1]
    if model_bundle.get("digest") != model_digest:
        raise ValueError("release-set predicate model_bundle digest does not match its image")
    if not isinstance(model_bundle.get("manifest_sha256"), str):
        raise ValueError("release-set predicate model manifest digest is missing")
    _digest(str(model_bundle["manifest_sha256"]), name="model manifest digest")

    approved_source = model_bundle.get("approved_source")
    if not isinstance(approved_source, dict) or set(approved_source) != {
        "image",
        "digest",
        "certificate_identity",
        "source_repository",
        "source_revision",
    }:
        raise ValueError("release-set predicate approved_source schema is invalid")
    approved_image = _immutable_image_reference(
        str(approved_source.get("image", "")),
        name="approved source image",
    )
    if approved_source.get("digest") != approved_image.rsplit("@", 1)[1]:
        raise ValueError("release-set predicate approved source digest does not match its image")
    _model_source_contract(
        certificate_identity=str(approved_source.get("certificate_identity", "")),
        source_repository=str(approved_source.get("source_repository", "")),
        source_revision=str(approved_source.get("source_revision", "")),
    )
    return _release_set_predicate_type(source_repository)


def verify_release_set_attestation_output(
    *,
    verification_output: Path,
    expected_predicate: Path,
    expected_api_image: str,
    expected_worker_image: str,
    expected_frontend_image: str,
    expected_model_bundle_image: str,
    expected_source_repository: str,
    expected_source_revision: str,
    expected_workflow_run: str,
) -> None:
    """Verify the final API-bound, four-image release-set attestation."""
    expected = _load_json_object(expected_predicate, name="release-set predicate")
    predicate_type = _validate_release_set_predicate(
        expected,
        expected_api_image=expected_api_image,
        expected_worker_image=expected_worker_image,
        expected_frontend_image=expected_frontend_image,
        expected_model_bundle_image=expected_model_bundle_image,
        expected_source_repository=expected_source_repository,
        expected_source_revision=expected_source_revision,
        expected_workflow_run=expected_workflow_run,
    )
    _verified_attestation_statement(
        verification_output=verification_output,
        expected_predicate=expected,
        expected_image=expected_api_image,
        expected_predicate_type=predicate_type,
    )


def finalize_publish(
    *,
    api_image: str,
    worker_image: str,
    frontend_image: str,
    model_bundle_image: str,
    approved_model_bundle_image: str,
    model_manifest_sha256: str,
    model_bundle_certificate_identity: str,
    model_bundle_source_repository: str,
    model_bundle_source_revision: str,
    github_repository: str,
    github_sha: str,
    github_ref: str,
    github_run_url: str,
    output_dir: Path,
) -> dict[str, str]:
    """Create immutable release inputs and per-image SLSA v1 predicates."""
    prepared = prepare_publish(
        approved_model_bundle_image=approved_model_bundle_image,
        model_manifest_sha256=model_manifest_sha256,
        model_bundle_certificate_identity=model_bundle_certificate_identity,
        model_bundle_source_repository=model_bundle_source_repository,
        model_bundle_source_revision=model_bundle_source_revision,
        github_repository=github_repository,
        github_sha=github_sha,
        github_ref=github_ref,
    )
    repository_name = _single_line("github_repository", github_repository)
    repository = _repository_namespace(repository_name)
    mirror = validate_model_mirror(
        model_bundle_image=model_bundle_image,
        approved_model_bundle_image=approved_model_bundle_image,
        model_manifest_sha256=model_manifest_sha256,
        model_bundle_certificate_identity=model_bundle_certificate_identity,
        model_bundle_source_repository=model_bundle_source_repository,
        model_bundle_source_revision=model_bundle_source_revision,
        github_repository=github_repository,
        github_sha=github_sha,
        github_ref=github_ref,
    )
    images = {
        "api": _expected_digest_reference(api_image, repository=repository, product="api"),
        "worker": _expected_digest_reference(
            worker_image,
            repository=repository,
            product="worker",
        ),
        "frontend": _expected_digest_reference(
            frontend_image,
            repository=repository,
            product="frontend",
        ),
    }
    run_url = _single_line("github_run_url", github_run_url)
    run_match = _GITHUB_RUN_URL.fullmatch(run_url)
    if run_match is None or run_match.group("repository").casefold() != repository_name.casefold():
        raise ValueError("github_run_url must identify a run in github_repository")

    output_dir.mkdir(parents=True, exist_ok=True)
    release_env = {
        "API_IMAGE": images["api"],
        "WORKER_IMAGE": images["worker"],
        "FRONTEND_IMAGE": images["frontend"],
        "MODEL_BUNDLE_IMAGE": mirror["model_bundle_image"],
        "MODEL_BUNDLE_DIGEST": mirror["model_bundle_digest"],
        "MODEL_MANIFEST_SHA256": prepared["model_manifest_sha256"],
    }
    (output_dir / "release-images.env").write_text(
        "".join(f"{name}={value}\n" for name, value in release_env.items()),
        encoding="utf-8",
        newline="\n",
    )

    record: dict[str, object] = {
        "schema": _RELEASE_SET_SCHEMA,
        "schema_version": 1,
        "source_repository": prepared["source_repository"],
        "source_revision": prepared["source_revision"],
        "publisher_identity": prepared["publisher_identity"],
        "workflow_run": run_url,
        "model_bundle": {
            "image": mirror["model_bundle_image"],
            "digest": mirror["model_bundle_digest"],
            "manifest_sha256": prepared["model_manifest_sha256"],
            "approved_source": {
                "image": prepared["approved_model_bundle_image"],
                "digest": prepared["approved_model_bundle_digest"],
                "certificate_identity": prepared["model_bundle_certificate_identity"],
                "source_repository": prepared["model_bundle_source_repository"],
                "source_revision": prepared["model_bundle_source_revision"],
            },
        },
        "images": {
            "model-bundle": mirror["model_bundle_image"],
            **images,
        },
    }
    (output_dir / "release-images.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    attested_images = {
        "model-bundle": mirror["model_bundle_image"],
        **images,
    }
    for product in attested_images:
        resolved_dependencies = [
            {
                "uri": "git+" + prepared["source_repository"],
                "digest": {"gitCommit": prepared["source_revision"]},
            },
            {
                "uri": prepared["approved_model_bundle_image"],
                "digest": {
                    "sha256": prepared["approved_model_bundle_digest"].removeprefix("sha256:")
                },
            },
        ]
        if product != "model-bundle":
            resolved_dependencies.insert(
                1,
                {
                    "uri": mirror["model_bundle_image"],
                    "digest": {"sha256": mirror["model_bundle_digest"].removeprefix("sha256:")},
                },
            )
        provenance = {
            "buildDefinition": {
                "buildType": (f"{prepared['source_repository']}/{_PUBLISH_WORKFLOW}@v1"),
                "externalParameters": {
                    "product": product,
                    "sourceRepository": prepared["source_repository"],
                    "sourceRevision": prepared["source_revision"],
                    "modelManifestSha256": prepared["model_manifest_sha256"],
                    "modelSourceRepository": prepared["model_bundle_source_repository"],
                    "modelSourceRevision": prepared["model_bundle_source_revision"],
                    "modelSignerIdentity": prepared["model_bundle_certificate_identity"],
                },
                "internalParameters": {
                    "workflow": _PUBLISH_WORKFLOW,
                    "workflowRef": _PRODUCTION_REF,
                },
                "resolvedDependencies": resolved_dependencies,
            },
            "runDetails": {
                "builder": {"id": prepared["publisher_identity"]},
                "metadata": {"invocationId": run_url},
            },
        }
        (output_dir / f"{product}-slsa-provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    return {
        **images,
        "model_bundle": mirror["model_bundle_image"],
        "model_bundle_digest": mirror["model_bundle_digest"],
        "model_manifest_sha256": prepared["model_manifest_sha256"],
        "publisher_identity": prepared["publisher_identity"],
        "source_repository": prepared["source_repository"],
        "source_revision": prepared["source_revision"],
        "workflow_run": run_url,
        "release_set_predicate_type": _release_set_predicate_type(
            prepared["source_repository"]
        ),
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--approved-model-bundle-image", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--model-bundle-certificate-identity", required=True)
    parser.add_argument("--model-bundle-source-repository", required=True)
    parser.add_argument("--model-bundle-source-revision", required=True)
    parser.add_argument("--github-repository", required=True)
    parser.add_argument("--github-sha", required=True)
    parser.add_argument("--github-ref", required=True)
    parser.add_argument("--github-output", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    _add_common_arguments(prepare_parser)

    mirror_parser = commands.add_parser("mirror")
    _add_common_arguments(mirror_parser)
    mirror_parser.add_argument("--model-bundle-image", required=True)

    finalize_parser = commands.add_parser("finalize")
    _add_common_arguments(finalize_parser)
    finalize_parser.add_argument("--model-bundle-image", required=True)
    finalize_parser.add_argument("--api-image", required=True)
    finalize_parser.add_argument("--worker-image", required=True)
    finalize_parser.add_argument("--frontend-image", required=True)
    finalize_parser.add_argument("--github-run-url", required=True)
    finalize_parser.add_argument("--output-dir", type=Path, required=True)

    attestation_parser = commands.add_parser("verify-attestation-output")
    attestation_parser.add_argument("--verification-output", type=Path, required=True)
    attestation_parser.add_argument("--expected-predicate", type=Path, required=True)
    attestation_parser.add_argument("--expected-image", required=True)
    attestation_parser.add_argument(
        "--expected-predicate-type",
        choices=sorted(_PREDICATE_TYPES),
        required=True,
    )
    release_set_parser = commands.add_parser("verify-release-set-attestation-output")
    release_set_parser.add_argument("--verification-output", type=Path, required=True)
    release_set_parser.add_argument("--expected-predicate", type=Path, required=True)
    release_set_parser.add_argument("--expected-api-image", required=True)
    release_set_parser.add_argument("--expected-worker-image", required=True)
    release_set_parser.add_argument("--expected-frontend-image", required=True)
    release_set_parser.add_argument("--expected-model-bundle-image", required=True)
    release_set_parser.add_argument("--expected-source-repository", required=True)
    release_set_parser.add_argument("--expected-source-revision", required=True)
    release_set_parser.add_argument("--expected-workflow-run", required=True)
    args = parser.parse_args()

    if args.command == "verify-attestation-output":
        verify_attestation_output(
            verification_output=args.verification_output,
            expected_predicate=args.expected_predicate,
            expected_image=args.expected_image,
            expected_predicate_type=args.expected_predicate_type,
        )
        print("Production image attestation content verified.")
        return
    if args.command == "verify-release-set-attestation-output":
        verify_release_set_attestation_output(
            verification_output=args.verification_output,
            expected_predicate=args.expected_predicate,
            expected_api_image=args.expected_api_image,
            expected_worker_image=args.expected_worker_image,
            expected_frontend_image=args.expected_frontend_image,
            expected_model_bundle_image=args.expected_model_bundle_image,
            expected_source_repository=args.expected_source_repository,
            expected_source_revision=args.expected_source_revision,
            expected_workflow_run=args.expected_workflow_run,
        )
        print("Atomic production release-set attestation verified.")
        return

    common = {
        "approved_model_bundle_image": args.approved_model_bundle_image,
        "model_manifest_sha256": args.model_manifest_sha256,
        "model_bundle_certificate_identity": args.model_bundle_certificate_identity,
        "model_bundle_source_repository": args.model_bundle_source_repository,
        "model_bundle_source_revision": args.model_bundle_source_revision,
        "github_repository": args.github_repository,
        "github_sha": args.github_sha,
        "github_ref": args.github_ref,
    }
    if args.command == "prepare":
        outputs = prepare_publish(**common)
    elif args.command == "mirror":
        outputs = validate_model_mirror(
            **common,
            model_bundle_image=args.model_bundle_image,
        )
    else:
        outputs = finalize_publish(
            **common,
            model_bundle_image=args.model_bundle_image,
            api_image=args.api_image,
            worker_image=args.worker_image,
            frontend_image=args.frontend_image,
            github_run_url=args.github_run_url,
            output_dir=args.output_dir,
        )
    _write_github_outputs(args.github_output, outputs)
    print(f"Production image publish contract {args.command} completed.")


if __name__ == "__main__":
    main()
