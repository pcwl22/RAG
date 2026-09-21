"""Build and verify a canonical, API-bound production image-set predicate.

Cryptographic verification remains Cosign's responsibility.  This module owns
the semantic boundary on either side of ``cosign attest``: it creates one
deterministic predicate for the four production images and, after
``cosign verify-attestation``, proves that the verified DSSE statement contains
that exact predicate and uses the API image as its sole subject.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
COMPONENTS = ("model-bundle", "api", "worker", "frontend")
MATERIAL_PATHS = {
    "evaluation": (
        "eval/legal_expanded_240.jsonl",
        "eval/legal_expanded_ragas_40.jsonl",
        "eval/legal_holdout_150.jsonl",
    ),
    "contracts": (
        "app/llm/answer_contract.py",
        "config/base.yaml",
        "prompts/business.txt",
        "prompts/citation.txt",
        "prompts/output.txt",
        "prompts/rag_template.jinja2",
        "prompts/system.txt",
    ),
    "dependencyLocks": (
        "requirements-ci.lock.txt",
        "requirements-runtime.lock.txt",
        "requirements-gpu.lock.txt",
        "requirements-torch-cpu.lock.txt",
        "requirements-torch-cu126.lock.txt",
        "requirements-evaluation.lock.txt",
    ),
}

_REPOSITORY = re.compile(
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?)/"
    r"(?P<name>[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?)"
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_RUN_URL = re.compile(
    r"https://github\.com/(?P<repository>[^/]+/[^/]+)/actions/runs/"
    r"(?P<run_id>[1-9][0-9]*)(?:/attempts/(?P<attempt>[1-9][0-9]*))?"
)
_WORKFLOW_IDENTITY = re.compile(
    r"https://github\.com/(?P<repository>[^/]+/[^/]+)/"
    r"(?P<workflow>\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml)"
    r"@refs/(?P<ref_kind>heads|tags)/(?P<ref>[A-Za-z0-9._/-]+)"
)
_STATEMENT_TYPES = {
    "https://in-toto.io/Statement/v0.1",
    "https://in-toto.io/Statement/v1",
}


def _single_line(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized or "\n" in normalized or "\r" in normalized:
        raise ValueError(f"{name} must be a non-empty single-line value")
    return normalized


def _repository(value: str) -> str:
    normalized = _single_line("github_repository", value)
    if _REPOSITORY.fullmatch(normalized) is None:
        raise ValueError("github_repository must be an owner/repository name")
    return normalized.lower()


def _git_sha(value: str, *, name: str = "source_revision") -> str:
    normalized = _single_line(name, value)
    if _GIT_SHA.fullmatch(normalized) is None or normalized == "0" * 40:
        raise ValueError(f"{name} must be a non-zero, full lowercase 40-character Git SHA")
    return normalized


def _sha256(value: str, *, name: str) -> str:
    normalized = _single_line(name, value)
    if _SHA256.fullmatch(normalized) is None or normalized == "sha256:" + "0" * 64:
        raise ValueError(f"{name} must be a non-zero sha256:<64 lowercase hex> digest")
    return normalized


def predicate_type(github_repository: str) -> str:
    """Return the repository-owned atomic image-set predicate/schema URI.

    ``production_image_publish.py`` already owns ``release-set/v1`` with a
    different, publication-focused schema.  This richer release-gate contract
    therefore uses a distinct URI so verifiers can never interpret one schema
    as the other.
    """
    return f"https://github.com/{_repository(github_repository)}/attestations/image-set/v1"


def _workflow_run(value: str, *, repository: str) -> str:
    normalized = _single_line("workflow_run", value)
    match = _RUN_URL.fullmatch(normalized)
    if match is None or match.group("repository").casefold() != repository.casefold():
        raise ValueError("workflow_run must be an exact GitHub Actions run URL for the repository")
    result = f"https://github.com/{repository}/actions/runs/{match.group('run_id')}"
    if match.group("attempt"):
        result += f"/attempts/{match.group('attempt')}"
    return result


def _publisher_identity(value: str, *, repository: str) -> str:
    normalized = _single_line("publisher_identity", value)
    match = _WORKFLOW_IDENTITY.fullmatch(normalized)
    if match is None or match.group("repository").casefold() != repository.casefold():
        raise ValueError(
            "publisher_identity must be one exact GitHub workflow identity in the repository"
        )
    ref_parts = match.group("ref").split("/")
    if any(part in {"", ".", ".."} for part in ref_parts):
        raise ValueError("publisher_identity contains an invalid Git ref")
    workflow = match.group("workflow")
    ref_kind = match.group("ref_kind")
    ref = match.group("ref")
    publish_identity = (
        workflow == ".github/workflows/publish-production-images.yml"
        and ref_kind == "heads"
        and ref == "main"
    )
    if not publish_identity:
        raise ValueError("publisher_identity must be the protected production image publisher")
    # Preserve the certificate identity byte-for-byte. Cosign identity matching
    # is exact even though GitHub repository names themselves are case-insensitive.
    return normalized


def _image(value: str, *, repository: str, component: str) -> str:
    normalized = _single_line(f"{component}_image", value)
    prefix = f"ghcr.io/{repository}/{component}@"
    if not normalized.startswith(prefix):
        raise ValueError(
            f"{component}_image must be the immutable {prefix}sha256:<digest> reference"
        )
    digest = _sha256(normalized.removeprefix(prefix), name=f"{component}_image digest")
    return prefix + digest


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + suffix
    ).encode("utf-8")


def _material_group(project_root: Path, paths: Iterable[str]) -> dict[str, object]:
    root = project_root.resolve(strict=True)
    entries: list[dict[str, object]] = []
    for relative in sorted(paths):
        path = project_root / Path(relative)
        if path.is_symlink():
            raise ValueError(f"image-set material must not be a symlink: {relative}")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"image-set material is missing or escapes project root: {relative}") from exc
        if not resolved.is_file():
            raise ValueError(f"image-set material is not a file: {relative}")
        entries.append(
            {
                "path": Path(relative).as_posix(),
                "sha256": _digest_file(resolved),
                "size": resolved.stat().st_size,
            }
        )
    return {
        "sha256": "sha256:" + hashlib.sha256(_canonical_bytes(entries, newline=False)).hexdigest(),
        "files": entries,
    }


def build_predicate(
    *,
    project_root: Path,
    github_repository: str,
    source_revision: str,
    workflow_run: str,
    publisher_identity: str,
    model_bundle_image: str,
    api_image: str,
    worker_image: str,
    frontend_image: str,
    model_manifest_sha256: str,
) -> dict[str, object]:
    """Build the closed image-set schema from current repository materials."""
    repository = _repository(github_repository)
    revision = _git_sha(source_revision)
    run = _workflow_run(workflow_run, repository=repository)
    identity = _publisher_identity(publisher_identity, repository=repository)
    image_inputs = {
        "model-bundle": model_bundle_image,
        "api": api_image,
        "worker": worker_image,
        "frontend": frontend_image,
    }
    images = {
        component: _image(value, repository=repository, component=component)
        for component, value in image_inputs.items()
    }
    components = [
        {
            "name": component,
            "image": images[component],
            "digest": images[component].rsplit("@", 1)[1],
        }
        for component in COMPONENTS
    ]
    model_digest = images["model-bundle"].rsplit("@", 1)[1]
    materials = {
        group: _material_group(project_root, paths)
        for group, paths in MATERIAL_PATHS.items()
    }
    return {
        "schema": predicate_type(repository),
        "schemaVersion": SCHEMA_VERSION,
        "source": {
            "repository": f"https://github.com/{repository}",
            "revision": revision,
            "workflowRun": run,
            "publisherIdentity": identity,
        },
        "components": components,
        "model": {
            "bundleImage": images["model-bundle"],
            "bundleDigest": model_digest,
            "manifestSha256": _sha256(
                model_manifest_sha256,
                name="model_manifest_sha256",
            ),
        },
        "materials": materials,
    }


def write_predicate(output: Path, predicate: dict[str, object]) -> None:
    """Atomically publish canonical predicate bytes in the destination directory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(_canonical_bytes(predicate))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate object key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"JSON contains a non-standard numeric constant: {value}")


def _strict_json(raw: str, *, name: str) -> object:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc


def _read_predicate(path: Path, *, require_canonical: bool) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"image-set predicate is not readable UTF-8: {exc}") from exc
    document = _strict_json(text, name="image-set predicate")
    if not isinstance(document, dict):
        raise ValueError("image-set predicate must be one JSON object")
    if require_canonical and raw != _canonical_bytes(document):
        raise ValueError("image-set predicate is not in canonical JSON form")
    return document


def validate_predicate(
    *,
    predicate_path: Path,
    project_root: Path,
    github_repository: str,
    source_revision: str,
    workflow_run: str,
    publisher_identity: str,
    model_bundle_image: str,
    api_image: str,
    worker_image: str,
    frontend_image: str,
    model_manifest_sha256: str,
) -> dict[str, object]:
    """Require canonical bytes and exact equality with a locally rebuilt predicate."""
    actual = _read_predicate(predicate_path, require_canonical=True)
    expected = build_predicate(
        project_root=project_root,
        github_repository=github_repository,
        source_revision=source_revision,
        workflow_run=workflow_run,
        publisher_identity=publisher_identity,
        model_bundle_image=model_bundle_image,
        api_image=api_image,
        worker_image=worker_image,
        frontend_image=frontend_image,
        model_manifest_sha256=model_manifest_sha256,
    )
    if _canonical_bytes(actual) != _canonical_bytes(expected):
        raise ValueError(
            "image-set predicate does not match the expected source, images, model, or materials"
        )
    return actual


def _json_documents(raw: str) -> list[object]:
    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonstandard_constant,
    )
    documents: list[object] = []
    position = 0
    try:
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
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cosign verification output is not valid JSON: {exc}") from exc
    if not documents:
        raise ValueError("Cosign verification output contains no JSON documents")
    return documents


def _statement_from_envelope(document: object) -> dict[str, object] | None:
    if not isinstance(document, dict):
        return None
    payload = document.get("payload")
    if document.get("payloadType") != "application/vnd.in-toto+json" or not isinstance(
        payload, str
    ):
        return None
    try:
        decoded = base64.b64decode(payload, validate=True).decode("utf-8")
        statement = _strict_json(decoded, name="verified in-toto statement")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    return statement if isinstance(statement, dict) else None


def verify_attestation(
    *,
    verification_output: Path,
    project_root: Path,
    github_repository: str,
    source_revision: str,
    workflow_run: str,
    publisher_identity: str,
    model_bundle_image: str,
    api_image: str,
    worker_image: str,
    frontend_image: str,
    model_manifest_sha256: str,
) -> dict[str, object]:
    """Validate one Cosign-verified image-set statement and its API subject."""
    expected = build_predicate(
        project_root=project_root,
        github_repository=github_repository,
        source_revision=source_revision,
        workflow_run=workflow_run,
        publisher_identity=publisher_identity,
        model_bundle_image=model_bundle_image,
        api_image=api_image,
        worker_image=worker_image,
        frontend_image=frontend_image,
        model_manifest_sha256=model_manifest_sha256,
    )
    repository = _repository(github_repository)
    expected_api = _image(api_image, repository=repository, component="api")
    api_name, api_digest = expected_api.rsplit("@sha256:", 1)
    try:
        documents = _json_documents(verification_output.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cosign verification output is not readable: {exc}") from exc

    matches: list[dict[str, object]] = []
    for document in documents:
        statement = _statement_from_envelope(document)
        if statement is None or statement.get("_type") not in _STATEMENT_TYPES:
            continue
        if set(statement) != {"_type", "subject", "predicateType", "predicate"}:
            continue
        if statement.get("predicateType") != expected["schema"]:
            continue
        subjects = statement.get("subject")
        expected_subject = {"name": api_name, "digest": {"sha256": api_digest}}
        if subjects != [expected_subject]:
            continue
        predicate = statement.get("predicate")
        if isinstance(predicate, dict) and _canonical_bytes(predicate) == _canonical_bytes(expected):
            matches.append(statement)
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one verified image-set attestation binding the API subject "
            "and exact canonical predicate"
        )
    return matches[0]


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--github-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--workflow-run", required=True)
    parser.add_argument("--publisher-identity", required=True)
    parser.add_argument("--model-bundle-image", required=True)
    parser.add_argument("--api-image", required=True)
    parser.add_argument("--worker-image", required=True)
    parser.add_argument("--frontend-image", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    type_parser = commands.add_parser("predicate-type")
    type_parser.add_argument("--github-repository", required=True)

    create_parser = commands.add_parser("create")
    _add_contract_arguments(create_parser)
    create_parser.add_argument("--output", type=Path, required=True)

    validate_parser = commands.add_parser("validate")
    _add_contract_arguments(validate_parser)
    validate_parser.add_argument("--predicate", type=Path, required=True)

    verify_parser = commands.add_parser("verify-attestation")
    _add_contract_arguments(verify_parser)
    verify_parser.add_argument("--verification-output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "predicate-type":
        print(predicate_type(args.github_repository))
        return

    values = {
        "project_root": args.project_root,
        "github_repository": args.github_repository,
        "source_revision": args.source_revision,
        "workflow_run": args.workflow_run,
        "publisher_identity": args.publisher_identity,
        "model_bundle_image": args.model_bundle_image,
        "api_image": args.api_image,
        "worker_image": args.worker_image,
        "frontend_image": args.frontend_image,
        "model_manifest_sha256": args.model_manifest_sha256,
    }
    if args.command == "create":
        write_predicate(args.output, build_predicate(**values))
        print(f"Wrote canonical image-set predicate: {args.output}")
    elif args.command == "validate":
        validate_predicate(predicate_path=args.predicate, **values)
        print("Image-set predicate validated.")
    else:
        verify_attestation(verification_output=args.verification_output, **values)
        print("Verified API-bound image-set attestation predicate.")


if __name__ == "__main__":
    main()
