import base64
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from app.evaluation.retrieval_contract import (
    build_retrieval_runtime_contract,
    normalize_llm_runtime_identity,
)
from app.utils.strict_dotenv import (
    load_release_env_file,
    release_config_digest,
    release_resource_suffix,
)
from scripts.approve_release_baseline import (
    _protected_dataset_contract,
    _require_matching_retrieval_runtime_contract,
    _require_matching_retrieval_runtime_identity,
    approve,
)
from scripts.build_model_bundle_manifest import build_manifest, write_manifest
from scripts.check_holdout_quality_gate import build_holdout_gate
from scripts.export_release_values import release_outputs
from scripts.list_workload_resource_refs import workload_resource_refs
from scripts.render_kubernetes_secrets import render
from scripts.render_production_manifests import render as render_manifests
from scripts.validate_kubernetes_manifests import (
    _container_gpu_contract,
    _validate_pod_security_contract,
    _validate_probe_contract,
    _validate_service_account_contract,
    validate_expected_release_images,
)
from scripts.validate_kubernetes_manifests import (
    validate_manifest as validate_kubernetes_manifest,
)
from scripts.validate_model_bundle_manifest import tree_sha256, validate_manifest
from scripts.validate_release_baseline import (
    CURRENT_SCHEMA_VERSION,
    HASH_CONTRACT,
    REQUIRED_IMPLEMENTATION_PATHS,
    canonical_file_sha256,
    implementation_sha256,
)
from scripts.validate_release_baseline import (
    validate as validate_release_baseline,
)
from scripts.validate_runtime_credential_rotation import validate_rotation

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _retrieval_contract(*, score_threshold=None):
    return build_retrieval_runtime_contract(
        {
            "_meta": {"config_path": "config/base.yaml"},
            "reranker": {"enabled": True, "score_threshold": score_threshold},
            "rag": {"retrieval": {}},
        }
    )


def _load_canary_module():
    path = REPOSITORY_ROOT / "industrial-rag/deploy/kustomize/base/run_release_canary.py"
    spec = importlib.util.spec_from_file_location("release_canary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_holdout_gate_requires_exact_count_and_contract():
    report = build_holdout_gate(
        {
            "sample_count": 150,
            "evaluation_contract": {
                "input_sha256": "b" * 64,
                "citation_leakage_checked": True,
                "use_production_pipeline": True,
                "production_decomposition": True,
                "fail_closed_query_understanding": True,
                "checkpoint_fingerprint": "f" * 64,
                "minimum_citation_recall": 0.95,
                "llm_runtime_identity": {
                    "provider": "openai_compatible",
                    "model_name": "model-a",
                    "endpoint_sha256": "a" * 64,
                },
                "retrieval_runtime_contract": _retrieval_contract(),
            },
            "in_domain": {"citation_recall": 0.98, "citation_mrr": 0.75},
        }
    )
    assert report["passed"] is True

    report = build_holdout_gate(
        {
            "sample_count": 149,
            "evaluation_contract": {
                "input_sha256": "b" * 64,
                "citation_leakage_checked": True,
                "use_production_pipeline": True,
                "production_decomposition": True,
                "fail_closed_query_understanding": True,
                "checkpoint_fingerprint": "f" * 64,
                "minimum_citation_recall": 0.95,
                "llm_runtime_identity": {
                    "provider": "openai_compatible",
                    "model_name": "model-a",
                    "endpoint_sha256": "a" * 64,
                },
                "retrieval_runtime_contract": _retrieval_contract(),
            },
            "in_domain": {"citation_recall": 1.0, "citation_mrr": 1.0},
        }
    )
    assert report["passed"] is False


def test_retrieval_runtime_identity_is_bounded_and_rejects_extra_fields():
    valid = {
        "provider": "OPENAI_COMPATIBLE",
        "model_name": "model-a",
        "endpoint_sha256": "A" * 64,
    }

    assert normalize_llm_runtime_identity(valid) == {
        "provider": "openai_compatible",
        "model_name": "model-a",
        "endpoint_sha256": "a" * 64,
    }
    assert normalize_llm_runtime_identity({**valid, "api_key": "secret"}) is None


def test_release_approval_rejects_retrieval_model_drift_between_suites():
    expanded = {
        "retrieval_llm_runtime_identity": {
            "provider": "openai_compatible",
            "model_name": "model-a",
            "endpoint_sha256": "a" * 64,
        }
    }
    holdout = copy.deepcopy(expanded)
    holdout["retrieval_llm_runtime_identity"]["model_name"] = "model-b"

    with pytest.raises(ValueError, match="different LLM runtime identities"):
        _require_matching_retrieval_runtime_identity(expanded, holdout)

    production = copy.deepcopy(expanded["retrieval_llm_runtime_identity"])
    production["endpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="production environment does not match"):
        _require_matching_retrieval_runtime_identity(
            expanded,
            expanded,
            production,
        )

    ragas = copy.deepcopy(expanded)
    ragas["retrieval_llm_runtime_identity"]["model_name"] = "model-c"
    with pytest.raises(ValueError, match="quality gates used different"):
        _require_matching_retrieval_runtime_identity(
            expanded,
            expanded,
            ragas_gate=ragas,
        )


def test_release_approval_rejects_retrieval_configuration_drift_between_suites():
    matching = {"retrieval_runtime_contract_sha256": "c" * 64}
    drifted = {"retrieval_runtime_contract_sha256": "d" * 64}

    assert (
        _require_matching_retrieval_runtime_contract(matching, matching, matching)
        == "c" * 64
    )
    with pytest.raises(ValueError, match="different retrieval runtime contracts"):
        _require_matching_retrieval_runtime_contract(matching, drifted, matching)


def test_release_approval_binds_reports_to_current_protected_datasets(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    hashes = {}
    for name, count in {
        "legal_expanded_240.jsonl": 240,
        "legal_expanded_ragas_40.jsonl": 40,
        "legal_holdout_150.jsonl": 150,
    }.items():
        path = eval_dir / name
        path.write_text(
            "".join(json.dumps({"id": index}) + "\n" for index in range(count)),
            encoding="utf-8",
        )
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    contract = _protected_dataset_contract(
        tmp_path,
        ragas_gate={"source_dataset_sha256": hashes["legal_expanded_ragas_40.jsonl"]},
        expanded_gate={"retrieval_input_sha256": hashes["legal_expanded_240.jsonl"]},
        holdout_gate={"retrieval_input_sha256": hashes["legal_holdout_150.jsonl"]},
    )

    assert contract["legal_expanded_240.jsonl"]["sample_count"] == 240
    assert contract["legal_expanded_ragas_40.jsonl"]["sha256"] == canonical_file_sha256(
        eval_dir / "legal_expanded_ragas_40.jsonl"
    )
    with pytest.raises(ValueError, match="not bound to the current dataset"):
        _protected_dataset_contract(
            tmp_path,
            ragas_gate={"source_dataset_sha256": hashes["legal_expanded_ragas_40.jsonl"]},
            expanded_gate={"retrieval_input_sha256": "f" * 64},
            holdout_gate={"retrieval_input_sha256": hashes["legal_holdout_150.jsonl"]},
        )


def test_release_approval_produces_a_self_validating_refreshed_baseline(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    hashes = {}
    for name, count in {
        "legal_expanded_240.jsonl": 240,
        "legal_expanded_ragas_40.jsonl": 40,
        "legal_holdout_150.jsonl": 150,
    }.items():
        path = eval_dir / name
        path.write_text(
            "".join(
                json.dumps(
                    {"id": f"{name}-{index}", "query": "question", "expected_citations": []}
                )
                + "\n"
                for index in range(count)
            ),
            encoding="utf-8",
        )
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    implementation_paths = sorted(REQUIRED_IMPLEMENTATION_PATHS)
    for relative in implementation_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")

    baseline_path = eval_dir / "release_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "status": "blocked",
                "datasets": {},
                "implementation": {"paths": implementation_paths, "sha256": "stale"},
                "evaluation_contract": {
                    "holdout_dataset": "legal_holdout_150.jsonl",
                    "citation_leakage_policy": "reject",
                },
                "checks": {},
            }
        ),
        encoding="utf-8",
    )
    identity = {
        "provider": "openai_compatible",
        "model_name": "model-a",
        "endpoint_sha256": hashlib.sha256(b"https://llm.example/v1").hexdigest(),
    }
    retrieval_contract_sha256 = "c" * 64

    def write_report(name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    ragas_checks = {
        name: {"value": 0.9, "minimum": 0.7, "status": "passed"}
        for name in (
            "answer_accuracy",
            "faithfulness",
            "context_precision",
            "context_recall",
            "citation_recall",
            "citation_mrr",
        )
    }
    ragas_gate = write_report(
        "ragas-gate.json",
        {
            "passed": True,
            "judge_sample_count": 40,
            "top_k": 5,
            "evaluation_engine": "industrial-rag-native-text-judge",
            "evaluation_engine_version": "1.1",
            "judge_model": "judge-a",
            "source_dataset_sha256": hashes["legal_expanded_ragas_40.jsonl"],
            "retrieval_llm_runtime_identity": identity,
            "retrieval_runtime_contract_sha256": retrieval_contract_sha256,
            "answer_generation_policy_sha256": "d" * 64,
            "judge_request_policy_sha256": "e" * 64,
            "checks": ragas_checks,
        },
    )
    expanded_gate = write_report(
        "expanded-gate.json",
        {
            "passed": True,
            "full_suite_sample_count": 240,
            "representative_judge_sample_count": 40,
            "retrieval_input_sha256": hashes["legal_expanded_240.jsonl"],
            "retrieval_llm_runtime_identity": identity,
            "retrieval_runtime_contract_sha256": retrieval_contract_sha256,
            "checks": {
                name: {"value": 0.96, "minimum": minimum, "status": "passed"}
                for name, minimum in {
                    "in_domain_citation_recall": 0.95,
                    "adversarial_citation_recall": 0.85,
                    "comparison_citation_recall": 0.85,
                    "no_answer_abstention_rate": 0.90,
                }.items()
            },
        },
    )
    holdout_gate = write_report(
        "holdout-gate.json",
        {
            "passed": True,
            "sample_count": 150,
            "retrieval_input_sha256": hashes["legal_holdout_150.jsonl"],
            "retrieval_llm_runtime_identity": identity,
            "retrieval_runtime_contract_sha256": retrieval_contract_sha256,
            "checks": {
                "citation_recall": {"value": 0.96, "minimum": 0.95, "status": "passed"},
                "citation_mrr": {"value": 0.8, "minimum": 0.7, "status": "passed"},
            },
        },
    )
    production_env = tmp_path / "production.env"
    production_env.write_text(
        "DEEPSEEK_MODEL=model-a\nDEEPSEEK_API_URL=https://llm.example/v1\n",
        encoding="utf-8",
    )

    candidate = approve(
        baseline_path,
        tmp_path,
        ragas_gate_path=ragas_gate,
        expanded_gate_path=expanded_gate,
        holdout_gate_path=holdout_gate,
        production_env_path=production_env,
    )
    candidate_path = eval_dir / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    assert candidate["status"] == "approved"
    assert candidate["hash_contract"] == HASH_CONTRACT
    assert candidate["datasets"]["legal_expanded_240.jsonl"][
        "sha256"
    ] == canonical_file_sha256(eval_dir / "legal_expanded_240.jsonl")
    assert validate_release_baseline(candidate_path, tmp_path)["passed"] is True


def test_release_hash_contract_normalizes_text_line_endings(tmp_path):
    lf_path = tmp_path / "lf.txt"
    crlf_path = tmp_path / "crlf.txt"
    lf_path.write_bytes(b"first\nsecond\n")
    crlf_path.write_bytes(b"first\r\nsecond\r\n")

    assert canonical_file_sha256(lf_path) == canonical_file_sha256(crlf_path)

    lf_root = tmp_path / "lf-root"
    crlf_root = tmp_path / "crlf-root"
    (lf_root / "app").mkdir(parents=True)
    (crlf_root / "app").mkdir(parents=True)
    (lf_root / "app/config.txt").write_bytes(b"key=value\n")
    (crlf_root / "app/config.txt").write_bytes(b"key=value\r\n")
    assert implementation_sha256(lf_root, ["app"]) == implementation_sha256(
        crlf_root, ["app"]
    )


def test_kubernetes_secret_renderer_separates_runtime_and_migration_credentials(tmp_path):
    canary_sample = ("受保护生产发布功能验证样本。" * 12).encode("utf-8")
    env_file = tmp_path / "production.env"
    env_file.write_text(
        "\n".join(
            [
                "RELEASE_ID=123e4567-e89b-42d3-a456-426614174000",
                "RAG_API_KEY=api-secret",
                "RAG_METRICS_TOKEN=metrics-secret",
                "RAG_SERVICE_TENANT_ID=00000000-0000-0000-0000-000000000001",
                "RAG_SERVICE_ROLES=viewer",
                "POSTGRES_HOST=db.example.test",
                "POSTGRES_PORT=5432",
                "POSTGRES_DB=rag",
                "POSTGRES_ADMIN_USER=postgres_admin",
                "POSTGRES_PASSWORD=admin-secret",
                "POSTGRES_RUNTIME_USER=rag_runtime",
                "POSTGRES_APP_PASSWORD=runtime-secret",
                "POSTGRES_SSLMODE=verify-full",
                "POSTGRES_SSLROOTCERT=/etc/ssl/certs/ca-certificates.crt",
                "REDIS_URL=rediss://:redis-secret@redis.example.test:6379/0",
                "DEEPSEEK_API_KEY=deepseek-secret",
                "DEEPSEEK_API_URL=https://llm.example.test/v1",
                "DEEPSEEK_MODEL=deepseek-v4-flash-0731",
                "OIDC_ISSUER=https://id.example.test/realms/rag",
                "OIDC_AUDIENCE=rag-api",
                "OIDC_JWKS_URL=https://id.example.test/realms/rag/certs",
                "VITE_OIDC_ISSUER=https://id.example.test/realms/rag",
                "VITE_OIDC_CLIENT_ID=rag-frontend",
                "VITE_OIDC_AUDIENCE=rag-api",
                "S3_ENDPOINT_URL=https://s3.example.test",
                "S3_BUCKET=rag",
                "S3_REGION=us-east-1",
                "S3_ACCESS_KEY_ID=s3-access",
                "S3_SECRET_ACCESS_KEY=s3-secret",
                "CANARY_OIDC_TOKEN_URL=https://id.example.test/realms/rag/token",
                "CANARY_OIDC_CLIENT_AUTH_METHOD=client_secret_post",
                "CANARY_PRIMARY_CLIENT_ID=canary-primary",
                "CANARY_PRIMARY_CLIENT_SECRET=primary-secret-123456",
                "CANARY_SECONDARY_CLIENT_ID=canary-secondary",
                "CANARY_SECONDARY_CLIENT_SECRET=secondary-secret-123456",
                "CANARY_UPLOAD_CONTENT_B64=" + base64.b64encode(canary_sample).decode("ascii"),
                "CANARY_NO_ANSWER_QUERY=无线电频率许可",
                "CANARY_NO_ANSWER_EXPECTED_TEXT=当前知识库没有找到足够相关的信息",
            ]
        ),
        encoding="utf-8",
    )
    runtime_path = tmp_path / "runtime.yaml"
    migration_path = tmp_path / "migration.yaml"
    canary_path = tmp_path / "canary.yaml"

    render(
        env_file,
        namespace="rag-production",
        runtime_output=runtime_path,
        migration_output=migration_path,
        canary_output=canary_path,
    )

    runtime = runtime_path.read_text(encoding="utf-8")
    migration = migration_path.read_text(encoding="utf-8")
    suffix = release_resource_suffix(load_release_env_file(env_file))
    runtime_document = yaml.safe_load(runtime)
    migration_document = yaml.safe_load(migration)
    canary_document = yaml.safe_load(canary_path.read_text(encoding="utf-8"))
    assert runtime_document["metadata"]["name"] == f"rag-runtime-{suffix}"
    assert migration_document["metadata"]["name"] == f"rag-migration-{suffix}"
    assert canary_document["metadata"]["name"] == f"rag-canary-credentials-{suffix}"
    assert runtime_document["metadata"]["annotations"]["industrial-rag/config-digest"]
    assert runtime_document["immutable"] is True
    assert migration_document["immutable"] is True
    assert canary_document["immutable"] is True
    assert "  POSTGRES_USER:" in runtime
    assert "  POSTGRES_PASSWORD:" in runtime
    assert "  RAG_SERVICE_TENANT_ID:" in runtime
    assert "  RAG_SERVICE_ROLES:" in runtime
    assert "admin-secret" not in runtime
    assert "  POSTGRES_ADMIN_USER:" in migration
    assert "admin-secret" not in migration
    assert "CANARY_PRIMARY_CLIENT_SECRET" in canary_document["data"]
    assert Path(runtime_path).is_file() and Path(migration_path).is_file()


def test_model_source_manifest_is_generated_without_reading_model_weights(tmp_path):
    manifest_path = tmp_path / "model-manifest.json"
    write_manifest(manifest_path)

    assert validate_manifest(manifest_path) == []
    assert build_manifest()["models"]["bge-m3"]["revision"] == (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )
    canonical = Path("model-sources/model-manifest.json")
    assert manifest_path.read_bytes() == canonical.read_bytes()


def test_model_bundle_tree_hash_uses_platform_independent_path_order(tmp_path):
    model_root = tmp_path / "model"
    (model_root / "assets").mkdir(parents=True)
    contents = {
        "README.md": b"uppercase path\n",
        "alpha.txt": b"lowercase path\n",
        "assets/.metadata": b"hidden path\n",
        "assets/Z.bin": b"nested uppercase path\n",
    }
    for relative, content in contents.items():
        target = model_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    expected = hashlib.sha256()
    for relative in sorted(contents):
        relative_bytes = relative.encode("utf-8")
        content = contents[relative]
        expected.update(len(relative_bytes).to_bytes(8, "big"))
        expected.update(relative_bytes)
        expected.update(len(content).to_bytes(8, "big"))
        expected.update(content)

    assert tree_sha256(model_root) == expected.hexdigest()


def test_production_manifest_renderer_replaces_every_release_placeholder(tmp_path):
    manifest = tmp_path / "kustomize.yaml"
    manifest.write_text(
        "image: registry.example.invalid/industrial-rag-api@sha256:" + "0" * 64 + "\n"
        "model: sha256:REPLACE_WITH_APPROVED_MODEL_BUNDLE_DIGEST\n"
        "modelManifest: sha256:REPLACE_WITH_APPROVED_MODEL_MANIFEST_SHA256\n"
        "config: sha256:REPLACE_WITH_RELEASE_CONFIG_DIGEST\n"
        "host: rag.example.invalid\n"
        "oidc: https://id.example.invalid\n"
        'runtimeIssuer: "__RAG_RUNTIME_OIDC_ISSUER__"\n'
        'runtimeClient: "__RAG_RUNTIME_OIDC_CLIENT_ID__"\n'
        'runtimeAudience: "__RAG_RUNTIME_OIDC_AUDIENCE__"\n'
        's3Style: "__RAG_S3_ADDRESSING_STYLE__"\n'
        's3Failed: "__RAG_S3_FAILED_PREFIX__"\n'
        'releaseId: "__RAG_RELEASE_ID__"\n'
        'apiImage: "__RAG_API_IMAGE__"\n'
        'workerImage: "__RAG_WORKER_IMAGE__"\n'
        'frontendImage: "__RAG_FRONTEND_IMAGE__"\n'
        'modelImage: "__RAG_MODEL_BUNDLE_IMAGE__"\n'
        "runtimeConfig: rag-runtime-config\n"
        "frontendConfig: rag-frontend-config\n"
        "runtimeSecret: rag-runtime\n"
        "migrationSecret: rag-migration\n"
        "canarySecret: rag-canary-credentials\n"
        "canaryScript: rag-canary-script\n",
        encoding="utf-8",
    )
    env_file = tmp_path / "production.env"
    env_file.write_text(
        "\n".join(
            [
                "RELEASE_ID=123e4567-e89b-42d3-a456-426614174001",
                "API_IMAGE=registry.corp.test/rag/api@sha256:" + "a" * 64,
                "WORKER_IMAGE=registry.corp.test/rag/worker@sha256:" + "b" * 64,
                "FRONTEND_IMAGE=registry.corp.test/rag/frontend@sha256:" + "c" * 64,
                "MODEL_BUNDLE_IMAGE=registry.corp.test/rag/model-bundle@sha256:" + "e" * 64,
                "MODEL_BUNDLE_DIGEST=sha256:" + "e" * 64,
                "MODEL_MANIFEST_SHA256=sha256:" + "f" * 64,
                "INGRESS_HOST=rag.corp.test",
                "INGRESS_PROXY_CIDR=10.244.0.0/16",
                "OIDC_CONNECT_SRC=https://id.corp.test",
                "VITE_OIDC_ISSUER=https://id.corp.test/realms/rag",
                "VITE_OIDC_CLIENT_ID=rag-frontend",
                "VITE_OIDC_AUDIENCE=rag-api",
                "S3_ADDRESSING_STYLE=virtual",
                "S3_FAILED_PREFIX=quarantine",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "rendered.yaml"

    render_manifests(manifest, output, env_file)
    rendered = output.read_text(encoding="utf-8")
    suffix = release_resource_suffix(load_release_env_file(env_file))

    assert "example.invalid" not in rendered
    assert "sha256:" + "0" * 64 not in rendered
    assert "registry.corp.test/rag/api@sha256:" + "a" * 64 in rendered
    assert "REPLACE_WITH_RELEASE_CONFIG_DIGEST" not in rendered
    assert 'runtimeIssuer: "https://id.corp.test/realms/rag"' in rendered
    assert 's3Style: "virtual"' in rendered
    assert 's3Failed: "quarantine"' in rendered
    assert 'releaseId: "123e4567-e89b-42d3-a456-426614174001"' in rendered
    assert 'apiImage: "registry.corp.test/rag/api@sha256:' + "a" * 64 + '"' in rendered
    assert 'workerImage: "registry.corp.test/rag/worker@sha256:' + "b" * 64 + '"' in rendered
    assert 'frontendImage: "registry.corp.test/rag/frontend@sha256:' + "c" * 64 + '"' in rendered
    assert 'modelImage: "registry.corp.test/rag/model-bundle@sha256:' + "e" * 64 + '"' in rendered
    assert f"runtimeConfig: rag-runtime-config-{suffix}" in rendered
    assert f"frontendConfig: rag-frontend-config-{suffix}" in rendered
    assert f"runtimeSecret: rag-runtime-{suffix}" in rendered
    assert f"migrationSecret: rag-migration-{suffix}" in rendered
    assert f"canarySecret: rag-canary-credentials-{suffix}" in rendered
    assert f"canaryScript: rag-canary-script-{suffix}" in rendered

    exported = release_outputs(env_file)
    assert exported == {
        "api": "registry.corp.test/rag/api@sha256:" + "a" * 64,
        "worker": "registry.corp.test/rag/worker@sha256:" + "b" * 64,
        "frontend": "registry.corp.test/rag/frontend@sha256:" + "c" * 64,
        "model_bundle": "registry.corp.test/rag/model-bundle@sha256:" + "e" * 64,
        "model_bundle_digest": "sha256:" + "e" * 64,
        "model_manifest_sha256": "sha256:" + "f" * 64,
        "ingress_host": "rag.corp.test",
    }


def test_final_workload_images_must_match_verified_release_outputs():
    images = {
        "api": "registry.test/api@sha256:" + "a" * 64,
        "worker": "registry.test/worker@sha256:" + "b" * 64,
        "frontend": "registry.test/frontend@sha256:" + "c" * 64,
    }

    def workload(kind: str, name: str, image: str) -> dict:
        container_names = {
            "rag-api": "api",
            "rag-worker": "worker",
            "rag-frontend": "frontend",
            "rag-postgres-migration": "migration",
        }
        containers = [{"name": container_names.get(name, "api"), "image": image}]
        if name == "rag-canary":
            containers = [
                {"name": "smoke", "image": images["api"]},
                {"name": "candidate-worker", "image": images["worker"]},
            ]
        init_containers = []
        if (kind, name) in {
            ("Deployment", "rag-api"),
            ("Deployment", "rag-worker"),
            ("Job", "rag-canary"),
        }:
            init_containers = [{"name": "model-source-init", "image": image}]
        return {
            "apiVersion": "apps/v1" if kind == "Deployment" else "batch/v1",
            "kind": kind,
            "metadata": {"name": name},
            "spec": {
                "template": {
                    "spec": {
                        "containers": containers,
                        **({"initContainers": init_containers} if init_containers else {}),
                    }
                }
            },
        }

    documents = [
        workload("Deployment", "rag-api", images["api"]),
        workload("Deployment", "rag-worker", images["worker"]),
        workload("Deployment", "rag-frontend", images["frontend"]),
        workload("Job", "rag-postgres-migration", images["api"]),
        workload("Job", "rag-canary", images["api"]),
    ]

    assert validate_expected_release_images(documents, images) == []
    documents[0]["spec"]["template"]["spec"]["containers"][0]["image"] = (
        "registry.test/unverified@sha256:" + "d" * 64
    )
    assert validate_expected_release_images(documents, images) == [
        "Deployment/rag-api container api image does not match verified api image"
    ]

    documents[0]["spec"]["template"]["spec"]["containers"][0]["image"] = images["api"]
    documents[0]["spec"]["template"]["spec"]["initContainers"] = [
        {"image": "registry.test/attacker@sha256:" + "9" * 64}
    ]
    documents.append(workload("Deployment", "rogue", images["api"]))
    assert validate_expected_release_images(documents, images) == [
        "Deployment/rag-api must contain the exact bound initContainer set",
        "unexpected release workload Deployment/rogue",
    ]

    documents.pop()
    documents[0]["spec"]["template"]["spec"]["initContainers"] = [
        {"name": "model-source-init", "image": images["api"]}
    ]
    canary_containers = documents[4]["spec"]["template"]["spec"]["containers"]
    canary_containers[1]["image"] = images["api"]
    assert validate_expected_release_images(documents, images) == [
        "Job/rag-canary container candidate-worker image does not match verified worker image"
    ]
    canary_containers.append({"name": "rogue", "image": images["api"]})
    assert "Job/rag-canary must contain the exact bound container set" in (
        validate_expected_release_images(documents, images)
    )


def test_production_manifest_inventory_rejects_duplicate_and_extra_platform_objects(
    tmp_path,
):
    manifest = tmp_path / "ambiguous-platform.yaml"
    manifest.write_text(
        """apiVersion: v1
kind: Namespace
metadata:
  name: rag-production
---
apiVersion: v1
kind: Namespace
metadata:
  name: rag-production
---
apiVersion: v1
kind: Service
metadata:
  name: rogue-service
spec:
  ports:
    - port: 80
""",
        encoding="utf-8",
    )

    errors = validate_kubernetes_manifest(manifest, allow_placeholders=True)

    assert "duplicate production resource Namespace/rag-production" in errors
    assert "unexpected Service/rogue-service" in errors


@pytest.mark.parametrize("probe_name", ["readinessProbe", "livenessProbe"])
def test_production_manifest_rejects_null_deployment_probe(tmp_path, probe_name):
    source = REPOSITORY_ROOT / "industrial-rag/deploy/kustomize/base/deployments.yaml"
    documents = list(yaml.safe_load_all(source.read_text(encoding="utf-8")))
    deployment = next(
        item
        for item in documents
        if item.get("kind") == "Deployment" and item["metadata"]["name"] == "rag-api"
    )
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    container[probe_name] = None
    manifest = tmp_path / f"null-{probe_name}.yaml"
    manifest.write_text(
        yaml.safe_dump_all(documents, sort_keys=False),
        encoding="utf-8",
    )

    errors = validate_kubernetes_manifest(manifest, allow_placeholders=True)

    assert f"Deployment/rag-api container api {probe_name} must be a non-empty mapping" in errors


@pytest.mark.parametrize("probe_name", ["readinessProbe", "livenessProbe"])
@pytest.mark.parametrize(
    "probe",
    [
        None,
        {},
        [],
        "httpGet",
        False,
        1,
        {"periodSeconds": 10},
        {"custom": {"port": 8080}},
        {"httpGet": None},
        {"httpGet": []},
        {"httpGet": {}},
        {"httpGet": {"path": "/health"}},
        {"httpGet": {"port": True}},
        {"exec": {"command": []}},
        {"exec": {"command": [1]}},
        {"tcpSocket": {"host": "localhost"}},
        {"grpc": {"service": "health"}},
        {"grpc": {"port": "grpc"}},
        {"httpGet": {"port": "http"}, "exec": {"command": ["true"]}},
    ],
    ids=[
        "null",
        "empty-mapping",
        "sequence",
        "string",
        "boolean",
        "number",
        "no-handler",
        "unsupported-handler",
        "null-handler",
        "sequence-handler",
        "empty-handler",
        "http-without-port",
        "boolean-port",
        "empty-command",
        "non-string-command",
        "tcp-without-port",
        "grpc-without-port",
        "grpc-named-port",
        "multiple-handlers",
    ],
)
def test_probe_contract_rejects_malformed_semantic_shapes(probe_name, probe):
    assert _validate_probe_contract(
        "Deployment",
        "rag-api",
        "api",
        probe_name,
        probe,
    )


@pytest.mark.parametrize(
    "probe",
    [
        {
            "exec": {"command": ["python", "-c", "raise SystemExit(0)"]},
            "periodSeconds": 10,
        },
        {"httpGet": {"path": "/health/ready", "port": "http"}},
        {"tcpSocket": {"host": "127.0.0.1", "port": 5432}},
        {"grpc": {"port": 9090, "service": "health"}},
    ],
    ids=["exec", "http-get", "tcp-socket", "grpc"],
)
def test_probe_contract_accepts_each_supported_handler(probe):
    assert (
        _validate_probe_contract(
            "Deployment",
            "rag-api",
            "api",
            "readinessProbe",
            probe,
        )
        == []
    )


def test_probe_named_ports_match_kubernetes_contract():
    invalid_errors = _validate_probe_contract(
        "Deployment",
        "rag-api",
        "api",
        "readinessProbe",
        {"httpGet": {"port": "http--admin"}},
    )
    valid_errors = _validate_probe_contract(
        "Deployment",
        "rag-api",
        "api",
        "readinessProbe",
        {"httpGet": {"port": "8080-http"}},
    )

    assert invalid_errors
    assert valid_errors == []


def test_workload_identity_and_security_contract_is_exact_and_fail_closed():
    service_accounts = {
        ("Deployment", "rag-api"): "rag-api",
        ("Deployment", "rag-worker"): "rag-worker",
        ("Deployment", "rag-frontend"): "rag-frontend",
        ("Job", "rag-postgres-migration"): "rag-migrate",
        ("Job", "rag-canary"): "rag-canary",
    }
    container_names = {
        ("Deployment", "rag-api"): ["api"],
        ("Deployment", "rag-worker"): ["worker"],
        ("Deployment", "rag-frontend"): ["frontend"],
        ("Job", "rag-postgres-migration"): ["migration"],
        ("Job", "rag-canary"): ["smoke", "candidate-worker"],
    }
    init_container_names = {
        ("Deployment", "rag-api"): ["model-source-init"],
        ("Deployment", "rag-worker"): ["model-source-init"],
        ("Deployment", "rag-frontend"): [],
        ("Job", "rag-postgres-migration"): [],
        ("Job", "rag-canary"): ["model-source-init"],
    }

    def hardened_pod(kind: str, name: str) -> dict:
        numeric_id = 101 if (kind, name) == ("Deployment", "rag-frontend") else 10001
        container_security = {
            "runAsNonRoot": True,
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
        }
        return {
            "serviceAccountName": service_accounts[(kind, name)],
            "automountServiceAccountToken": False,
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": numeric_id,
                "runAsGroup": numeric_id,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": container_name,
                    "securityContext": copy.deepcopy(container_security),
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "128Mi"},
                        "limits": {"cpu": "1", "memory": "1Gi"},
                    },
                }
                for container_name in container_names[(kind, name)]
            ],
            "initContainers": [
                {
                    "name": container_name,
                    "securityContext": copy.deepcopy(container_security),
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "128Mi"},
                        "limits": {"cpu": "1", "memory": "1Gi"},
                    },
                }
                for container_name in init_container_names[(kind, name)]
            ],
        }

    for workload in service_accounts:
        assert _validate_pod_security_contract(*workload, hardened_pod(*workload)) == []

    pod = hardened_pod("Job", "rag-canary")
    pod["serviceAccountName"] = "default"
    assert "Job/rag-canary must not use the default ServiceAccount" in (
        _validate_pod_security_contract("Job", "rag-canary", pod)
    )

    pod = hardened_pod("Job", "rag-canary")
    pod["serviceAccountName"] = "rag-migrate"
    assert "Job/rag-canary must use ServiceAccount/rag-canary" in (
        _validate_pod_security_contract("Job", "rag-canary", pod)
    )

    pod = hardened_pod("Deployment", "rag-api")
    pod.pop("automountServiceAccountToken")
    pod["securityContext"]["seccompProfile"]["type"] = "Unconfined"
    pod["containers"][0]["securityContext"]["runAsNonRoot"] = False
    pod["containers"][0]["securityContext"]["capabilities"]["drop"] = []
    errors = _validate_pod_security_contract("Deployment", "rag-api", pod)
    assert "Deployment/rag-api must set automountServiceAccountToken=false" in errors
    assert "Deployment/rag-api must set seccompProfile.type=RuntimeDefault" in errors
    assert "Deployment/rag-api container api must set runAsNonRoot=true" in errors
    assert "Deployment/rag-api container api must drop all Linux capabilities" in errors

    pod = hardened_pod("Deployment", "rag-worker")
    pod["containers"].append(copy.deepcopy(pod["containers"][0]) | {"name": "rogue"})
    assert "Deployment/rag-worker must contain exactly these containers: worker" in (
        _validate_pod_security_contract("Deployment", "rag-worker", pod)
    )

    pod = hardened_pod("Deployment", "rag-worker")
    pod["initContainers"] = [
        {"name": "unexpected-init", "securityContext": pod["containers"][0]["securityContext"]}
    ]
    assert "Deployment/rag-worker must contain exactly these initContainers: model-source-init" in (
        _validate_pod_security_contract("Deployment", "rag-worker", pod)
    )

    pod = hardened_pod("Deployment", "rag-api")
    pod["hostNetwork"] = True
    pod["volumes"] = [{"name": "host", "hostPath": {"path": "/"}}]
    container = pod["containers"][0]
    container["securityContext"]["privileged"] = True
    container["securityContext"]["capabilities"]["add"] = ["NET_ADMIN"]
    container["ports"] = [{"containerPort": 8000, "hostPort": 8000}]
    container["resources"]["limits"].pop("memory")
    errors = _validate_pod_security_contract("Deployment", "rag-api", pod)
    assert "Deployment/rag-api must not enable hostNetwork" in errors
    assert "Deployment/rag-api must not mount hostPath volumes" in errors
    assert "Deployment/rag-api container api must not be privileged" in errors
    assert "Deployment/rag-api container api must not add Linux capabilities" in errors
    assert "Deployment/rag-api container api must not bind hostPort" in errors
    assert "Deployment/rag-api container api must set limits.memory" in errors


def test_checked_in_service_accounts_are_tokenless_and_role_specific():
    path = REPOSITORY_ROOT / "industrial-rag/deploy/kustomize/base/serviceaccounts.yaml"
    service_accounts = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    expected = {"rag-api", "rag-worker", "rag-frontend", "rag-migrate", "rag-canary"}

    assert {item["metadata"]["name"] for item in service_accounts} == expected
    for item in service_accounts:
        assert _validate_service_account_contract(item["metadata"]["name"], item) == []

    assert _validate_service_account_contract(
        "rag-api", {"automountServiceAccountToken": True}
    ) == ["ServiceAccount/rag-api must set automountServiceAccountToken=false"]


def test_release_output_rejects_shell_image_injection(tmp_path):
    env_file = tmp_path / "release.env"
    env_file.write_text(
        "\n".join(
            [
                "API_IMAGE=registry.test/$(id)/api@sha256:" + "a" * 64,
                "WORKER_IMAGE=registry.test/rag/worker@sha256:" + "b" * 64,
                "FRONTEND_IMAGE=registry.test/rag/frontend@sha256:" + "c" * 64,
                "MODEL_BUNDLE_IMAGE=registry.test/rag/model@sha256:" + "d" * 64,
                "MODEL_MANIFEST_SHA256=sha256:" + "e" * 64,
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe immutable OCI image"):
        release_outputs(env_file)


def test_runtime_password_rotation_requires_a_new_versioned_role(tmp_path):
    import base64

    def secret(path: Path, user: str, password: str) -> None:
        encoded = {
            "POSTGRES_USER": base64.b64encode(user.encode()).decode(),
            "POSTGRES_PASSWORD": base64.b64encode(password.encode()).decode(),
        }
        path.write_text(
            "apiVersion: v1\nkind: Secret\ndata:\n"
            + "".join(f"  {key}: {value}\n" for key, value in encoded.items()),
            encoding="utf-8",
        )

    current = tmp_path / "current.yaml"
    desired = tmp_path / "desired.yaml"
    secret(current, "rag_runtime_v1", "old-password")
    secret(desired, "rag_runtime_v1", "new-password")

    assert validate_rotation(current, desired) == [
        "runtime PostgreSQL password rotation must use a new versioned "
        "POSTGRES_RUNTIME_USER so old and new pods can overlap safely"
    ]

    secret(desired, "rag_runtime_v2", "new-password")
    assert validate_rotation(current, desired) == []


def test_release_uses_exact_cosign_identity_and_source_bound_images():
    workflow = (REPOSITORY_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert '--certificate-identity "$identity"' in workflow
    assert "--certificate-identity-regexp" not in workflow
    assert 'test "$revision" = "$EXPECTED_REVISION"' in workflow
    assert 'test "$source" = "$EXPECTED_SOURCE"' in workflow


def test_release_rollback_is_armed_only_for_changed_generations():
    workflow = (REPOSITORY_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "changed-deployments/$deployment" in workflow
    assert "previous-generations/$deployment" in workflow
    assert '[ "$current_generation" != "$release_generation" ]' in workflow
    assert 'rollout undo "deployment/$deployment" --to-revision="$previous_revision"' in workflow


def test_release_rollout_supports_bootstrap_without_deleting_concurrent_objects():
    workflow = (REPOSITORY_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "--ignore-not-found" in workflow
    assert "new-deployments/$deployment" in workflow
    assert '[ "$current_uid" != "$release_uid" ]' in workflow
    assert 'delete "deployment/$deployment" --wait=true' in workflow


def test_release_preserves_and_preflights_previous_versioned_resource_refs():
    workflow = (REPOSITORY_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "Snapshot stable rollback state and validate previous release resources" in workflow
    assert "previous-resource-refs/$deployment" in workflow
    assert "scripts/list_workload_resource_refs.py" in workflow
    assert "--prune" not in workflow
    assert 'delete "configmap/' not in workflow
    assert 'delete "secret/' not in workflow


def test_application_release_never_mutates_shared_platform_baseline():
    workflow = (REPOSITORY_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "Verify pre-provisioned platform baseline has no drift" in workflow
    assert "kubectl diff --server-side" in workflow
    assert "platform-baseline.yaml" in workflow
    assert "release-configmaps.yaml" in workflow
    for kind in (
        "Namespace",
        "ServiceAccount",
        "Service",
        "Ingress",
        "PodDisruptionBudget",
        "NetworkPolicy",
    ):
        assert f"--kind {kind}" in workflow
    assert "--exclude-kind" not in workflow
    assert "Apply static production resources" not in workflow


@pytest.mark.parametrize(
    "relative_path",
    [
        ".github/workflows/release.yml",
        ".github/workflows/publish-production-images.yml",
    ],
)
def test_job_environment_does_not_use_step_only_runner_context(relative_path):
    source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    workflow = yaml.load(source, Loader=yaml.BaseLoader)

    for job_name, job in workflow["jobs"].items():
        for variable, value in (job.get("env") or {}).items():
            assert "${{ runner." not in str(value), (
                f"{relative_path} job {job_name} env {variable} uses runner context "
                "before a runner is allocated"
            )


def test_actionlint_knows_protected_self_hosted_runner_labels():
    config = yaml.safe_load(
        (REPOSITORY_ROOT / ".github/actionlint.yaml").read_text(encoding="utf-8")
    )

    assert set(config["self-hosted-runner"]["labels"]) == {
        "rag-evaluation-source",
        "rag-production",
    }


def test_failure_rollback_runs_after_the_last_protected_artifact_gate():
    workflow = (REPOSITORY_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    artifact_step = workflow.index("- name: Upload protected release artifacts")
    rollback_step = workflow.index(
        "- name: Restore exact previous Deployment revisions after any release failure"
    )
    assert artifact_step < rollback_step


def test_named_gpu_contract_is_independent_of_canary_container_order():
    pod_spec = {
        "containers": [
            {
                "name": "candidate-worker",
                "resources": {"requests": {"cpu": "1"}, "limits": {"cpu": "2"}},
            },
            {
                "name": "smoke",
                "resources": {
                    "requests": {"nvidia.com/gpu": "1"},
                    "limits": {"nvidia.com/gpu": "1"},
                },
            },
        ]
    }

    assert _container_gpu_contract(pod_spec, "smoke") == ("1", "1")
    assert _container_gpu_contract(pod_spec, "candidate-worker") == ("", "")


def test_release_resource_identity_never_hashes_secret_snapshot_values():
    release_id = "123e4567-e89b-42d3-a456-426614174003"
    first = {"RELEASE_ID": release_id, "RAG_API_KEY": "first-secret"}
    second = {"RELEASE_ID": release_id, "RAG_API_KEY": "different-secret"}

    assert release_config_digest(first) == release_config_digest(second)
    assert release_resource_suffix(first) == release_resource_suffix(second)
    assert len(release_resource_suffix(first)) == 20

    with pytest.raises(ValueError, match="UUIDv4"):
        release_config_digest({"RELEASE_ID": "predictable-release-name"})


def test_functional_canary_settings_fail_closed_and_parse_sse():
    canary = _load_canary_module()
    sample = base64.b64encode(("生产发布验证样本。" * 20).encode("utf-8")).decode("ascii")
    values = {
        "CANARY_OIDC_TOKEN_URL": "https://id.corp.test/token",
        "CANARY_OIDC_CLIENT_AUTH_METHOD": "client_secret_post",
        "CANARY_PRIMARY_CLIENT_ID": "primary",
        "CANARY_PRIMARY_CLIENT_SECRET": "primary-secret-123456",
        "CANARY_SECONDARY_CLIENT_ID": "secondary",
        "CANARY_SECONDARY_CLIENT_SECRET": "secondary-secret-123456",
        "CANARY_UPLOAD_CONTENT_B64": sample,
        "CANARY_NO_ANSWER_QUERY": "无线电频率许可",
        "CANARY_NO_ANSWER_EXPECTED_TEXT": "当前知识库没有找到足够相关的信息",
    }

    settings = canary.load_settings(values)
    assert settings.upload_content.startswith("生产发布验证样本")
    events = canary.parse_sse_events(
        b'data: {"type":"chunk","data":"ok"}\n\ndata: {"type":"done"}\n\n'
    )
    assert [event["type"] for event in events] == ["chunk", "done"]

    values.pop("CANARY_SECONDARY_CLIENT_SECRET")
    with pytest.raises(canary.CanaryError, match="CANARY_SECONDARY_CLIENT_SECRET"):
        canary.load_settings(values)


def test_functional_canary_uses_a_unique_non_colliding_source_identity(monkeypatch):
    canary = _load_canary_module()
    captured = {}

    def fake_request(_url, *, method="GET", headers=None, body=None, timeout):
        captured["body"] = body
        return 200, {}, b'{"task_id":"task-1"}'

    monkeypatch.setattr(canary, "_request", fake_request)
    marker = "industrial_rag_release_canary_" + "a" * 32

    assert (
        canary._multipart_upload(
            "http://127.0.0.1:18000",
            "token",
            "x" * 100,
            marker,
            timeout=5,
        )
        == "task-1"
    )
    payload = captured["body"]
    assert isinstance(payload, bytes)
    assert f'"source_id":"release-canary-functional-fixture-{marker}"'.encode() in payload
    assert f'filename="release-canary-{marker}.txt"'.encode() in payload


def test_functional_canary_cleanup_is_mandatory_and_verified(monkeypatch):
    canary = _load_canary_module()
    statuses = iter((200, 404))
    methods = []

    def fake_request(_url, *, method="GET", headers=None, body=None, timeout):
        methods.append(method)
        return next(statuses), {}, b"{}"

    monkeypatch.setattr(canary, "_request", fake_request)

    canary._cleanup_document(
        "http://127.0.0.1:18000",
        "token",
        "document-id",
        timeout=5,
    )

    assert methods == ["DELETE", "DELETE"]


def test_rollback_reference_inventory_covers_all_pod_reference_forms():
    document = {
        "kind": "Deployment",
        "spec": {
            "template": {
                "spec": {
                    "imagePullSecrets": [{"name": "registry-auth"}],
                    "containers": [
                        {
                            "envFrom": [
                                {"configMapRef": {"name": "runtime-config"}},
                                {"secretRef": {"name": "runtime-secret"}},
                            ],
                            "env": [
                                {
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "field-secret",
                                            "key": "value",
                                        }
                                    }
                                }
                            ],
                        }
                    ],
                    "volumes": [
                        {"configMap": {"name": "script-config"}},
                        {"projected": {"sources": [{"secret": {"name": "projected-secret"}}]}},
                    ],
                }
            }
        },
    }

    assert workload_resource_refs(document) == [
        "configmap/runtime-config",
        "configmap/script-config",
        "secret/field-secret",
        "secret/projected-secret",
        "secret/registry-auth",
        "secret/runtime-secret",
    ]
