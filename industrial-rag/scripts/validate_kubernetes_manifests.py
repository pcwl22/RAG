"""Validate the portable production manifest contract without a live cluster."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$", re.IGNORECASE)
_ZERO_DIGEST = "@sha256:" + "0" * 64
_NAMED_PORT = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,13}[a-z0-9])?")
_PROBE_HANDLERS = ("exec", "httpGet", "tcpSocket", "grpc")
_REQUIRED_DEPLOYMENTS = {"rag-api", "rag-worker", "rag-frontend"}
_REQUIRED_NAMESPACES = {"rag-production"}
_REQUIRED_SERVICE_ACCOUNTS = {
    "rag-api",
    "rag-worker",
    "rag-frontend",
    "rag-migrate",
    "rag-canary",
}
_REQUIRED_SERVICES = {"rag-api", "rag-frontend"}
_REQUIRED_INGRESSES = {"rag"}
_REQUIRED_PDBS = {"rag-api", "rag-worker", "rag-frontend"}
_REQUIRED_JOBS = {"rag-postgres-migration", "rag-canary"}
_REQUIRED_NETWORK_POLICIES = {
    "rag-default-deny",
    "rag-api-ingress",
    "rag-frontend-ingress",
    "rag-frontend-egress",
    "rag-egress",
}
_API_VERSIONS = {
    "Namespace": "v1",
    "ServiceAccount": "v1",
    "ConfigMap": "v1",
    "Deployment": "apps/v1",
    "Service": "v1",
    "Ingress": "networking.k8s.io/v1",
    "PodDisruptionBudget": "policy/v1",
    "Job": "batch/v1",
    "NetworkPolicy": "networking.k8s.io/v1",
}
_RELEASE_CONTAINER_ROLES = {
    ("Deployment", "rag-api"): {"api": "api"},
    ("Deployment", "rag-worker"): {"worker": "worker"},
    ("Deployment", "rag-frontend"): {"frontend": "frontend"},
    ("Job", "rag-postgres-migration"): {"migration": "api"},
    ("Job", "rag-canary"): {"smoke": "api", "candidate-worker": "worker"},
}
_RELEASE_INIT_CONTAINER_ROLES = {
    ("Deployment", "rag-api"): {"model-source-init": "api"},
    ("Deployment", "rag-worker"): {"model-source-init": "worker"},
    ("Job", "rag-canary"): {"model-source-init": "api"},
}
_WORKLOAD_SERVICE_ACCOUNTS = {
    ("Deployment", "rag-api"): "rag-api",
    ("Deployment", "rag-worker"): "rag-worker",
    ("Deployment", "rag-frontend"): "rag-frontend",
    ("Job", "rag-postgres-migration"): "rag-migrate",
    ("Job", "rag-canary"): "rag-canary",
}
_WORKLOAD_NUMERIC_IDS = {
    ("Deployment", "rag-api"): 10001,
    ("Deployment", "rag-worker"): 10001,
    ("Deployment", "rag-frontend"): 101,
    ("Job", "rag-postgres-migration"): 10001,
    ("Job", "rag-canary"): 10001,
}


def _load_documents(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for item in yaml.safe_load_all(path.read_text(encoding="utf-8")):
        if isinstance(item, dict) and item.get("kind"):
            documents.append(item)
    return documents


def _validate_service_account_contract(
    name: str, service_account: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if name == "default":
        errors.append("ServiceAccount/default must not be used by production workloads")
    if service_account.get("automountServiceAccountToken") is not False:
        errors.append(f"ServiceAccount/{name} must set automountServiceAccountToken=false")
    return errors


def _validate_pod_security_contract(kind: str, name: str, pod_spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    workload_key = (kind, name)
    expected_service_account = _WORKLOAD_SERVICE_ACCOUNTS.get(workload_key)
    service_account = str(pod_spec.get("serviceAccountName") or "")
    if service_account in {"", "default"}:
        errors.append(f"{kind}/{name} must not use the default ServiceAccount")
    elif expected_service_account and service_account != expected_service_account:
        errors.append(
            f"{kind}/{name} must use ServiceAccount/{expected_service_account}"
        )
    if pod_spec.get("automountServiceAccountToken") is not False:
        errors.append(f"{kind}/{name} must set automountServiceAccountToken=false")
    for field in ("hostNetwork", "hostPID", "hostIPC", "shareProcessNamespace"):
        if pod_spec.get(field) is True:
            errors.append(f"{kind}/{name} must not enable {field}")
    for volume in pod_spec.get("volumes") or []:
        if isinstance(volume, dict) and "hostPath" in volume:
            errors.append(f"{kind}/{name} must not mount hostPath volumes")

    pod_security = pod_spec.get("securityContext") or {}
    if pod_security.get("runAsNonRoot") is not True:
        errors.append(f"{kind}/{name} must set runAsNonRoot=true")
    expected_numeric_id = _WORKLOAD_NUMERIC_IDS.get(workload_key)
    for field in ("runAsUser", "runAsGroup"):
        value = pod_security.get(field)
        if not isinstance(value, int) or value <= 0 or value != expected_numeric_id:
            errors.append(
                f"{kind}/{name} must set {field}={expected_numeric_id}"
            )
    seccomp = pod_security.get("seccompProfile") or {}
    if seccomp.get("type") != "RuntimeDefault":
        errors.append(f"{kind}/{name} must set seccompProfile.type=RuntimeDefault")

    containers = pod_spec.get("containers") or []
    expected_containers = set((_RELEASE_CONTAINER_ROLES.get(workload_key) or {}).keys())
    actual_names = [
        str(container.get("name") or "")
        for container in containers
        if isinstance(container, dict)
    ]
    if (
        len(containers) != len(expected_containers)
        or len(actual_names) != len(expected_containers)
        or set(actual_names) != expected_containers
    ):
        expected_text = ", ".join(sorted(expected_containers))
        errors.append(
            f"{kind}/{name} must contain exactly these containers: {expected_text}"
        )

    init_containers = pod_spec.get("initContainers") or []
    expected_init_containers = set(
        (_RELEASE_INIT_CONTAINER_ROLES.get(workload_key) or {}).keys()
    )
    actual_init_names = [
        str(container.get("name") or "")
        for container in init_containers
        if isinstance(container, dict)
    ]
    if (
        len(init_containers) != len(expected_init_containers)
        or len(actual_init_names) != len(expected_init_containers)
        or set(actual_init_names) != expected_init_containers
    ):
        expected_text = ", ".join(sorted(expected_init_containers)) or "none"
        errors.append(
            f"{kind}/{name} must contain exactly these initContainers: {expected_text}"
        )

    for container_kind, declared_containers in (
        ("container", containers),
        ("initContainer", init_containers),
    ):
        for container in declared_containers:
            if not isinstance(container, dict):
                errors.append(f"{kind}/{name} contains an invalid {container_kind}")
                continue
            container_name = str(container.get("name") or "<unnamed>")
            security = container.get("securityContext") or {}
            prefix = f"{kind}/{name} {container_kind} {container_name}"
            if security.get("runAsNonRoot") is not True:
                errors.append(f"{prefix} must set runAsNonRoot=true")
            if security.get("allowPrivilegeEscalation") is not False:
                errors.append(f"{prefix} must disable privilege escalation")
            if security.get("privileged") is True:
                errors.append(f"{prefix} must not be privileged")
            if security.get("readOnlyRootFilesystem") is not True:
                errors.append(f"{prefix} must use a read-only root filesystem")
            capabilities = security.get("capabilities") or {}
            dropped = capabilities.get("drop")
            if not isinstance(dropped, list) or {
                str(capability).upper() for capability in dropped
            } != {"ALL"}:
                errors.append(f"{prefix} must drop all Linux capabilities")
            if capabilities.get("add"):
                errors.append(f"{prefix} must not add Linux capabilities")
            for field in ("runAsUser", "runAsGroup"):
                if field in security and security[field] != expected_numeric_id:
                    errors.append(f"{prefix} must not override {field}={expected_numeric_id}")
            for port in container.get("ports") or []:
                if isinstance(port, dict) and port.get("hostPort") is not None:
                    errors.append(f"{prefix} must not bind hostPort")
            resources = container.get("resources") or {}
            for resource_scope in ("requests", "limits"):
                values = resources.get(resource_scope) or {}
                for resource_name in ("cpu", "memory"):
                    if not str(values.get(resource_name) or "").strip():
                        errors.append(
                            f"{prefix} must set {resource_scope}.{resource_name}"
                        )
    return errors


def _container_gpu_contract(pod_spec: dict[str, Any], container_name: str) -> tuple[str, str]:
    """Return GPU request/limit for one named container, independent of order."""
    container: dict[str, Any] = next(
        (
            item
            for item in pod_spec.get("containers") or []
            if str(item.get("name") or "") == container_name
        ),
        {},
    )
    resources = container.get("resources") or {}
    request = str((resources.get("requests") or {}).get("nvidia.com/gpu") or "")
    limit = str((resources.get("limits") or {}).get("nvidia.com/gpu") or "")
    return request, limit


def _validate_model_cache_contract(
    kind: str, name: str, pod_spec: dict[str, Any]
) -> list[str]:
    """Require one writable downloader and read-only model consumer mounts."""
    consumers = {
        ("Deployment", "rag-api"): {"api"},
        ("Deployment", "rag-worker"): {"worker"},
        ("Job", "rag-canary"): {"smoke", "candidate-worker"},
    }.get((kind, name))
    if consumers is None:
        return []

    errors: list[str] = []
    init_containers = {
        str(container.get("name") or ""): container
        for container in pod_spec.get("initContainers") or []
        if isinstance(container, dict)
    }
    initializer = init_containers.get("model-source-init") or {}
    if initializer.get("command") != [
        "python",
        "-m",
        "app.embedding.model_bundle",
        "prepare",
    ]:
        errors.append(f"{kind}/{name} model-source-init must run the pinned downloader")

    init_mount: dict[str, Any] = next(
        (
            mount
            for mount in initializer.get("volumeMounts") or []
            if mount.get("name") == "model-cache"
        ),
        {},
    )
    if init_mount.get("mountPath") != "/app/models" or init_mount.get("readOnly") is True:
        errors.append(
            f"{kind}/{name} model-source-init must mount writable model-cache at /app/models"
        )

    by_container = {
        str(container.get("name") or ""): container
        for container in pod_spec.get("containers") or []
        if isinstance(container, dict)
    }
    for consumer_name in sorted(consumers):
        consumer = by_container.get(consumer_name) or {}
        mount: dict[str, Any] = next(
            (
                item
                for item in consumer.get("volumeMounts") or []
                if item.get("name") == "model-cache"
            ),
            {},
        )
        if mount.get("mountPath") != "/app/models" or mount.get("readOnly") is not True:
            errors.append(
                f"{kind}/{name} container {consumer_name} must mount model-cache read-only"
            )

    model_volume: dict[str, Any] = next(
        (
            volume
            for volume in pod_spec.get("volumes") or []
            if volume.get("name") == "model-cache"
        ),
        {},
    )
    empty_dir = model_volume.get("emptyDir")
    if not isinstance(empty_dir, dict) or empty_dir.get("sizeLimit") != "16Gi":
        errors.append(f"{kind}/{name} must provide a 16Gi emptyDir model-cache")
    return errors


def _valid_probe_port(value: Any, *, allow_named: bool) -> bool:
    if type(value) is int:
        return 1 <= value <= 65535
    return bool(
        allow_named
        and isinstance(value, str)
        and _NAMED_PORT.fullmatch(value)
        and any("a" <= character <= "z" for character in value)
        and "--" not in value
    )


def _validate_probe_contract(
    kind: str,
    workload_name: str,
    container_name: str,
    probe_name: str,
    probe: Any,
) -> list[str]:
    """Require one effective Kubernetes probe handler after YAML parsing."""
    prefix = f"{kind}/{workload_name} container {container_name} {probe_name}"
    if not isinstance(probe, Mapping) or not probe:
        return [f"{prefix} must be a non-empty mapping"]

    handlers = [handler for handler in _PROBE_HANDLERS if handler in probe]
    if len(handlers) != 1:
        supported = ", ".join(_PROBE_HANDLERS)
        return [f"{prefix} must define exactly one handler: {supported}"]

    handler_name = handlers[0]
    handler = probe[handler_name]
    if not isinstance(handler, Mapping) or not handler:
        return [f"{prefix} {handler_name} handler must be a non-empty mapping"]

    if handler_name == "exec":
        command = handler.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(argument, str) for argument in command)
            or not command[0].strip()
        ):
            return [f"{prefix} exec handler must define a non-empty command list"]
    elif handler_name in {"httpGet", "tcpSocket"}:
        if not _valid_probe_port(handler.get("port"), allow_named=True):
            return [
                f"{prefix} {handler_name} handler must define a valid numeric or named port"
            ]
    elif not _valid_probe_port(handler.get("port"), allow_named=False):
        return [f"{prefix} grpc handler must define a valid numeric port"]

    return []


def _release_resource_name(base: str, suffix: str | None) -> str:
    return f"{base}-{suffix}" if suffix else base


def _deployment_config_suffix(deployment: dict[str, Any] | None) -> str | None:
    if not deployment:
        return None
    annotations = (
        ((deployment.get("spec") or {}).get("template") or {}).get("metadata") or {}
    ).get("annotations") or {}
    digest = str(annotations.get("industrial-rag/config-digest") or "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest, re.IGNORECASE):
        return digest.split(":", 1)[1][:20].lower()
    return None


def validate_expected_release_images(
    documents: list[dict[str, Any]], expected_images: Mapping[str, str]
) -> list[str]:
    """Bind every deployable workload image to the references already verified."""
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in documents:
        kind = str(item.get("kind") or "")
        name = str((item.get("metadata") or {}).get("name") or "")
        if kind not in {"Deployment", "Job"}:
            continue
        expected_containers = _RELEASE_CONTAINER_ROLES.get((kind, name))
        if expected_containers is None:
            errors.append(f"unexpected release workload {kind}/{name}")
            continue
        seen.add((kind, name))
        pod_spec = ((item.get("spec") or {}).get("template") or {}).get("spec") or {}
        containers = pod_spec.get("containers") or []
        init_containers = pod_spec.get("initContainers") or []
        expected_init_containers = _RELEASE_INIT_CONTAINER_ROLES.get((kind, name)) or {}
        actual_names = {str(container.get("name") or "") for container in containers}
        if len(containers) != len(expected_containers) or actual_names != set(expected_containers):
            errors.append(f"{kind}/{name} must contain the exact bound container set")
        actual_init_names = {
            str(container.get("name") or "") for container in init_containers
        }
        if (
            len(init_containers) != len(expected_init_containers)
            or actual_init_names != set(expected_init_containers)
        ):
            errors.append(f"{kind}/{name} must contain the exact bound initContainer set")
        bound_containers = [
            (container, expected_containers) for container in containers
        ] + [
            (container, expected_init_containers) for container in init_containers
        ]
        for container, roles in bound_containers:
            container_name = str(container.get("name") or "")
            role = roles.get(container_name)
            if role is None:
                continue
            expected = str(expected_images.get(role) or "").strip()
            if not expected:
                errors.append(f"expected {role} release image is missing")
                continue
            actual = str(container.get("image") or "").strip()
            if actual != expected:
                errors.append(
                    f"{kind}/{name} container {container_name} image does not match verified {role} image"
                )
    for resource in sorted(set(_RELEASE_CONTAINER_ROLES) - seen):
        errors.append(f"missing release image binding target {resource[0]}/{resource[1]}")
    return errors


def validate_manifest(
    path: Path,
    *,
    allow_placeholders: bool = False,
    expected_images: Mapping[str, str] | None = None,
) -> list[str]:
    documents = _load_documents(path)
    errors: list[str] = []
    for item in documents:
        kind = str(item.get("kind") or "")
        name = str((item.get("metadata") or {}).get("name") or "")
        if not name:
            errors.append(f"{kind or 'resource'} is missing metadata.name")
        namespace = str((item.get("metadata") or {}).get("namespace") or "")
        if kind != "Namespace" and namespace != "rag-production":
            errors.append(f"{kind}/{name} must be scoped to namespace rag-production")
        expected_api_version = _API_VERSIONS.get(kind)
        if expected_api_version and item.get("apiVersion") != expected_api_version:
            errors.append(f"{kind}/{name} must use apiVersion {expected_api_version}")
        elif not expected_api_version:
            errors.append(f"unsupported Kubernetes kind: {kind or 'unknown'}")

        spec = item.get("spec")
        if kind in {"Deployment", "Job"}:
            pod_template = (spec or {}).get("template") if isinstance(spec, dict) else None
            pod_spec = (pod_template or {}).get("spec") if isinstance(pod_template, dict) else None
            if not isinstance(spec, dict) or not isinstance(pod_template, dict):
                errors.append(f"{kind}/{name} is missing spec.template")
            if not isinstance(pod_spec, dict) or not isinstance(pod_spec.get("containers"), list):
                errors.append(f"{kind}/{name} is missing spec.template.spec.containers")
        elif kind == "Service":
            if not isinstance(spec, dict) or not isinstance(spec.get("ports"), list):
                errors.append(f"Service/{name} is missing spec.ports")
        elif kind == "Ingress":
            if not isinstance(spec, dict) or not isinstance(spec.get("rules"), list):
                errors.append(f"Ingress/{name} is missing spec.rules")
        elif kind == "PodDisruptionBudget":
            if not isinstance(spec, dict) or not isinstance(spec.get("selector"), dict):
                errors.append(f"PodDisruptionBudget/{name} is missing spec.selector")
        elif kind == "NetworkPolicy":
            if not isinstance(spec, dict) or not isinstance(spec.get("policyTypes"), list):
                errors.append(f"NetworkPolicy/{name} is missing spec.policyTypes")
    by_kind_name: dict[tuple[str, str], dict[str, Any]] = {}
    for item in documents:
        key = (
            str(item.get("kind")),
            str((item.get("metadata") or {}).get("name")),
        )
        if key in by_kind_name:
            errors.append(f"duplicate production resource {key[0]}/{key[1]}")
            continue
        by_kind_name[key] = item

    exact_names_by_kind = {
        "Namespace": _REQUIRED_NAMESPACES,
        "ServiceAccount": _REQUIRED_SERVICE_ACCOUNTS,
        "Service": _REQUIRED_SERVICES,
        "Ingress": _REQUIRED_INGRESSES,
        "PodDisruptionBudget": _REQUIRED_PDBS,
        "NetworkPolicy": _REQUIRED_NETWORK_POLICIES,
    }
    for kind, expected_names in exact_names_by_kind.items():
        actual_names = {name for (actual_kind, name) in by_kind_name if actual_kind == kind}
        for name in sorted(expected_names - actual_names):
            errors.append(f"missing {kind}/{name}")
        for name in sorted(actual_names - expected_names):
            errors.append(f"unexpected {kind}/{name}")

    for (kind, name), item in by_kind_name.items():
        if kind == "ServiceAccount":
            errors.extend(_validate_service_account_contract(name, item))

    deployments = {
        name: item for (kind, name), item in by_kind_name.items() if kind == "Deployment"
    }
    release_suffix = _deployment_config_suffix(deployments.get("rag-api"))
    config_maps = {name for (kind, name) in by_kind_name if kind == "ConfigMap"}
    expected_config_maps = {
        _release_resource_name(base_name, release_suffix)
        for base_name in ("rag-runtime-config", "rag-frontend-config", "rag-canary-script")
    }
    for base_name in ("rag-runtime-config", "rag-frontend-config", "rag-canary-script"):
        expected_name = _release_resource_name(base_name, release_suffix)
        if expected_name not in config_maps:
            errors.append(f"missing versioned ConfigMap/{expected_name}")
        elif by_kind_name[("ConfigMap", expected_name)].get("immutable") is not True:
            errors.append(f"ConfigMap/{expected_name} must set immutable=true")
    for name in sorted(config_maps - expected_config_maps):
        errors.append(f"unexpected ConfigMap/{name}")
    for name in sorted(_REQUIRED_DEPLOYMENTS - set(deployments)):
        errors.append(f"missing Deployment/{name}")
    for name, deployment in deployments.items():
        if name not in _REQUIRED_DEPLOYMENTS:
            errors.append(f"unexpected Deployment/{name}")
            continue
        spec = deployment.get("spec") or {}
        if int(spec.get("replicas", 0)) < 2:
            errors.append(f"Deployment/{name} must have at least two replicas")
        strategy = spec.get("strategy") or {}
        if strategy.get("type") != "RollingUpdate":
            errors.append(f"Deployment/{name} must use RollingUpdate")
        elif str((strategy.get("rollingUpdate") or {}).get("maxUnavailable", "1")) not in {
            "0",
            "0%",
        }:
            errors.append(f"Deployment/{name} must set maxUnavailable=0")
        template = (spec.get("template") or {}).get("spec") or {}
        errors.extend(_validate_pod_security_contract("Deployment", name, template))
        errors.extend(_validate_model_cache_contract("Deployment", name, template))
        containers = template.get("containers") or []
        if not containers:
            errors.append(f"Deployment/{name} has no container")
            continue
        for container in containers + (template.get("initContainers") or []):
            image = str(container.get("image") or "")
            if not _DIGEST.search(image):
                errors.append(f"Deployment/{name} image is not digest-pinned")
            if not allow_placeholders and (_ZERO_DIGEST in image or "example.invalid" in image):
                errors.append(f"Deployment/{name} contains a placeholder image")
        for container in containers:
            container_name = str(container.get("name") or "<unnamed>")
            for probe_name in ("readinessProbe", "livenessProbe"):
                if probe_name not in container:
                    errors.append(
                        f"Deployment/{name} container {container_name} is missing {probe_name}"
                    )
                    continue
                errors.extend(
                    _validate_probe_contract(
                        "Deployment",
                        name,
                        container_name,
                        probe_name,
                        container[probe_name],
                    )
                )
        spread = template.get("topologySpreadConstraints") or []
        anti_affinity = (template.get("affinity") or {}).get("podAntiAffinity")
        if not spread and not anti_affinity:
            errors.append(f"Deployment/{name} is missing anti-affinity/topology spread")
        env_from = [item for container in containers for item in container.get("envFrom", [])]
        annotations = ((spec.get("template") or {}).get("metadata") or {}).get("annotations") or {}
        config_digest = str(annotations.get("industrial-rag/config-digest") or "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", config_digest, re.IGNORECASE):
            if (
                not allow_placeholders
                or config_digest != "sha256:REPLACE_WITH_RELEASE_CONFIG_DIGEST"
            ):
                errors.append(f"Deployment/{name} must carry the release config digest")
        deployment_suffix = _deployment_config_suffix(deployment)
        if deployment_suffix != release_suffix:
            errors.append(f"Deployment/{name} must use the shared release config digest")
        config_names = {(item.get("configMapRef") or {}).get("name") for item in env_from}
        secret_names = {(item.get("secretRef") or {}).get("name") for item in env_from}
        expected_config = _release_resource_name(
            "rag-frontend-config" if name == "rag-frontend" else "rag-runtime-config",
            release_suffix,
        )
        if expected_config not in config_names:
            errors.append(f"Deployment/{name} must consume {expected_config} ConfigMap")
        if name in {"rag-api", "rag-worker"}:
            expected_runtime = _release_resource_name("rag-runtime", release_suffix)
            if expected_runtime not in secret_names:
                errors.append(f"Deployment/{name} must consume {expected_runtime} Secret")
        if name == "rag-api" and _container_gpu_contract(template, "api") != ("1", "1"):
            errors.append(
                "Deployment/rag-api container api must request and limit one nvidia.com/gpu"
            )
        if name in {"rag-worker", "rag-frontend"}:
            container_name = "worker" if name == "rag-worker" else "frontend"
            if _container_gpu_contract(template, container_name) != ("", ""):
                errors.append(
                    f"Deployment/{name} container {container_name} must not request nvidia.com/gpu"
                )
        if name == "rag-frontend":
            mounts = {
                mount.get("mountPath")
                for container in containers
                for mount in container.get("volumeMounts", [])
            }
            if "/etc/nginx/conf.d" not in mounts:
                errors.append("Deployment/rag-frontend must mount writable /etc/nginx/conf.d")
            if "/usr/share/nginx/html/config.js" not in mounts:
                errors.append("Deployment/rag-frontend must mount runtime config.js")

    if ("Ingress", "rag") not in by_kind_name:
        errors.append("missing TLS Ingress/rag")
    else:
        ingress = by_kind_name[("Ingress", "rag")]
        if not (ingress.get("spec") or {}).get("tls"):
            errors.append("Ingress/rag must configure TLS")

    for name in sorted(_REQUIRED_PDBS):
        if ("PodDisruptionBudget", name) not in by_kind_name:
            errors.append(f"missing PodDisruptionBudget/{name}")
        else:
            pdb_spec = by_kind_name[("PodDisruptionBudget", name)].get("spec") or {}
            try:
                min_available = int(pdb_spec.get("minAvailable", 0))
            except (TypeError, ValueError):
                min_available = 0
            if min_available < 1:
                errors.append(f"PodDisruptionBudget/{name} must keep one pod available")
    jobs = {name: item for (kind, name), item in by_kind_name.items() if kind == "Job"}
    for name in sorted(_REQUIRED_JOBS - set(jobs)):
        errors.append(f"missing Job/{name}")
    for name, job in jobs.items():
        if name not in _REQUIRED_JOBS:
            errors.append(f"unexpected Job/{name}")
            continue
        pod_spec = ((job.get("spec") or {}).get("template") or {}).get("spec") or {}
        errors.extend(_validate_pod_security_contract("Job", name, pod_spec))
        errors.extend(_validate_model_cache_contract("Job", name, pod_spec))
        if pod_spec.get("restartPolicy") != "Never":
            errors.append(f"Job/{name} must set restartPolicy=Never")
        containers = pod_spec.get("containers") or []
        if not containers:
            errors.append(f"Job/{name} has no container")
            continue
        for container in containers + (pod_spec.get("initContainers") or []):
            image = str(container.get("image") or "")
            if not _DIGEST.search(image):
                errors.append(f"Job/{name} image is not digest-pinned")
            if not allow_placeholders and (_ZERO_DIGEST in image or "example.invalid" in image):
                errors.append(f"Job/{name} contains a placeholder image")
        secret_names = {
            (item.get("secretRef") or {}).get("name")
            for container in containers
            for item in container.get("envFrom", [])
        }
        config_names = {
            (item.get("configMapRef") or {}).get("name")
            for container in containers
            for item in container.get("envFrom", [])
        }
        expected_runtime_config = _release_resource_name("rag-runtime-config", release_suffix)
        if expected_runtime_config not in config_names:
            errors.append(f"Job/{name} must consume {expected_runtime_config} ConfigMap")
        if name == "rag-postgres-migration":
            expected_migration = _release_resource_name("rag-migration", release_suffix)
            if expected_migration not in secret_names:
                errors.append(
                    f"Job/rag-postgres-migration must consume {expected_migration} Secret"
                )
        if name == "rag-canary":
            expected_runtime = _release_resource_name("rag-runtime", release_suffix)
            if expected_runtime not in secret_names:
                errors.append(f"Job/rag-canary must consume {expected_runtime} Secret")
            expected_canary = _release_resource_name("rag-canary-credentials", release_suffix)
            if expected_canary not in secret_names:
                errors.append(f"Job/rag-canary must consume {expected_canary} Secret")
            expected_script = _release_resource_name("rag-canary-script", release_suffix)
            volume_config_maps = {
                (volume.get("configMap") or {}).get("name")
                for volume in pod_spec.get("volumes", [])
            }
            if expected_script not in volume_config_maps:
                errors.append(f"Job/rag-canary must mount {expected_script} ConfigMap")
            command_text = "\n".join(
                str(argument) for container in containers for argument in container.get("args", [])
            )
            if "/opt/industrial-rag-canary/run_release_canary.py" not in command_text:
                errors.append("Job/rag-canary must execute the functional canary script")
            by_container_name = {
                str(container.get("name") or ""): container for container in containers
            }
            smoke = by_container_name.get("smoke") or {}
            worker = by_container_name.get("candidate-worker") or {}
            smoke_command = "\n".join(str(item) for item in smoke.get("args", []))
            worker_command = "\n".join(str(item) for item in worker.get("args", []))
            queue_export = 'CELERY_TASK_DEFAULT_QUEUE="release-canary-${HOSTNAME}"'
            if queue_export not in smoke_command or queue_export not in worker_command:
                errors.append("Job/rag-canary API and worker must share a unique Pod queue")
            if '-Q "$CELERY_TASK_DEFAULT_QUEUE"' not in worker_command:
                errors.append("Job/rag-canary candidate worker must consume only the Pod queue")
            if _container_gpu_contract(pod_spec, "candidate-worker") != ("", ""):
                errors.append("Job/rag-canary candidate worker must use the production CPU profile")
            if not all(
                "canary-state"
                in {str(mount.get("name") or "") for mount in container.get("volumeMounts", [])}
                for container in (smoke, worker)
            ):
                errors.append("Job/rag-canary containers must share the completion marker volume")
            if int((job.get("spec") or {}).get("backoffLimit", -1)) != 0:
                errors.append("Job/rag-canary must fail closed without automatic retries")
            if int((job.get("spec") or {}).get("activeDeadlineSeconds", 0)) < 600:
                errors.append("Job/rag-canary must allow the functional ingestion flow to finish")
        if name == "rag-canary" and _container_gpu_contract(pod_spec, "smoke") != ("1", "1"):
            errors.append(
                "Job/rag-canary container smoke must request and limit one nvidia.com/gpu"
            )
        if name == "rag-postgres-migration" and _container_gpu_contract(pod_spec, "migration") != (
            "",
            "",
        ):
            errors.append("Job/rag-postgres-migration must not request nvidia.com/gpu")

    forbidden = {
        ("Service", "postgres"),
        ("Service", "redis"),
        ("Service", "keycloak"),
        ("Deployment", "postgres"),
        ("Deployment", "redis"),
        ("Deployment", "keycloak"),
    }
    for key in sorted(forbidden & set(by_kind_name)):
        errors.append(f"production manifest must not deploy {key[0]}/{key[1]}")
    if not allow_placeholders:
        serialized = path.read_text(encoding="utf-8")
        for marker in ("example.invalid", "REPLACE_WITH", "sha256:" + "0" * 64):
            if marker in serialized:
                errors.append(f"production manifest contains placeholder {marker}")
    if expected_images is not None:
        errors.extend(validate_expected_release_images(documents, expected_images))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-placeholders", action="store_true")
    parser.add_argument("--expected-api-image")
    parser.add_argument("--expected-worker-image")
    parser.add_argument("--expected-frontend-image")
    args = parser.parse_args()
    expected_values = {
        "api": args.expected_api_image,
        "worker": args.expected_worker_image,
        "frontend": args.expected_frontend_image,
    }
    supplied_expected = [bool(value) for value in expected_values.values()]
    if any(supplied_expected) and not all(supplied_expected):
        raise SystemExit("all three --expected-*-image values must be supplied together")
    errors = validate_manifest(
        args.manifest,
        allow_placeholders=args.allow_placeholders,
        expected_images=expected_values if all(supplied_expected) else None,
    )
    if errors:
        raise SystemExit("\n".join(errors))
    print("Kubernetes production manifest schema/policy contract passed.")


if __name__ == "__main__":
    main()
