"""List ConfigMap and Secret references used by Kubernetes workload pod templates.

Release rollback uses this list to prove that every dependency of the previous
Deployment revision still exists before changing any stable workload.  Only
DNS-safe resource identifiers are emitted, one ``kind/name`` reference per line.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

_DNS_SUBDOMAIN = re.compile(
    r"[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?"
)


def _add_reference(refs: set[tuple[str, str]], kind: str, name: Any) -> None:
    value = str(name or "").strip()
    if not value:
        return
    if not _DNS_SUBDOMAIN.fullmatch(value):
        raise ValueError(f"unsafe Kubernetes resource reference {value!r}")
    refs.add((kind, value))


def workload_resource_refs(document: dict[str, Any]) -> list[str]:
    """Return all ConfigMap/Secret dependencies of one workload pod template."""
    kind = str(document.get("kind") or "")
    if kind not in {"Deployment", "StatefulSet", "DaemonSet", "Job"}:
        raise ValueError(f"unsupported workload kind {kind!r}")
    pod_spec = (
        ((document.get("spec") or {}).get("template") or {}).get("spec") or {}
    )
    refs: set[tuple[str, str]] = set()
    containers = list(pod_spec.get("initContainers") or []) + list(
        pod_spec.get("containers") or []
    )
    for container in containers:
        for source in container.get("envFrom") or []:
            _add_reference(refs, "configmap", (source.get("configMapRef") or {}).get("name"))
            _add_reference(refs, "secret", (source.get("secretRef") or {}).get("name"))
        for item in container.get("env") or []:
            value_from = item.get("valueFrom") or {}
            _add_reference(
                refs,
                "configmap",
                (value_from.get("configMapKeyRef") or {}).get("name"),
            )
            _add_reference(
                refs,
                "secret",
                (value_from.get("secretKeyRef") or {}).get("name"),
            )
    for volume in pod_spec.get("volumes") or []:
        _add_reference(refs, "configmap", (volume.get("configMap") or {}).get("name"))
        _add_reference(refs, "secret", (volume.get("secret") or {}).get("secretName"))
        for source in (volume.get("projected") or {}).get("sources") or []:
            _add_reference(refs, "configmap", (source.get("configMap") or {}).get("name"))
            _add_reference(refs, "secret", (source.get("secret") or {}).get("name"))
    for pull_secret in pod_spec.get("imagePullSecrets") or []:
        _add_reference(refs, "secret", pull_secret.get("name"))
    return [f"{kind}/{name}" for kind, name in sorted(refs)]


def load_one_workload(path: Path) -> dict[str, Any]:
    documents = [
        item
        for item in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if isinstance(item, dict) and item.get("kind")
    ]
    if len(documents) != 1:
        raise ValueError("manifest must contain exactly one Kubernetes workload")
    return documents[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        references = workload_resource_refs(load_one_workload(args.manifest))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("\n".join(references))


if __name__ == "__main__":
    main()
