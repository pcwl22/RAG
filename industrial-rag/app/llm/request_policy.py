"""Fail-closed request policies for structured OpenAI-compatible output."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import urlsplit

STRUCTURED_OUTPUT_POLICY_SCHEMA_VERSION = 1
_OFFICIAL_DEEPSEEK_HOST = "api.deepseek.com"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_official_deepseek_endpoint(base_url: str | None) -> bool:
    raw = str(base_url or "").strip()
    if not raw or raw.startswith("${"):
        return False
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme.lower() == "https"
        and str(parsed.hostname or "").strip().lower() == _OFFICIAL_DEEPSEEK_HOST
        and parsed.username is None
        and parsed.password is None
        and port in {None, 443}
        and not parsed.query
        and not parsed.fragment
    )


def build_structured_output_policy(
    *,
    provider: str,
    model_name: str,
    base_url: str | None,
) -> dict[str, Any]:
    """Describe the non-secret request policy used for strict JSON output.

    DeepSeek Flash enables thinking by default. With a bounded ``max_tokens``
    value, reasoning can consume the entire allowance and leave the final
    ``content`` empty. Official DeepSeek endpoints therefore use explicit
    non-thinking JSON mode for contract-bound application answers and judge
    verdicts. Other OpenAI-compatible gateways retain prompt-only behavior so
    vendor-specific parameters are never sent to an unknown proxy.
    """

    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model_name or "").strip()
    official_deepseek = (
        normalized_provider in {"openai_compatible", "deepseek"}
        and _is_official_deepseek_endpoint(base_url)
    )
    snapshot = {
        "schema_version": STRUCTURED_OUTPUT_POLICY_SCHEMA_VERSION,
        "provider": normalized_provider,
        "model_name": normalized_model,
        "mode": (
            "deepseek_non_thinking_json"
            if official_deepseek
            else "prompt_only"
        ),
        "thinking": "disabled" if official_deepseek else "provider_default",
        "reasoning_effort": "none" if official_deepseek else "provider_default",
        "response_format": "json_object" if official_deepseek else "prompt_only",
    }
    return {"sha256": _canonical_sha256(snapshot), "snapshot": snapshot}


def normalize_structured_output_policy(value: Any) -> dict[str, Any] | None:
    """Validate a self-hashing structured-output policy without extra fields."""

    if not isinstance(value, dict) or set(value) != {"sha256", "snapshot"}:
        return None
    digest = str(value.get("sha256") or "").strip().lower()
    snapshot = value.get("snapshot")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "schema_version",
        "provider",
        "model_name",
        "mode",
        "thinking",
        "reasoning_effort",
        "response_format",
    }:
        return None
    if snapshot.get("schema_version") != STRUCTURED_OUTPUT_POLICY_SCHEMA_VERSION:
        return None
    if not str(snapshot.get("provider") or "").strip():
        return None
    if not str(snapshot.get("model_name") or "").strip():
        return None
    mode = snapshot.get("mode")
    if mode not in {"deepseek_non_thinking_json", "prompt_only"}:
        return None
    expected_modes = {
        "deepseek_non_thinking_json": ("disabled", "none", "json_object"),
        "prompt_only": ("provider_default", "provider_default", "prompt_only"),
    }
    actual_mode = (
        snapshot.get("thinking"),
        snapshot.get("reasoning_effort"),
        snapshot.get("response_format"),
    )
    if actual_mode != expected_modes[mode]:
        return None
    if mode == "deepseek_non_thinking_json" and snapshot.get("provider") not in {
        "openai_compatible",
        "deepseek",
    }:
        return None
    if _canonical_sha256(snapshot) != digest:
        return None
    return {"sha256": digest, "snapshot": snapshot}


def structured_output_request_options(
    *,
    provider: str,
    model_name: str,
    base_url: str | None,
) -> dict[str, Any]:
    """Return vendor options only when the endpoint is known to support them."""

    policy = build_structured_output_policy(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
    )
    if policy["snapshot"]["mode"] != "deepseek_non_thinking_json":
        return {}
    return {
        "reasoning_effort": "none",
        "extra_body": {"thinking": {"type": "disabled"}},
        "response_format": {"type": "json_object"},
    }
