"""Data-driven domain-specific recall query rules."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

DEFAULT_DOMAIN_SIGNAL_PATH = Path(__file__).with_name("domain_signal_queries.json")
DEFAULT_OUT_OF_SCOPE_SIGNAL_PATH = Path(__file__).with_name("out_of_scope_signals.json")


class DomainSignalRule(TypedDict):
    rule_id: str
    match_all: list[list[str]]
    queries: list[str]


def _validate_text(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Domain signal rule #{index} {field} must be a non-empty string")
    return value.strip()


def _validate_term_groups(value: Any, field: str, index: int) -> list[list[str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Domain signal rule #{index} {field} must be a non-empty list")

    groups: list[list[str]] = []
    for group_index, group in enumerate(value, 1):
        if not isinstance(group, list) or not group:
            raise ValueError(
                f"Domain signal rule #{index} {field}[{group_index}] must be a non-empty list"
            )
        terms = [_validate_text(term, f"{field}[{group_index}]", index) for term in group]
        groups.append(list(dict.fromkeys(terms)))
    return groups


def _validate_text_list(value: Any, field: str, index: int) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Domain signal rule #{index} {field} must be a non-empty list")
    return list(dict.fromkeys(_validate_text(item, field, index) for item in value))


def _validate_rule(rule: Any, index: int) -> DomainSignalRule:
    if not isinstance(rule, dict):
        raise ValueError(f"Domain signal rule #{index} must be an object")

    required = ("rule_id", "match_all", "queries")
    missing = [field for field in required if field not in rule]
    if missing:
        raise ValueError(f"Domain signal rule #{index} missing fields: {', '.join(missing)}")

    return {
        "rule_id": _validate_text(rule["rule_id"], "rule_id", index),
        "match_all": _validate_term_groups(rule["match_all"], "match_all", index),
        "queries": _validate_text_list(rule["queries"], "queries", index),
    }


@lru_cache(maxsize=4)
def load_domain_signal_rules(path: str | None = None) -> list[DomainSignalRule]:
    """Load domain recall rules from a UTF-8 JSON data file."""
    rule_path = Path(path) if path else DEFAULT_DOMAIN_SIGNAL_PATH
    with rule_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"{rule_path} must contain a JSON array")

    return [_validate_rule(rule, index) for index, rule in enumerate(data, 1)]


@lru_cache(maxsize=4)
def load_out_of_scope_signal_groups(path: str | None = None) -> list[list[list[str]]]:
    """Load auditable corpus-scope rules without embedding evaluation cases in code."""
    rule_path = Path(path) if path else DEFAULT_OUT_OF_SCOPE_SIGNAL_PATH
    with rule_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{rule_path} must contain a JSON array")

    groups: list[list[list[str]]] = []
    for index, rule in enumerate(data, 1):
        if not isinstance(rule, dict):
            raise ValueError(f"Out-of-scope rule #{index} must be an object")
        _validate_text(rule.get("rule_id"), "rule_id", index)
        groups.append(_validate_term_groups(rule.get("match_all"), "match_all", index))
    return groups


def _contains_all_groups(text: str, groups: list[list[str]]) -> bool:
    return all(any(term and term in text for term in group) for group in groups)


def is_known_out_of_scope(query: str) -> bool:
    """Return whether an auditable signal rule places a query outside the corpus."""
    normalized = query.strip()
    if not normalized:
        return False
    return any(
        _contains_all_groups(normalized, groups)
        for groups in load_out_of_scope_signal_groups()
    )


def build_domain_signal_queries(
    query: str,
    rewritten_query: str,
    signal_query: str,
    rules: list[DomainSignalRule] | None = None,
) -> list[str]:
    """Add statute-language recall terms without inventing article numbers."""
    combined = "\n".join(item for item in [query, rewritten_query, signal_query] if item)
    if not combined:
        return []

    queries: list[str] = []
    for rule in rules or load_domain_signal_rules():
        if not _contains_all_groups(combined, rule["match_all"]):
            continue
        for recall_query in rule["queries"]:
            recall_query = recall_query.strip()
            if recall_query and recall_query not in queries:
                queries.append(recall_query)
    return queries
