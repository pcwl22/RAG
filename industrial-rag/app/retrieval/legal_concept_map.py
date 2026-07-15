"""Controlled legal concept to statute-article recall mappings.

Mappings are retrieval hints only. They help convert common legal concepts in
user questions into statute language and known article ids, but final answers
must still cite only documents actually retrieved from the knowledge base.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

DEFAULT_MAPPING_PATH = Path(__file__).with_name("legal_concept_mappings.json")


class LegalConceptMapping(TypedDict):
    concept_id: str
    concept: str
    match_all: list[list[str]]
    articles: list[dict[str, str]]
    recall_terms: list[str]


def _validate_text(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Legal concept mapping #{index} {field} must be a non-empty string")
    return value.strip()


def _validate_term_groups(value: Any, field: str, index: int) -> list[list[str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Legal concept mapping #{index} {field} must be a non-empty list")

    groups: list[list[str]] = []
    for group_index, group in enumerate(value, 1):
        if not isinstance(group, list) or not group:
            raise ValueError(
                f"Legal concept mapping #{index} {field}[{group_index}] must be a non-empty list"
            )
        terms = [_validate_text(term, f"{field}[{group_index}]", index) for term in group]
        groups.append(list(dict.fromkeys(terms)))
    return groups


def _validate_text_list(value: Any, field: str, index: int) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Legal concept mapping #{index} {field} must be a non-empty list")
    return list(dict.fromkeys(_validate_text(item, field, index) for item in value))


def _validate_articles(value: Any, index: int) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Legal concept mapping #{index} articles must be a non-empty list")

    required = ("law", "article", "semantic_chunk_id")
    articles: list[dict[str, str]] = []
    for article_index, article in enumerate(value, 1):
        if not isinstance(article, dict):
            raise ValueError(
                f"Legal concept mapping #{index} article #{article_index} must be an object"
            )
        missing = [field for field in required if field not in article]
        if missing:
            raise ValueError(
                f"Legal concept mapping #{index} article #{article_index} missing fields: "
                f"{', '.join(missing)}"
            )
        normalized = {
            key: _validate_text(value, f"articles[{article_index}].{key}", index)
            for key, value in article.items()
        }
        articles.append(normalized)
    return articles


def _validate_mapping(mapping: Any, index: int) -> LegalConceptMapping:
    if not isinstance(mapping, dict):
        raise ValueError(f"Legal concept mapping #{index} must be an object")

    required = ("concept_id", "concept", "match_all", "articles", "recall_terms")
    missing = [field for field in required if field not in mapping]
    if missing:
        raise ValueError(f"Legal concept mapping #{index} missing fields: {', '.join(missing)}")

    return {
        "concept_id": _validate_text(mapping["concept_id"], "concept_id", index),
        "concept": _validate_text(mapping["concept"], "concept", index),
        "match_all": _validate_term_groups(mapping["match_all"], "match_all", index),
        "articles": _validate_articles(mapping["articles"], index),
        "recall_terms": _validate_text_list(mapping["recall_terms"], "recall_terms", index),
    }


@lru_cache(maxsize=4)
def load_legal_concept_article_mappings(path: str | None = None) -> list[LegalConceptMapping]:
    """Load concept-article mappings from a UTF-8 JSON data file."""
    mapping_path = Path(path) if path else DEFAULT_MAPPING_PATH
    with mapping_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"{mapping_path} must contain a JSON array")

    return [_validate_mapping(mapping, index) for index, mapping in enumerate(data, 1)]


LEGAL_CONCEPT_ARTICLE_MAPPINGS = load_legal_concept_article_mappings()


def _normalize_values(values: Any) -> list[str]:
    if isinstance(values, dict):
        flattened: list[str] = []
        for item in values.values():
            flattened.extend(_normalize_values(item))
        return flattened
    if isinstance(values, list):
        return [text for value in values for text in _normalize_values(value)]
    text = str(values or "").strip()
    return [text] if text else []


def _combined_text(*parts: Any) -> str:
    return "\n".join(text for part in parts for text in _normalize_values(part))


def _match_terms(text: str, groups: list[list[str]]) -> tuple[bool, list[str]]:
    matched: list[str] = []
    for group in groups:
        group_matches = [term for term in group if term and term in text]
        if not group_matches:
            return False, []
        matched.extend(group_matches)
    return True, list(dict.fromkeys(matched))


def _build_recall_query(mapping: dict[str, Any]) -> str:
    terms: list[str] = [mapping["concept"]]
    for article in mapping.get("articles", []):
        terms.extend(
            [
                article.get("law", ""),
                article.get("article", ""),
                article.get("semantic_chunk_id", ""),
            ]
        )
    terms.extend(mapping.get("recall_terms", []))
    return " ".join(term for term in dict.fromkeys(terms) if term)


def match_legal_concept_articles(
    query: str,
    rewritten_query: str = "",
    retrieval_signals: dict | None = None,
    mappings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return controlled concept-article mappings matched by a query."""
    text = _combined_text(query, rewritten_query, retrieval_signals or {})
    matches: list[dict[str, Any]] = []
    for mapping in mappings or LEGAL_CONCEPT_ARTICLE_MAPPINGS:
        ok, matched_terms = _match_terms(text, mapping.get("match_all", []))
        if not ok:
            continue
        matches.append(
            {
                "concept_id": mapping["concept_id"],
                "concept": mapping["concept"],
                "matched_terms": matched_terms,
                "articles": mapping.get("articles", []),
                "recall_query": _build_recall_query(mapping),
                "note": "映射仅用于提升召回，最终依据仍以实际检索到的上下文为准。",
            }
        )
    return matches


def build_concept_article_queries(mappings: list[dict[str, Any]]) -> list[str]:
    """Build retrieval queries from matched concept-article mappings."""
    queries: list[str] = []
    for mapping in mappings:
        query = str(mapping.get("recall_query") or "").strip()
        if query and query not in queries:
            queries.append(query)
    return queries
