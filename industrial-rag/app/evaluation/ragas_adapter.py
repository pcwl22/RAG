"""Ragas 0.4 adapter for Industrial RAG evaluation exports.

The adapter keeps Ragas as an optional dependency. Dataset conversion and
validation therefore remain usable in the lightweight CI environment.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

DEFAULT_METRICS = (
    "answer_accuracy",
    "faithfulness",
    "context_precision",
    "context_recall",
)

_METRIC_REQUIREMENTS = {
    "answer_accuracy": ("user_input", "response", "reference"),
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "context_precision": ("user_input", "reference", "retrieved_contexts"),
    "context_recall": ("user_input", "reference", "retrieved_contexts"),
}


@dataclass(frozen=True)
class RagasSample:
    """Normalized single-turn sample plus its stable project case ID."""

    case_id: str
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str
    source_context_count: int

    def metric_input(self, fields: Iterable[str] | None = None) -> dict[str, Any]:
        values = {
            "user_input": self.user_input,
            "response": self.response,
            "retrieved_contexts": self.retrieved_contexts,
            "reference": self.reference,
        }
        return values if fields is None else {field: values[field] for field in fields}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _context_text(context: Any) -> str:
    if isinstance(context, str):
        return context.strip()
    if not isinstance(context, dict):
        return ""
    metadata = context.get("metadata") or {}
    for value in (
        context.get("text"),
        context.get("child_content"),
        metadata.get("child_content"),
        metadata.get("article_text"),
        context.get("content"),
    ):
        text = _text(value)
        if text:
            return text
    return ""


def build_ragas_sample(record: dict[str, Any], line_number: int | None = None) -> RagasSample:
    """Convert an Industrial RAG/LRAGE export row to Ragas fields."""
    location = f" on line {line_number}" if line_number is not None else ""
    user_input = _text(record.get("query") or record.get("question") or record.get("Prompt"))
    response = _text(record.get("prediction") or record.get("answer") or record.get("response"))
    if not user_input:
        raise ValueError(f"Evaluation sample{location} is missing a question/query")
    if not response:
        raise ValueError(f"Evaluation sample{location} is missing a generated response")

    raw_contexts = record.get("contexts") or record.get("retrieved_contexts") or []
    if not isinstance(raw_contexts, list):
        raise ValueError(f"Evaluation sample{location} contexts must be a list")
    contexts = [text for item in raw_contexts if (text := _context_text(item))]
    source_context_count = len(contexts)
    case_metadata = (record.get("metadata") or {}).get("case_metadata") or {}
    if case_metadata.get("category") == "comparison" and contexts:
        # Ragas 0.4 judges each context independently. A comparison needs both
        # cited provisions together, otherwise each useful half is scored zero.
        contexts = ["\n\n".join(contexts)]
    reference = _text(
        record.get("expected_answer")
        or record.get("reference_answer")
        or record.get("target")
        or record.get("reference")
    )
    case_id = _text(record.get("id")) or f"case-{line_number or 1}"
    return RagasSample(
        case_id=case_id,
        user_input=user_input,
        response=response,
        retrieved_contexts=contexts,
        reference=reference,
        source_context_count=source_context_count,
    )


def load_ragas_samples(path: Path) -> list[RagasSample]:
    """Load and validate JSONL produced by ``scripts/export_lrage.py``."""
    samples: list[RagasSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Evaluation sample on line {line_number} must be an object")
            samples.append(build_ragas_sample(record, line_number))
    if not samples:
        raise ValueError(f"No evaluation samples found in {path}")
    case_ids = [sample.case_id for sample in samples]
    duplicates = sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate evaluation sample IDs: {', '.join(duplicates)}")
    return samples


def validate_metric_names(metric_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(name.strip() for name in metric_names if name.strip()))
    unknown = sorted(set(names) - set(_METRIC_REQUIREMENTS))
    if unknown:
        raise ValueError(
            f"Unknown metrics: {', '.join(unknown)}. Supported: {', '.join(DEFAULT_METRICS)}"
        )
    if not names:
        raise ValueError("At least one metric is required")
    return names


def sample_is_eligible(sample: RagasSample, metric_name: str) -> bool:
    """Return whether a sample contains every field required by a metric."""
    required = _METRIC_REQUIREMENTS[metric_name]
    values = sample.metric_input()
    return all(bool(values[field]) for field in required)


def eligibility_summary(
    samples: Sequence[RagasSample], metric_names: Iterable[str]
) -> dict[str, int]:
    names = validate_metric_names(metric_names)
    return {
        name: sum(sample_is_eligible(sample, name) for sample in samples)
        for name in names
    }


def _metric_instances(llm: Any, metric_names: Sequence[str]) -> dict[str, Any]:
    try:
        from ragas.metrics.collections import (
            AnswerAccuracy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:  # pragma: no cover - exercised by CLI environments
        raise RuntimeError(
            "Ragas evaluation dependencies are missing or incompatible. "
            "Install the pinned 'evaluation' extra."
        ) from exc

    factories = {
        "answer_accuracy": AnswerAccuracy,
        "faithfulness": Faithfulness,
        "context_precision": ContextPrecisionWithReference,
        "context_recall": ContextRecall,
    }
    return {name: factories[name](llm) for name in metric_names}


def evaluate_samples(
    samples: Sequence[RagasSample],
    *,
    llm: Any,
    metric_names: Iterable[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """Run Ragas collection metrics and return aggregate plus case-level scores."""
    names = validate_metric_names(metric_names)
    metrics = _metric_instances(llm, names)
    case_results: dict[str, dict[str, Any]] = {
        sample.case_id: {
            "id": sample.case_id,
            "question": sample.user_input,
            "context_count": len(sample.retrieved_contexts),
            "source_context_count": sample.source_context_count,
            "has_reference": bool(sample.reference),
            "scores": {},
            "reasons": {},
        }
        for sample in samples
    }

    for name, metric in metrics.items():
        eligible = [sample for sample in samples if sample_is_eligible(sample, name)]
        if not eligible:
            continue
        required = _METRIC_REQUIREMENTS[name]
        results = metric.batch_score(
            [sample.metric_input(required) for sample in eligible]
        )
        if len(results) != len(eligible):
            raise RuntimeError(
                f"Ragas metric {name} returned {len(results)} results for {len(eligible)} samples"
            )
        for sample, result in zip(eligible, results, strict=True):
            value = result.value
            score = float(value) if isinstance(value, int | float) else value
            if isinstance(score, float) and math.isnan(score):
                score = None
            case_results[sample.case_id]["scores"][name] = score
            if result.reason:
                case_results[sample.case_id]["reasons"][name] = result.reason

    aggregates: dict[str, dict[str, float | int | None]] = {}
    for name in names:
        values = [
            item["scores"].get(name)
            for item in case_results.values()
            if isinstance(item["scores"].get(name), int | float)
            and not math.isnan(float(item["scores"][name]))
        ]
        aggregates[name] = {
            "mean": fmean(values) if values else None,
            "evaluated_cases": len(values),
            "eligible_cases": sum(sample_is_eligible(sample, name) for sample in samples),
        }

    return {
        "sample_count": len(samples),
        "metrics": aggregates,
        "samples": list(case_results.values()),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
