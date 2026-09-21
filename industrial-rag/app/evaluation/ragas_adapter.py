"""Stable evaluation adapter for Industrial RAG/LRAGE exports.

The historical module name is retained for artifact compatibility. Metric
execution uses the repository's text-only native judge and has no Ragas,
LangChain, DiskCache, URL-loader, or multimodal dependency.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
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

_MAX_EVALUATION_CONCURRENCY = 8
_MAX_EVALUATION_RETRIES = 3
_MAX_RETRY_DELAY_SECONDS = 120.0
_CHECKPOINT_CONTEXT_FIELDS = frozenset(
    {
        "evaluation_engine",
        "evaluation_engine_version",
        "judge_model",
        "judge_base_url",
        "judge_request_policy_sha256",
        "max_tokens",
        "temperature",
    }
)
_SYSTEM_EVIDENCE_HEADING = re.compile(
    r"(?:\r?\n){2,}[ \t]*(?:"
    r"依据[:：]|"
    r"(?:##[ \t]*)?【匹配文件】|"
    r"(?:##[ \t]*)?【文件具体位置（文件章节页码）】|"
    r"(?:##[ \t]*)?【文件具体位置和内容（文件章节页码）】|"
    r"(?:##[ \t]*)?【文件具体位置和内容】"
    r")[ \t]*(?:\r?\n|$)"
)


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


def _ragas_response_text(value: Any) -> str:
    """Keep generated conclusions while excluding system-added evidence metadata."""
    response = _text(value)
    if not response:
        return ""
    response = re.sub(r"^(?:##\s*)?【回答】\s*", "", response, count=1)
    response = re.sub(r"^结论[:：]\s*", "", response, count=1).strip()
    response = _SYSTEM_EVIDENCE_HEADING.split(response, maxsplit=1)[0]
    return response.strip()


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
    response = _ragas_response_text(
        record.get("prediction") or record.get("answer") or record.get("response")
    )
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
    from app.evaluation.native_judge import metric_instances

    return metric_instances(llm, list(metric_names))


def _evaluation_fingerprint(
    samples: Sequence[RagasSample],
    metric_names: Sequence[str],
    evaluation_context: dict[str, Any],
) -> str:
    payload = {
        "evaluation_context": evaluation_context,
        "metric_names": list(metric_names),
        "samples": [
            {
                "id": sample.case_id,
                "user_input": sample.user_input,
                "response": sample.response,
                "retrieved_contexts": sample.retrieved_contexts,
                "reference": sample.reference,
            }
            for sample in samples
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_metric_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    metric_names: Sequence[str],
    evaluation_context: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Ragas checkpoint: {path}") from exc
    if (
        payload.get("schema_version") != 2
        or payload.get("fingerprint") != fingerprint
        or payload.get("evaluation_context") != evaluation_context
    ):
        raise ValueError(
            "Evaluation checkpoint does not match the current samples, metrics, "
            "and judge contract"
        )
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Ragas checkpoint metrics must be an object")
    unknown = sorted(set(metrics) - set(metric_names))
    if unknown:
        raise ValueError(f"Ragas checkpoint contains unexpected metrics: {', '.join(unknown)}")
    return {
        str(name): dict(results)
        for name, results in metrics.items()
        if isinstance(results, dict)
    }


def _write_metric_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    evaluation_context: dict[str, Any],
    metrics: dict[str, dict[str, dict[str, Any]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "fingerprint": fingerprint,
        "evaluation_context": evaluation_context,
        "metrics": metrics,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    try:
        temporary.replace(path)
    except PermissionError:
        # Managed workspaces can deny replace while allowing ordinary writes.
        path.write_text(encoded, encoding="utf-8", newline="\n")
        temporary.unlink(missing_ok=True)


def _bounded_int(value: int, *, minimum: int, maximum: int, name: str) -> int:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _validated_evaluation_context(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("evaluation_context is required when checkpointing")
    missing = sorted(_CHECKPOINT_CONTEXT_FIELDS - set(value))
    if missing:
        raise ValueError(
            "evaluation_context is missing checkpoint identity fields: " + ", ".join(missing)
        )
    for key in (
        "evaluation_engine",
        "evaluation_engine_version",
        "judge_model",
        "judge_base_url",
    ):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"evaluation_context.{key} must be a non-empty string")
    if type(value.get("max_tokens")) is not int or value["max_tokens"] < 1:
        raise ValueError("evaluation_context.max_tokens must be a positive integer")
    temperature = value.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("evaluation_context.temperature must be a finite number")
    if not math.isfinite(float(temperature)):
        raise ValueError("evaluation_context.temperature must be a finite number")
    try:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation_context must be JSON serializable") from exc
    normalized = json.loads(serialized)
    if not isinstance(normalized, dict):
        raise ValueError("evaluation_context must be a JSON object")
    return normalized


def _error_status_code(exc: BaseException) -> int | None:
    for candidate in (exc, getattr(exc, "last_attempt", None), getattr(exc, "__cause__", None)):
        status = getattr(candidate, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(candidate, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
    return None


def _is_retryable_evaluation_error(exc: BaseException) -> bool:
    status = _error_status_code(exc)
    if status in {408, 409, 429} or (status is not None and status >= 500):
        return True

    class_name = exc.__class__.__name__
    if class_name in {
        "APIConnectionError",
        "APITimeoutError",
        "TimeoutException",
        "JudgeOutputError",
    }:
        return True

    message = str(exc).lower()
    if "function" in message and (
        "not found" in message or "does not exist" in message or "not exist" in message
    ):
        # A known compatibility-gateway transient: it can reject a single
        # structured request while its internal function registry refreshes.
        return True
    return any(
        marker in message
        for marker in (
            "rate limit",
            "rate-limit",
            "rate_limited",
            "temporarily unavailable",
            "connection error",
            "connection reset",
            "timed out",
            "timeout",
            "retryable",
        )
    )


def _retry_after_seconds(exc: BaseException, fallback: float) -> float:
    """Read a numeric Retry-After value without exposing response bodies."""
    candidates = [exc, getattr(exc, "last_attempt", None), getattr(exc, "__cause__", None)]
    for candidate in candidates:
        response = getattr(candidate, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            raw = headers.get("retry-after") or headers.get("Retry-After")
            try:
                if raw is not None:
                    return min(max(float(raw), 1.0), _MAX_RETRY_DELAY_SECONDS)
            except (TypeError, ValueError):
                pass

    # Some compatibility gateways expose this only in a structured exception
    # representation. Parse the number locally; never log the representation.
    match = re.search(r"retry[_ -]?after['\" ]*[:=]\s*([0-9]+(?:\.[0-9]+)?)", str(exc), re.I)
    if match:
        return min(max(float(match.group(1)), 1.0), _MAX_RETRY_DELAY_SECONDS)
    return min(max(float(fallback), 0.1), _MAX_RETRY_DELAY_SECONDS)


def _error_category(exc: BaseException) -> str:
    status = _error_status_code(exc)
    if status is not None:
        return f"http_{status}"
    class_name = exc.__class__.__name__
    if class_name in {"APIConnectionError", "APITimeoutError", "TimeoutException"}:
        return "transport"
    if _is_retryable_evaluation_error(exc):
        return "transient"
    return "non_retryable"


async def _score_metric_inputs_async(
    metric: Any,
    inputs: Sequence[dict[str, Any]],
    *,
    max_concurrency: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[Any]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def score_one(payload: dict[str, Any]) -> Any:
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    return await metric.ascore(**payload)
            except Exception as exc:
                if attempt >= max_retries or not _is_retryable_evaluation_error(exc):
                    raise
                delay = _retry_after_seconds(exc, retry_backoff_seconds) * (2**attempt)
                await asyncio.sleep(min(delay, _MAX_RETRY_DELAY_SECONDS))
        raise RuntimeError("unreachable")

    return list(await asyncio.gather(*(score_one(payload) for payload in inputs)))


def _score_metric_inputs(
    metric: Any,
    inputs: Sequence[dict[str, Any]],
    *,
    max_concurrency: int,
    max_retries: int,
    retry_backoff_seconds: float,
    runner: asyncio.Runner | None = None,
) -> list[Any]:
    """Score with bounded async concurrency, retaining a test/fallback adapter."""
    if hasattr(metric, "ascore"):
        coroutine = _score_metric_inputs_async(
            metric,
            inputs,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        if runner is not None:
            return runner.run(coroutine)
        return asyncio.run(coroutine)
    # Older/fake metrics expose only batch_score. This path remains compatible,
    # while real Ragas 0.4 collection metrics use the bounded ascore path above.
    return list(metric.batch_score(list(inputs)))


def evaluate_samples(
    samples: Sequence[RagasSample],
    *,
    llm: Any,
    metric_names: Iterable[str] = DEFAULT_METRICS,
    max_concurrency: int = 1,
    max_retries: int = 1,
    retry_backoff_seconds: float = 2.0,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    evaluation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run protected text metrics and return aggregate plus case-level scores."""
    names = validate_metric_names(metric_names)
    max_concurrency = _bounded_int(
        max_concurrency,
        minimum=1,
        maximum=_MAX_EVALUATION_CONCURRENCY,
        name="max_concurrency",
    )
    max_retries = _bounded_int(
        max_retries,
        minimum=0,
        maximum=_MAX_EVALUATION_RETRIES,
        name="max_retries",
    )
    if retry_backoff_seconds < 0 or retry_backoff_seconds > _MAX_RETRY_DELAY_SECONDS:
        raise ValueError(
            f"retry_backoff_seconds must be between 0 and {_MAX_RETRY_DELAY_SECONDS}"
        )
    if resume and checkpoint_path is None:
        raise ValueError("checkpoint_path is required when resume=True")
    checkpoint_context = (
        _validated_evaluation_context(evaluation_context)
        if checkpoint_path is not None
        else dict(evaluation_context or {})
    )
    fingerprint = _evaluation_fingerprint(samples, names, checkpoint_context)
    completed_metrics = (
        _load_metric_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            metric_names=names,
            evaluation_context=checkpoint_context,
        )
        if resume and checkpoint_path is not None and checkpoint_path.is_file()
        else {}
    )
    if checkpoint_path is not None and not resume:
        _write_metric_checkpoint(
            path=checkpoint_path,
            fingerprint=fingerprint,
            evaluation_context=checkpoint_context,
            metrics={},
        )
    metrics = _metric_instances(llm, names)
    # AsyncOpenAI owns an async connection pool. Keep every judge metric on
    # one event loop instead of creating a new loop around each metric.
    async_runner = (
        asyncio.Runner()
        if any(hasattr(metric, "ascore") for metric in metrics.values())
        else None
    )
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
        completed = completed_metrics.get(name) or {}
        eligible_ids = {sample.case_id for sample in eligible}
        if set(completed) == eligible_ids:
            for sample in eligible:
                saved = completed[sample.case_id]
                case_results[sample.case_id]["scores"][name] = saved.get("value")
                if saved.get("reason"):
                    case_results[sample.case_id]["reasons"][name] = saved["reason"]
            continue
        required = _METRIC_REQUIREMENTS[name]
        try:
            results = _score_metric_inputs(
                metric,
                [sample.metric_input(required) for sample in eligible],
                max_concurrency=max_concurrency,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                runner=async_runner,
            )
        except Exception as exc:
            if async_runner is not None:
                async_runner.close()
                async_runner = None
            raise RuntimeError(
                f"Evaluation metric {name} failed with {_error_category(exc)}"
            ) from None
        if len(results) != len(eligible):
            if async_runner is not None:
                async_runner.close()
                async_runner = None
            raise RuntimeError(
                f"Evaluation metric {name} returned {len(results)} results for "
                f"{len(eligible)} samples"
            )
        metric_results: dict[str, dict[str, Any]] = {}
        for sample, result in zip(eligible, results, strict=True):
            value = result.value
            score = float(value) if isinstance(value, int | float) else value
            if isinstance(score, float) and math.isnan(score):
                score = None
            case_results[sample.case_id]["scores"][name] = score
            if result.reason:
                case_results[sample.case_id]["reasons"][name] = result.reason
            metric_results[sample.case_id] = {
                "value": score,
                "reason": result.reason,
            }
        if checkpoint_path is not None:
            completed_metrics[name] = metric_results
            _write_metric_checkpoint(
                path=checkpoint_path,
                fingerprint=fingerprint,
                evaluation_context=checkpoint_context,
                metrics=completed_metrics,
            )

    if async_runner is not None:
        async_runner.close()

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
