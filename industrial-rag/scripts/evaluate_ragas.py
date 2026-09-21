"""Evaluate an Industrial RAG JSONL export with the protected text-only judge."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.native_judge import (  # noqa: E402
    EVALUATION_ENGINE,
    EVALUATION_ENGINE_VERSION,
    OpenAITextJudge,
    build_judge_request_policy,
)
from app.evaluation.ragas_adapter import (  # noqa: E402
    DEFAULT_METRICS,
    eligibility_summary,
    evaluate_samples,
    load_ragas_samples,
    validate_metric_names,
    write_report,
)
from app.utils.strict_dotenv import apply_release_env_file  # noqa: E402


def _load_project_env() -> None:
    configured_env_file = os.getenv("RAG_ENV_FILE", "").strip()
    env_file = Path(configured_env_file) if configured_env_file else PROJECT_ROOT / ".env"
    release_snapshot = os.getenv("RAG_RELEASE_SNAPSHOT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if release_snapshot:
        apply_release_env_file(env_file)
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_file, override=False)


def _resolve_model(args: argparse.Namespace) -> str:
    return str(
        args.model
        or os.getenv("RAGAS_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "deepseek-chat"
    ).strip()


def _normalize_openai_base_url(value: str) -> str:
    """Normalize the judge endpoint without importing runtime-only config deps."""
    raw = str(value or "").strip().rstrip("/")
    if not raw or raw.startswith("${"):
        return raw
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Judge base_url must be an absolute http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Judge base_url must not contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError("Judge base_url must not contain a query string or fragment")
    path = parsed.path.rstrip("/") or "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _resolve_base_url(args: argparse.Namespace) -> str:
    raw = str(
        args.base_url
        or os.getenv("RAGAS_BASE_URL")
        or os.getenv("DEEPSEEK_API_URL")
        or "https://api.deepseek.com/v1"
    ).strip()
    return _normalize_openai_base_url(raw)


def _build_llm(args: argparse.Namespace) -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install the hash-locked evaluation dependencies first: "
            "python -m pip install --require-hashes "
            "-r requirements-evaluation.lock.txt"
        ) from exc

    _load_project_env()
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key or api_key.startswith("${") or api_key == "sk-xxxxx":
        raise RuntimeError(
            f"Environment variable {args.api_key_env} is not configured; "
            "use --dry-run to validate the dataset without calling a judge model."
        )
    base_url = _resolve_base_url(args)
    model = _resolve_model(args)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout,
        max_retries=0,
    )
    return OpenAITextJudge(
        client=client,
        model=model,
        max_tokens=args.max_tokens,
        provider="openai_compatible",
        base_url=base_url,
    )


def _configure_evaluation_logging() -> None:
    """Keep dependency errors bounded and prevent upstream bodies entering logs."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return safe terminal output without questions, answers, or contexts."""
    summary: dict[str, Any] = {}
    for key in (
        "adapter",
        "evaluation_engine",
        "evaluation_engine_version",
        "judge_model",
        "metric_names",
        "dry_run",
        "status",
        "error",
        "sample_count",
        "eligible_cases",
        "metrics",
    ):
        if key in report:
            summary[key] = report[key]
    if isinstance(report.get("samples"), list):
        summary["evaluated_sample_records"] = len(report["samples"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="LRAGE/Industrial RAG JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ragas_eval/results.json"),
        help="JSON report path",
    )
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help=f"Comma-separated metrics (default: {','.join(DEFAULT_METRICS)})",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--base-url",
        default=None,
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum judge output tokens for structured metric responses",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent judge requests (default: 1 for rate-limit safety)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Bounded retries for transient judge failures (default: 1)",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Fallback delay before retrying a transient judge failure",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Metric checkpoint path (defaults to a sidecar next to --output)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed metrics from the checkpoint sidecar",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_project_env()
    names = validate_metric_names(args.metrics.split(","))
    samples = load_ragas_samples(args.input)
    if args.limit > 0:
        samples = samples[: args.limit]

    judge_model = _resolve_model(args)
    judge_base_url = _resolve_base_url(args)
    judge_request_policy = build_judge_request_policy(
        provider="openai_compatible",
        model_name=judge_model,
        base_url=judge_base_url,
    )
    metadata = {
        "adapter": "industrial-rag-native-text-judge-1",
        "evaluation_engine": EVALUATION_ENGINE,
        "evaluation_engine_version": EVALUATION_ENGINE_VERSION,
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "judge_model": judge_model,
        "judge_request_policy": judge_request_policy,
        "metric_names": list(names),
    }
    if args.dry_run:
        report = {
            **metadata,
            "dry_run": True,
            "sample_count": len(samples),
            "eligible_cases": eligibility_summary(samples, names),
        }
    else:
        # Legal questions and model outputs stay local unless the user explicitly
        # runs this command with a configured external judge endpoint.
        _configure_evaluation_logging()
        evaluation_context = {
            "evaluation_engine": EVALUATION_ENGINE,
            "evaluation_engine_version": EVALUATION_ENGINE_VERSION,
            "judge_model": metadata["judge_model"],
            "judge_base_url": judge_base_url,
            "judge_request_policy_sha256": judge_request_policy["sha256"],
            "max_tokens": max(256, min(int(args.max_tokens), 8192)),
            "temperature": 0,
        }
        checkpoint_path = args.checkpoint or args.output.with_name(
            f"{args.output.name}.partial.json"
        )
        try:
            report = {
                **metadata,
                "dry_run": False,
                **evaluate_samples(
                    samples,
                    llm=_build_llm(args),
                    metric_names=names,
                    max_concurrency=args.max_concurrency,
                    max_retries=args.max_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    checkpoint_path=checkpoint_path,
                    resume=args.resume,
                    evaluation_context=evaluation_context,
                ),
            }
        except RuntimeError as exc:
            # Persist a machine-readable failure so the release gate fails
            # closed while retaining only the sanitized error category.
            report = {
                **metadata,
                "dry_run": False,
                "status": "failed",
                "error": str(exc),
                "sample_count": len(samples),
                "metrics": {},
                "samples": [],
            }
            write_report(args.output, report)
            print(json.dumps(_report_summary(report), ensure_ascii=False, indent=2, allow_nan=False))
            raise SystemExit(1) from None
    write_report(args.output, report)
    if not args.dry_run:
        checkpoint_path = args.checkpoint or args.output.with_name(
            f"{args.output.name}.partial.json"
        )
        checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(_report_summary(report), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
