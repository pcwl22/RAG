"""Evaluate an Industrial RAG JSONL export with Ragas 0.4 metrics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.ragas_adapter import (  # noqa: E402
    DEFAULT_METRICS,
    eligibility_summary,
    evaluate_samples,
    load_ragas_samples,
    validate_metric_names,
    write_report,
)


def _package_version() -> str | None:
    try:
        return version("ragas")
    except PackageNotFoundError:
        return None


def _build_llm(args: argparse.Namespace) -> Any:
    try:
        from dotenv import load_dotenv
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError(
            'Install evaluation dependencies first: pip install -e ".[evaluation]"'
        ) from exc

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key or api_key.startswith("${") or api_key == "sk-xxxxx":
        raise RuntimeError(
            f"Environment variable {args.api_key_env} is not configured; "
            "use --dry-run to validate the dataset without calling a judge model."
        )
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    return llm_factory(
        args.model,
        provider="openai",
        client=client,
        temperature=0,
        max_tokens=args.max_tokens,
    )


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
    parser.add_argument("--model", default=os.getenv("RAGAS_MODEL", "deepseek-chat"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAGAS_BASE_URL", "https://api.deepseek.com/v1"),
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum judge output tokens for structured Ragas responses",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = validate_metric_names(args.metrics.split(","))
    samples = load_ragas_samples(args.input)
    if args.limit > 0:
        samples = samples[: args.limit]

    metadata = {
        "adapter": "industrial-rag-ragas-0.4",
        "ragas_version": _package_version(),
        "input": str(args.input.resolve()),
        "judge_model": args.model,
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
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        report = {
            **metadata,
            "dry_run": False,
            **evaluate_samples(samples, llm=_build_llm(args), metric_names=names),
        }
    write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
