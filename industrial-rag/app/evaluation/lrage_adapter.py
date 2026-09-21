"""Adapter for exporting Industrial RAG runs into LRAGE-friendly files.

LRAGE is an external evaluation toolkit with its own dependency stack and
Python version constraints. This module intentionally does not import LRAGE;
it only writes JSONL/YAML artifacts that a separate LRAGE environment can load.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_RUBRIC = """Evaluate whether the answer is legally correct, grounded in the provided documents, complete enough for the question, and clear about unresolved facts. Give higher scores to answers that cite or rely on the retrieved legal provisions accurately and avoid unsupported claims."""


@dataclass(frozen=True)
class LrageExportPaths:
    """Paths produced by an LRAGE export."""

    dataset_jsonl: Path
    task_yaml: Path
    manifest_json: Path


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _trim_text(value: Any, max_chars: int | None) -> str:
    text = _as_text(value)
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _stable_case_id(query: str) -> str:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    return f"case-{digest}"


def context_text(doc: dict[str, Any]) -> str:
    """Return the text that best represents what the model saw."""
    metadata = doc.get("metadata") or {}
    return _as_text(
        doc.get("child_content")
        or metadata.get("child_content")
        or metadata.get("article_text")
        or doc.get("content")
        or ""
    )


def context_title(doc: dict[str, Any]) -> str:
    """Build a compact source label for a retrieved chunk."""
    metadata = doc.get("metadata") or {}
    parts = [
        metadata.get("filename"),
        metadata.get("legal_citation"),
        metadata.get("level_5_article"),
        doc.get("id"),
    ]
    return " | ".join(_as_text(part) for part in parts if part)


def normalize_context_doc(
    doc: dict[str, Any],
    rank: int,
    max_context_chars: int | None = 4000,
) -> dict[str, Any]:
    """Normalize one retrieved document for JSONL export."""
    metadata = doc.get("metadata") or {}
    text = _trim_text(context_text(doc), max_context_chars)
    return {
        "rank": rank,
        "id": doc.get("id"),
        "score": doc.get("score"),
        "rrf_score": doc.get("rrf_score"),
        "rerank_prob": (
            metadata["rerank_prob"]
            if metadata.get("rerank_prob") is not None
            else doc.get("rerank_prob")
        ),
        "source": metadata.get("filename"),
        "citation": metadata.get("legal_citation"),
        "article": metadata.get("level_5_article"),
        "semantic_chunk_id": metadata.get("semantic_chunk_id"),
        "title": context_title(doc),
        "text": text,
        "metadata": metadata,
    }


def build_document_block(contexts: list[dict[str, Any]], max_total_chars: int = 16000) -> str:
    """Format retrieved contexts for LRAGE's Document field."""
    blocks: list[str] = []
    total = 0

    for item in contexts:
        header = f"[{item['rank']}] {item.get('title') or item.get('id') or 'context'}"
        text = _as_text(item.get("text"))
        block = f"{header}\n{text}".strip()
        if not block:
            continue

        remaining = max_total_chars - total
        if max_total_chars > 0 and remaining <= 0:
            break
        if max_total_chars > 0 and len(block) > remaining:
            block = block[:remaining].rstrip() + "\n...[truncated]"

        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks)


def build_lrage_sample(
    *,
    query: str,
    answer: str,
    results: list[dict[str, Any]] | None = None,
    understanding: dict[str, Any] | None = None,
    sub_answers: list[dict[str, Any]] | None = None,
    case_id: str | None = None,
    expected_answer: str | None = None,
    rubric: str | None = None,
    metadata: dict[str, Any] | None = None,
    max_context_chars: int | None = 4000,
    max_document_chars: int = 16000,
) -> dict[str, Any]:
    """Build one JSONL sample compatible with LRAGE-style task templates."""
    contexts = [
        normalize_context_doc(doc, rank=index + 1, max_context_chars=max_context_chars)
        for index, doc in enumerate(results or [])
    ]
    document = build_document_block(contexts, max_total_chars=max_document_chars)
    target = expected_answer or ""
    sample_id = case_id or _stable_case_id(query)
    sample_rubric = rubric or DEFAULT_RUBRIC

    return {
        "id": sample_id,
        "query": query,
        "question": query,
        "Prompt": query,
        "Document": document,
        "Rubric": sample_rubric,
        "target": target,
        "expected_answer": target,
        "answer": answer,
        "prediction": answer,
        "contexts": contexts,
        "understanding": understanding or {},
        "sub_answers": sub_answers or [],
        "metadata": metadata or {},
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_lrage_task_yaml(
    *,
    task_name: str,
    dataset_jsonl: Path,
    max_score: int = 10,
) -> str:
    """Create a LRAGE task YAML for evaluating the exported dataset."""
    dataset_path = dataset_jsonl.resolve().as_posix()
    doc_to_text = (
        "{{Prompt}}\n\n"
        "<DOCUMENTS>{{Document}}</DOCUMENTS>\n\n"
        "请基于以上材料回答问题，并尽量说明依据。"
    )
    return "\n".join(
        [
            f"task: {task_name}",
            "dataset_path: json",
            "dataset_kwargs:",
            "  data_files:",
            f"    test: {_yaml_string(dataset_path)}",
            "output_type: generate_until",
            "test_split: test",
            f"doc_to_text: {_yaml_string(doc_to_text)}",
            "doc_to_target: \"{{target}}\"",
            "doc_to_rubric: Rubric",
            "generation_kwargs:",
            "  until:",
            "    - \"</s>\"",
            "judge_generation_kwargs:",
            "  until:",
            "    - \"</s>\"",
            "  max_gen_toks: 1024",
            "metric_list:",
            "  - metric: LLM-Eval",
            "    aggregation: mean",
            "    higher_is_better: true",
            f"    max_score: {max_score}",
            "metadata:",
            "  source: industrial-rag",
            "  adapter: lrage_adapter",
            "",
        ]
    )


def write_lrage_export(
    *,
    output_dir: Path,
    task_name: str,
    samples: list[dict[str, Any]],
    max_score: int = 10,
    evaluation_contract: dict[str, Any] | None = None,
) -> LrageExportPaths:
    """Write JSONL dataset, LRAGE task YAML, and export manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_jsonl = output_dir / f"{task_name}.jsonl"
    task_yaml = output_dir / f"{task_name}.yaml"
    manifest_json = output_dir / f"{task_name}.manifest.json"

    write_jsonl(dataset_jsonl, samples)
    task_yaml.write_text(
        build_lrage_task_yaml(
            task_name=task_name,
            dataset_jsonl=dataset_jsonl,
            max_score=max_score,
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "task_name": task_name,
        "sample_count": len(samples),
        "dataset_jsonl": str(dataset_jsonl.resolve()),
        "task_yaml": str(task_yaml.resolve()),
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_contract": evaluation_contract or {},
        "schema": {
            "lrage_fields": ["Prompt", "Document", "Rubric", "target"],
            "industrial_rag_fields": ["prediction", "contexts", "understanding", "sub_answers"],
        },
    }
    manifest_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    return LrageExportPaths(
        dataset_jsonl=dataset_jsonl,
        task_yaml=task_yaml,
        manifest_json=manifest_json,
    )


def load_cases_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL cases accepted by the export script."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must be a JSON object")
            if not record.get("query"):
                raise ValueError(f"Line {line_number} is missing required field 'query'")
            records.append(record)
    return records
