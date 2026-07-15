# Ragas evaluation

The project evaluates generated answers and retrieved legal contexts with
[Ragas](https://github.com/vibrantlabsai/ragas). It reuses the JSONL produced by
the existing LRAGE exporter, so the same RAG run can be compared with both
frameworks.

## Metrics

The default suite uses four Ragas 0.4 collection metrics:

- `answer_accuracy`: agreement between the generated and reference answers
- `faithfulness`: whether answer statements are supported by retrieved context
- `context_precision`: whether relevant contexts are ranked ahead of noise
- `context_recall`: whether contexts cover the claims in the reference answer

All four are LLM-judged. The last three require retrieved contexts, and every
metric except faithfulness requires a reference answer. Ineligible cases remain
in the report but are not included in that metric's mean.

## Install

Use a dedicated environment when possible. The versions are pinned because
`ragas==0.4.3` currently fails to import with `langchain-community==0.4.2`.

```powershell
pip install -r requirements-evaluation.txt
```

The equivalent project extra is available as `pip install -e ".[evaluation]"`
when the full application dependencies are also needed.

## Prepare the RAG run

Start the database/API dependencies, ingest the legal documents, then export
the included nine-case legal benchmark:

```powershell
python scripts/export_lrage.py --input eval/legal_core.jsonl --output-dir data/lrage_exports --task-name legal_core --top-k 5 --enable-rerank
```

Validate the mapping without making an LLM call:

```powershell
python scripts/evaluate_ragas.py --input data/lrage_exports/legal_core.jsonl --output data/ragas_eval/legal_core_dry_run.json --dry-run
```

## Run Ragas

The defaults use the project's `DEEPSEEK_API_KEY`, `deepseek-chat`, and
OpenAI-compatible DeepSeek endpoint. The script loads `.env` but never writes
the key to the report.

```powershell
python scripts/evaluate_ragas.py --input data/lrage_exports/legal_core.jsonl --output data/ragas_eval/legal_core.json
```

To evaluate a small smoke subset or select metrics:

```powershell
python scripts/evaluate_ragas.py --input data/lrage_exports/legal_core.jsonl --limit 2 --metrics answer_accuracy,faithfulness
```

For another OpenAI-compatible judge, set `RAGAS_MODEL`, `RAGAS_BASE_URL`, and
pass the environment variable holding its key with `--api-key-env`. Evaluation
questions, answers, and contexts are sent to that judge endpoint. Ragas
telemetry is disabled by the script.

The JSON report contains metric means, evaluated/eligible case counts, and
case-level scores. Continue to run deterministic citation Recall/MRR alongside
Ragas, because LLM-judged context metrics do not replace exact article checks:

```powershell
python scripts/score_retrieval.py data/lrage_exports/legal_core.jsonl --top-k 5
```

Run both metric families as one CI quality gate after producing the Ragas
report. The command exits non-zero when any default threshold is missed:

```powershell
python scripts/check_quality_gate.py --ragas-report data/ragas_eval/legal_core.json --export data/lrage_exports/legal_core.jsonl --top-k 5
```

Defaults are answer accuracy 0.70, faithfulness 0.80, context precision 0.60,
context recall 0.70, citation Recall@5 0.80, and citation MRR@5 0.70. Every
threshold has a corresponding `--min-*` override. Missing labels fail by
default; use `--allow-missing` only for exploratory datasets.
