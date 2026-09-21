# Native text-judge evaluation

The protected release evaluation uses the project's dependency-minimal native
text judge. The filename `evaluate_ragas.py`, the `--ragas-report` options, and
the `ragas-judge` workflow job are retained as compatibility identifiers; the
runtime does not import or install Ragas, LangChain, DiskCache, or Instructor.

The judge accepts only JSON-serialized question, answer, reference, and
retrieved-context text. It has no tool/function-call argument, URL or file
loader, multimodal input, or persistent cache. Judge output must be a bounded
JSON object that satisfies each metric contract, otherwise evaluation fails
closed.

## Metrics

The four protected metrics are:

- `answer_accuracy`: average of two 0/2/4 agreement ratings, normalized to 0–1
- `faithfulness`: supported candidate claims divided by all candidate claims
- `context_precision`: rank-aware average precision of relevant contexts
- `context_recall`: reference claims attributable to context divided by all reference claims

All four are LLM-judged. Ineligible cases remain visible in the report but are
not included in that metric's mean. Deterministic citation Recall/MRR remains a
separate mandatory gate because an LLM score cannot replace exact article
matching.

## Install the isolated judge

Generate the lock with pip-tools 7.5.3 on Python 3.12, then install it with hash
verification:

```powershell
pip install --require-hashes -r requirements-evaluation.lock.txt
```

The direct dependency surface is exactly `openai==3.6.0`. CI audits this lock
and every production lock without vulnerability ignores.

## Prepare and validate a run

Start the database dependencies, ingest the legal corpus, and export the
benchmark through the production retrieval path:

```powershell
python scripts/export_lrage.py --input eval/legal_core.jsonl --output-dir data/lrage_exports --task-name legal_core --top-k 5 --enable-rerank
```

The export writes an atomic `<task-name>.partial.json` checkpoint. Its v2
fingerprint binds the source dataset SHA-256, effective generation LLM
provider/model/endpoint hash, sample count, and every output-affecting export
option. Older JSONL checkpoints and checkpoints from a different runtime are
rejected instead of being merged into release evidence. Each exported sample
and the export manifest carry the same non-secret contract.

Validate mapping and eligibility without making an external judge call:

```powershell
python scripts/evaluate_ragas.py --input data/lrage_exports/legal_core.jsonl --output data/ragas_eval/legal_core_dry_run.json --dry-run
```

Run the deterministic retrieval suites through the production query-understanding
and decomposition path. `--resume` writes an atomic `<report>.partial.json`
checkpoint after every completed case. The checkpoint is accepted only when the
input SHA-256, sample count, metric-affecting options, LLM provider/model, and a
non-secret endpoint hash match. It is removed only after the complete report is
written. Release runs also set the minimum citation recall so the evaluator stops
as soon as the full-suite gate is mathematically unattainable.
The 40-case generation, expanded, and holdout gates must report the same runtime
identity. Baseline approval also compares it with the production environment,
persists it, and rejects model or endpoint drift. All three gate reports carry
the evaluated source dataset hash; approval refreshes the baseline hashes only
after those hashes match the current protected datasets exactly.

```powershell
python scripts/evaluate_retrieval_suite.py --input eval/legal_expanded_240.jsonl --output data/retrieval-240.json --top-k 5 --use-production-pipeline --production-decomposition --minimum-citation-recall 0.95 --resume
python scripts/evaluate_retrieval_suite.py --input eval/legal_holdout_150.jsonl --output data/retrieval-holdout-150.json --top-k 5 --use-production-pipeline --production-decomposition --minimum-citation-recall 0.95 --resume
```

Production query understanding is processed in bounded batches of four cases by
default (`--understanding-concurrency`), so a failed gate cannot leave the rest of
the suite prefetching paid model calls. Provider failures are sanitized, recorded
at the evaluation boundary, and fail the run even though the online query path
normally has a user-facing fallback. A partial or degraded run therefore cannot
produce a release report.

## Run the judge

The script loads `DEEPSEEK_API_KEY` from `.env` by default. Model precedence is
`--model`, `RAGAS_MODEL`, `DEEPSEEK_MODEL`, then `deepseek-chat`. Endpoint
precedence is `--base-url`, `RAGAS_BASE_URL`, `DEEPSEEK_API_URL`, then the
DeepSeek API default. The legacy `RAGAS_*` environment names remain only for
workflow compatibility.

```powershell
python scripts/evaluate_ragas.py --input data/lrage_exports/legal_core.jsonl --output data/ragas_eval/legal_core.json
```

For a bounded smoke run:

```powershell
python scripts/evaluate_ragas.py --input data/lrage_exports/legal_core.jsonl --limit 2 --metrics answer_accuracy,faithfulness
```

Requests are serial by default, retries and output sizes are bounded, API keys
are never written to reports, and terminal output omits questions, answers, and
contexts. Evaluation text is sent to the configured OpenAI-compatible judge
endpoint, so production operators must treat that endpoint as a data processor.

## Enforce the combined gate

```powershell
python scripts/check_quality_gate.py --ragas-report data/ragas_eval/legal_core.json --export data/lrage_exports/legal_core.jsonl --top-k 5
```

The compatibility option `--ragas-report` points to the native judge report.
Default minima are answer accuracy 0.70, faithfulness 0.80, context precision
0.60, context recall 0.70, citation Recall@5 0.80, and citation MRR@5 0.70.
Missing labels and judge-contract mismatches fail by default; `--allow-missing`
is for exploratory datasets only and must not be used for release approval.
The release gate additionally verifies that the judge report hashes the exact
export it scored and that all export rows have one source dataset hash and one
generation-runtime identity.
