# Industrial RAG

## Local startup

1. Create `.env` from `.env.example` and set `POSTGRES_PASSWORD` and one LLM key. The loopback-only `laptop` profile keeps authentication disabled; `RAG_API_KEY` is required by the `base`/Compose profile.
2. Start dependencies with `docker compose -f ..\docker-compose.yml up -d postgres redis`.
3. Use the pinned GPU environment in `requirements-gpu-verified.txt` when running CUDA workloads.
4. Start the API with `uvicorn app.main:app --reload`.

The production Compose profile runs `python -m app.vectorstore.migrate` with the
database owner credential, then starts API and worker processes as the dedicated
`rag_runtime` login. Configure a distinct `POSTGRES_APP_PASSWORD`; runtime services
have no schema-owner credential and keep automatic DDL disabled.

Production operators should run `scripts/validate_production_env.py` before deployment and use the
guarded `scripts/backup_postgres.py` / `scripts/restore_postgres.py` utilities. External IdP, secret
management, HA, and recovery requirements are tracked in
[`deploy/production-checklist.md`](deploy/production-checklist.md).

For Docker production-style startup, set `QUEUE_PROVIDER=celery`; the Compose
`api` profile then also starts the durable Celery worker. The `frontend` profile
automatically starts the local Keycloak identity services and uses OIDC bearer
tokens. API-key service calls remain
available through `X-API-Key: $RAG_API_KEY`. Prometheus is available only at the internal
`/internal/metrics` path and should use `X-Metrics-Token: $RAG_METRICS_TOKEN`.
Liveness is available at `/health/live`; readiness is `/health/ready`.

API-key service accounts default to the `viewer` role. Set
`RAG_SERVICE_ROLES=viewer,editor` only for upload automation; add `admin` only
for a service that must delete documents.

To idempotently reprocess uploaded files for an explicit tenant:

```powershell
python scripts\reprocess_documents.py `
  --tenant-id 00000000-0000-0000-0000-000000000001 `
  --upload-dir data\uploads `
  --pattern "*.pdf"
```

The default `reranker.failure_mode=closed` returns no context if the reranker
cannot load or fails during inference. Use `open` only when preserving raw
retrieval results is an intentional availability tradeoff.

## Verification

```powershell
python -m compileall -q app
python -m ruff check app tests scripts
python -m mypy app/auth.py app/security.py app/api/input_validation.py app/evaluation `
  scripts/validate_evaluation_assets.py scripts/validate_production_env.py `
  scripts/validate_release_baseline.py --follow-imports=skip
python -m pytest tests -q
python scripts/validate_evaluation_assets.py
python scripts/validate_release_baseline.py
```

For the lightweight CI/test environment, install `requirements-ci.lock.txt` with
`--require-hashes`. For the controlled Windows CUDA stack, run
`scripts/setup_gpu_env.ps1`; it installs the SHA-256-locked CUDA 12.6 Torch
carrier and then `requirements-gpu.lock.txt`. Linux workers use the separate
CPU Torch carrier; API images inherit Torch only from the digest-pinned CUDA
base. `requirements-runtime.lock.txt` intentionally excludes Torch itself so it
cannot replace either carrier. Do not install the project with unconstrained
dependencies in production.

The starter retrieval benchmark is `eval/legal_core.jsonl`. The committed expanded and representative
judge files are generated from the screened fact-pattern holdout and pass the leakage-aware asset
validator. See `../README.md#发布质量状态` for the current release gate status. See
`docs/lrage_export.md` for export and Recall/MRR scoring commands.

For zero-exception, hash-locked LLM-judged answer and context evaluation, see
`docs/ragas_evaluation.md`.
