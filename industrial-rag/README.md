# Industrial RAG

## Local startup

1. Create `.env` from `.env.example` and set `POSTGRES_PASSWORD`, `RAG_API_KEY`, and one LLM key.
2. Start dependencies with `docker compose -f ..\docker-compose.yml up -d`.
3. Use the pinned GPU environment in `requirements-gpu-verified.txt` when running CUDA workloads.
4. Start the API with `uvicorn app.main:app --reload`.

Protected API routes require `X-API-Key: $RAG_API_KEY`. Liveness is available at `/health/live`; readiness is `/health/ready`.

## Verification

```powershell
python -m compileall -q app
python -m pytest tests -q
```

For the lightweight CI/test environment, install `requirements-ci.txt`. For
the verified Windows CUDA stack, use `requirements-gpu-verified.txt`; do not
mix unconstrained upgrades into that environment.

The starter retrieval benchmark is `eval/legal_core.jsonl`. See
`docs/lrage_export.md` for export and Recall/MRR scoring commands.

For LLM-judged answer and context evaluation with Ragas, see
`docs/ragas_evaluation.md`.
