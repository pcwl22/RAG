# Contributing

All changes should be made on a short-lived branch and merged through a reviewed pull request.
Do not commit `.env`, model weights, uploaded documents, generated evaluation exports, or secrets.

Before opening a pull request, run:

```powershell
Set-Location E:\RAG\industrial-rag
python -m ruff check app tests scripts
python -m mypy app/auth.py app/security.py app/api/input_validation.py app/evaluation `
  scripts/validate_evaluation_assets.py scripts/validate_production_env.py `
  scripts/validate_release_baseline.py --follow-imports=skip
python -m pytest tests -q

Set-Location E:\RAG\rag-frontend
npm.cmd test
npm.cmd run build
```

Changes to retrieval, chunking, prompts, embeddings, reranking, generation, or evaluation code must
rerun the committed evaluation suites and update the approved release baseline. Database changes
must be added as a new immutable SQL file under
`industrial-rag/app/vectorstore/migrations/versions`; never edit an applied migration.

Pull requests should describe tenant-isolation impact, rollback steps, configuration changes,
quality results, and whether data or model behavior changed. Security-sensitive and database
migration paths require approval from their CODEOWNER.
