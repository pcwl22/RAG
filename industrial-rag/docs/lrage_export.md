# LRAGE export

This project integrates LRAGE as an external evaluation target. The main
`industrial-rag` runtime does not import LRAGE and does not install LRAGE
dependencies, because LRAGE currently expects Python `>=3.8,<3.11` while this
project may run in a newer Python environment.

## Input cases

Create a JSONL file where each line is one case:

```json
{"id":"case-1","query":"Question to run through Industrial RAG","expected_answer":"Optional reference answer","rubric":"Optional judge rubric"}
```

Required field:

- `query`: question sent to `/query/enhanced`

Optional fields:

- `id`: stable case identifier
- `expected_answer`, `reference_answer`, or `target`: reference answer copied to LRAGE `target`
- `rubric`: custom LRAGE judge rubric
- `chat_history`: chat history passed to enhanced query
- `metadata`: case metadata preserved in the export
- `expected_citations`: article identifiers used for deterministic retrieval metrics
- `expected_sources`: expected source names preserved in export metadata

## Export

Run the exporter from the `industrial-rag` project directory:

```powershell
python scripts/export_lrage.py --input cases.jsonl --output-dir exports/lrage
```

Useful options:

```powershell
python scripts/export_lrage.py --input cases.jsonl --output-dir exports/lrage --task-name industrial_rag_lrage --top-k 5 --enable-rerank --max-document-chars 16000
```

The script executes each case through the current enhanced query pipeline and
writes:

- `<task>.jsonl`: LRAGE-style dataset records
- `<task>.yaml`: LRAGE task template using `LLM-Eval`
- `<task>.manifest.json`: export metadata

The repository includes a starter legal set at `eval/legal_core.jsonl`. After
exporting it, calculate deterministic retrieval metrics with:

```powershell
python scripts/score_retrieval.py exports/lrage/industrial_rag_lrage.jsonl --top-k 5
```

## Output schema

Each JSONL record includes LRAGE-facing fields:

- `Prompt`: original query
- `Document`: retrieved contexts formatted as one document block
- `Rubric`: judge rubric
- `target`: optional reference answer

It also keeps debug fields from the current system:

- `prediction` / `answer`: generated Industrial RAG answer
- `contexts`: normalized retrieved chunks with scores and metadata
- `understanding`: query understanding payload
- `sub_answers`: decomposition answers when present
- `metadata`: export and case metadata

Context text prioritizes the text actually visible to the model:
`child_content`, then metadata `child_content`, then metadata `article_text`,
then stored `content`.

## Running in LRAGE

Use a separate LRAGE environment, for example Python 3.10:

```powershell
conda create -n lrage python=3.10 -y
conda activate lrage
pip install -e E:\RAG\LRAGE
```

The local smoke run was verified with this dedicated environment:

```powershell
conda create -n lrage python=3.10 -y
conda install -n lrage -c conda-forge openjdk=21 -y
C:\Users\14659\anaconda3\envs\lrage\python.exe -m pip install --no-deps -e E:\RAG\LRAGE
C:\Users\14659\anaconda3\envs\lrage\python.exe -m pip install numpy torch transformers datasets evaluate scikit-learn jsonlines zstandard sqlitedict tqdm-multiprocess pytablewriter rouge-score sacrebleu word2number more_itertools numexpr accelerate peft pyserini rerankers faiss-cpu pybind11
```

LRAGE imports retrieval dependencies at startup, so set Java and cache paths even
when evaluating pre-exported Industrial RAG predictions:

```powershell
$env:JAVA_HOME = 'C:\Users\14659\anaconda3\envs\lrage\Library\lib\jvm'
$env:PATH = "$env:JAVA_HOME\bin\server;$env:JAVA_HOME\bin;C:\Users\14659\anaconda3\envs\lrage\Library\bin;$env:PATH"
$env:HF_HOME = 'E:\RAG\industrial-rag\data\lrage_hf_cache'
$env:HF_DATASETS_CACHE = 'E:\RAG\industrial-rag\data\lrage_hf_cache\datasets'
```

Use the exported task directory, not a single YAML path. The local LRAGE checkout
contains a `replay_prediction` model that replays each JSONL row's `prediction`
or `answer`, so LRAGE judges the Industrial RAG answer instead of generating a
new answer.

Dummy judge smoke test:

```powershell
$env:OPENAI_API_KEY = 'EMPTY'
C:\Users\14659\anaconda3\envs\lrage\python.exe -m lrage `
  --model replay_prediction `
  --judge_model dummy `
  --tasks E:\RAG\industrial-rag\data\lrage_exports `
  --output_path E:\RAG\industrial-rag\data\lrage_eval\smoke_utf8 `
  --limit 1 `
  --log_samples `
  --verbosity INFO
```

DeepSeek judge run. This reads `DEEPSEEK_API_KEY` from `.env` into the current
process without printing the value:

```powershell
$keyLine = Get-Content -Encoding UTF8 -Path E:\RAG\industrial-rag\.env |
  Where-Object { $_ -match '^DEEPSEEK_API_KEY=' } |
  Select-Object -First 1
$env:OPENAI_API_KEY = ($keyLine -replace '^DEEPSEEK_API_KEY=', '').Trim().Trim('"').Trim("'")

C:\Users\14659\anaconda3\envs\lrage\python.exe -m lrage `
  --model replay_prediction `
  --judge_model openai-chat-completions `
  --judge_model_args 'model=deepseek-chat,base_url=https://api.deepseek.com/v1' `
  --tasks E:\RAG\industrial-rag\data\lrage_exports `
  --output_path E:\RAG\industrial-rag\data\lrage_eval\deepseek `
  --limit 1 `
  --log_samples `
  --verbosity INFO
```

The generated task follows LRAGE's `LLM-Eval` task shape and points to the
absolute JSONL dataset path. Sample-level results are written as
`samples_<task>_<timestamp>.jsonl`; aggregate results are written as
`results_<timestamp>.json`.
