# promptTester

Batch-run a **prompt template** over a **question set** using the LegalOS RAG pipeline and append each result to a JSONL run file.

## What it does

For each question in the question set, promptTester calls **`chatbot.main.run_rag()`** and appends the result to the run file immediately (no in-memory accumulation). When no chunks are retrieved, `run_rag` returns `(None, [], None)` and output is recorded as `null`.

So: **one config → one prompt template + one question set → one JSONL run file** containing all question–result pairs.

## Outputs

- **Directory:** `outputpath` from config (e.g. `outputs/`).
- **File:** `run_<run_id>.jsonl`, e.g. `run_20260207_115216.jsonl`.

Each run file is JSONL (one JSON object per line):

- **Line 1:** Run metadata with `run_id`, `model`, `vectordbpath`, `questionsetfile`, `promptTemplate`, `type: "metadata"`.
- **Lines 2+:** One object per question with:
  - `question_id`, `question`
  - `retrieved_chunks` (chunks used for that question)
  - `output`: the structured `LegalAnswer` (e.g. `answer_found`, `explanation`, `citations`), or `null` when no chunks were retrieved.

You can use this file for evaluation (e.g. with `evaluatePrompt`) or manual inspection.

## Config file

The batch runner uses the same config validation as the interactive RAG CLI via **`chatbot.legalos_rag.ensure_requirements(config)`**, plus a required question set path. A JSON config file must provide:

| Key | Description |
|-----|-------------|
| `vectordbpath` | Path to the Qdrant vector DB directory (e.g. `./vectorDB`). |
| `promptTemplate` | Object with a `"text"` key: the full prompt template string. Must include placeholders `{format_instructions}`, `{facts}`, and `{question}` (used by the RAG prompt builder). |
| `questionsetfile` | Path to the question set JSON file (relative to the directory you run the script from, or absolute). |
| `outputpath` | Directory for run output files (e.g. `outputs/`). |

Paths in the config are resolved with `pathlib.Path().resolve()`. The batch uses **`chatbot.main.run_rag`** for each question.

## Question set format

A JSON array of objects, each with at least:

- **id** – unique identifier (e.g. integer).
- **question** – text of the question.

Example:

```json
[
  { "id": 1, "section": "POSH", "question": "My boss keeps sending me personal messages..." },
  { "id": 2, "section": "HMA", "question": "My husband has abandoned me. Can I remarry?" }
]
```

Entries with an empty `question` are skipped (no result is written for them).

## How to run

From the **legalos** project root:

```bash
python -m test.promptTester.promptRunBatch --config test/promptTester/config/v1.json
```

Use a different config file to change the vector DB, template, or question set.

## Files

- **promptRunBatch.py** – Entrypoint: loads config, runs the batch, appends each result to `outputpath/run_<run_id>.jsonl`.
- **config/** – Example config(s), e.g. `v1.json`.
- **questionSet.json** – Example question set.
