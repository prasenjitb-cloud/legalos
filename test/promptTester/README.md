# promptTester

Batch-run a **prompt template** over a **question set** using the LegalOS RAG pipeline and write all results to a single run file.

## What it does

For each question in the question set, promptTester:

1. **Retrieves** relevant chunks from the vector DB (`getFacts`).
2. **Runs the RAG pipeline** with your template sent to the SLM, and the response is parsed into a structured `LegalAnswer` (answer_found, explanation, citations, etc.).
3. **If nothing was retrieved**, it still records a result with `answer_found=False` and empty explanation/citations.
4. **Appends** that result to an in-memory run log.

After processing every question, it writes **one JSON file** per batch run: `outputs/run_<run_id>.json`. The run ID is a timestamp (`YYYYMMDD_HHMMSS`).

So: **one config → one prompt template + one question set → one run file** containing all question–result pairs.

## Outputs

- **Directory:** `outputs/` (created under the current working directory).
- **File:** `run_<run_id>.json`, e.g. `run_20260207_115216.json`.

Each run file contains:

- **Run metadata:** `run_id`, `model`, `vectordbpath`, `questionsetfile`, `promptTemplate`.
- **results:** list of one object per question with:
  - `question_id`, `question`
  - `retrieved_docs` (chunks used for that question)
  - `output`: the structured `LegalAnswer` (e.g. `answer_found`, `explanation`, `citations`).

You can use this file for evaluation (e.g. with `evaluatePrompt`) or manual inspection.

## Config file

The batch runner uses the same config validation as the interactive RAG CLI via **`chatbot.legalos_rag.ensure_requirements(config)`**, plus a required question set path. A JSON config file must provide:

| Key | Description |
|-----|-------------|
| `vectordbpath` | Path to the Qdrant vector DB directory (e.g. `./vectorDB`). |
| `promptTemplate` | Object with a `"text"` key: the full prompt template string. Must include placeholders `{format_instructions}`, `{facts}`, and `{question}` (used by the RAG prompt builder). |
| `questionsetfile` | Path to the question set JSON file (relative to the directory you run the script from, or absolute). |

Paths in the config are resolved with `os.path.abspath()` from the current working directory. Retrieval and invocation use **`chatbot.legalos_rag.runRag.getFacts`** and **`chatbot.legalos_rag.runRag.invoker`**.

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

- **promptRunBatch.py** – Entrypoint: loads config, runs the batch, writes `outputs/run_<run_id>.json`.
- **config/** – Example config(s), e.g. `v1.json`.
- **questionSet.json** – Example question set.
