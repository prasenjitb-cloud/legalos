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

You can use this file for evaluation (e.g. with the evaluator script `evaluate.py` and its `evaluate_rag` function) or manual inspection.

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
# 1. Run the batch prompt tester (produces outputs/run_<run_id>.jsonl)
python -m test.promptTester.promptRunBatch --config test/promptTester/config/v1.json
```

Use a different config file to change the vector DB, template, or question set.

## Files

- **promptRunBatch.py** – Entrypoint: loads config, runs the batch, appends each result to `outputpath/run_<run_id>.jsonl`.
- **config/** – Example config(s), e.g. `v1.json`.
- **questionSet.json** – Example question set.

## Evaluator and evaluatorPrompt

The evaluator pipeline has two pieces:

- `evaluator/evaluatorPrompt.py` – defines the **`RAGEvaluation`** schema and the **evaluator prompt template**.
- `evaluator/evaluate.py` – CLI script that reads a batch run JSONL file and uses the evaluator LLM to score each question.

### RAGEvaluation schema (`evaluatorPrompt.py`)

The `RAGEvaluation` Pydantic model specifies all evaluation dimensions the LLM must output as JSON:

- **factual_existence**: Whether the model correctly decided if an answer exists in the Retrieved Facts.
- **factual_faithfulness**: How strictly the answer is supported by the retrieved facts.
- **query_relevance**: How directly the answer addresses the user’s question using those facts.
- **legal_precision**: Accuracy of legal acts, sections, and terminology.
- **clarity**: Structure and readability of the answer.
- **citation_quality**: Quality of selected citations from the facts.
- **explanation_from_citations**: How well the explanation is grounded in the cited text.
- **total**: Computed sum of all the above fields (max score = 28).

### Evaluator prompt template (`evaluatorPrompt.py`)

The `setup_evaluator_prompt(parser)` function builds a `PromptTemplate` that:

- Injects `format_instructions` from a `PydanticOutputParser` for `RAGEvaluation`, forcing the LLM to return valid JSON only.
- Takes `question`, `facts`, `model_answer`, and `citations` as inputs.
- Guides the LLM to:
  1. Decide whether the facts contain statutory language that answers the question.
  2. Decide if the model’s `answer_found` decision was correct.
  3. Score all fields according to the rubric and hard constraints (e.g. lower scores if relevant text exists but the model abstained, or if it hallucinates beyond the facts).

### Evaluator script (`evaluator/evaluate.py`)

The evaluator script:

- Loads a run JSONL file produced by `promptRunBatch`:
  - First line: run metadata.
  - Subsequent lines: one result per question (`question`, `retrieved_chunks`, `output`).
- Builds the evaluator LLM and wiring:
  - `evaluator_parser = PydanticOutputParser(pydantic_object=RAGEvaluation)`
  - `evaluator_prompt = setup_evaluator_prompt(evaluator_parser)`
- For each result without error:
  - Normalizes `retrieved_chunks` into a `facts` string.
  - Extracts `model_answer` + `citations` from the RAG output.
  - Formats the evaluator prompt and calls the evaluator LLM.
  - Parses the JSON response into a `RAGEvaluation` instance and stores it under the `evaluation` key.
- Aggregates numeric scores across all evaluated questions:
  - Sums each field (`factual_existence`, `factual_faithfulness`, `query_relevance`, `legal_precision`, `clarity`, `citation_quality`, `explanation_from_citations`, `total`).
  - Computes per-field averages and writes them to `aggregate_scores`.
  - Includes `max_scores` for each field (matching the schema).

### How to run the evaluator

From the **legalos** project root:

```bash
# 2. Evaluate a run (produces evaluation_<run_id>.json in the configured outputpath)
python -m test.promptTester.evaluator.evaluate --config test/promptTester/evaluator/config/eval1.json
```

The evaluator config must provide:

- **batchResultFile**: Path to the JSONL run file (e.g. `test/promptTester/outputs/run_<run_id>.jsonl`).
- **outputpath**: Directory where `evaluation_<run_id>_<timestamp>.json` will be written.

The evaluation JSON contains:

- `run_metadata`: Source run info, SLM model, evaluator model, counts, and `max_scores`.
- `aggregate_scores`: Average scores per field across evaluated questions.
- `results`: Per-question entries with `question`, original `rag_output`, and `evaluation` (the `RAGEvaluation` fields).
