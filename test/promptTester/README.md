# promptTester

Batch-run a **prompt template** over a **question set** using the LegalOS RAG pipeline and append each result to a JSONL run file.

## What it does

The batch runner compiles the same **rewrite → retrieve → generate** LangGraph workflow used by the chatbot and reuses it for every question. It calls **`chatbot.main.run_rag()`** and appends each result immediately, without keeping the full batch in memory. When no chunks are retrieved, `run_rag` returns `(None, [], None, queries)` and the output is recorded as `null`.

So: **one config → one prompt template + one question set → one JSONL run file** containing all question–result pairs.

## Outputs

- **Directory:** `outputpath` from config (e.g. `outputs/`).
- **File:** `run_<run_id>.jsonl`, e.g. `run_20260207_115216.jsonl`.

Each run file is JSONL (one JSON object per line):

- **Line 1:** Run metadata with `run_id`, `model`, `vectordbpath`, `questionsetfile`, `promptTemplate`, `type: "metadata"`.
- **Lines 2+:** One object per question with:
  - `question_id`, `question`
  - `rewritten_queries`: list of strings passed to retrieval (original query plus SLM-generated search variants when rewriting succeeds)
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

## Running the new path and comparing it

With a `pipeline` object in the batch config (see the [chatbot README](../../chatbot/README.md#run-the-new-path-from-the-cli-or-the-batch-runner)),
the runner can also run the new graph. A question may carry its own `"act_id"`; otherwise the config's Act is used.

```bash
python -m test.promptTester.promptRunBatch --config test/promptTester/config/new_path.json
```

`test/promptTester/config/new_path.json` runs both paths over `questionSet_new_path.json` (four questions, each with its Act).
The run file changes only when `pipeline.mode` is `new` or `both`:

- **Line 1 (metadata)** also has `pipeline` and `act_id`.
- **`both`:** each question line keeps the keys above unchanged and adds a `new` object, so the old and new result for
  the same question are on one line. **`new`:** each line has `question_id`, `question` and `new` only.
- **`new` object:** `status` (`answered`, `needs_fact`, `partial`, `no_evidence`, `failed_checks`, `call_limit` or `error`), `answer`,
  `asks`, `missing` (the gaps), `sources` (source ID, Act, section and pages), `decisions` (the path through the graph), `calls`
  (jobs run), `verdict` (the evaluator's) and `seconds`. Without a `pipeline` object the file is exactly as before.

The two paths are not searching the same material: the current path searches all Acts in the vector database, the new path
one checked Act. Review the pairs by hand; the evaluator script above scores only the current path's `output`.

## Release check

`test/promptTester/questionSet_release.json` is a fresh set of 17 questions for the five Acts that passed their source check
(three answerable questions for each of the five Acts, plus two questions the Acts cannot answer: one for Hindu Marriage and one for GST). Each has an
`act_id` and the sections that answer it (`expected_sections`), which I fixed from the Acts' section titles before any run. Motor Vehicles is
not enabled: it has not passed its source check. Run it with the new path only:

```bash
python -m test.promptTester.promptRunBatch --config test/promptTester/config/release.json
```

Criteria, fixed before the run. An answerable question passes when its `new.status` is `answered`, `partial` or `needs_fact` and an expected section
is among `new.sources`. A question the Act cannot answer passes only when the status is `no_evidence`. The release bar is: every Act passes at
least 2 of its 3 answerable questions, both unanswerable questions pass, and at least 14 of 17 pass overall.

**Result with `qwen2.5:3b-instruct`: not met.** 10 of 17 passed (59%). Both unanswerable questions passed, but three Acts
(Hindu Marriage, GST and Juvenile Justice) passed only 1 of their 3 answerable questions, and the overall total is 4 short of 14.

| Act | Answerable passed | Failed |
| --- | ---: | --- |
| Consumer Protection | 3 of 3 | |
| POSH | 2 of 3 | duties of the District Officer: cited sections 6, 21 and 25, not 20 |
| Hindu Marriage | **1 of 3** answerable | alternate relief in divorce: cited 23A and 14, not 13A; the court a petition goes to: `failed_checks` |
| Juvenile Justice | **1 of 3** | Child Welfare Committee: cited the definitions section, not 27; observation homes: `no_evidence` |
| GST | **1 of 3** answerable | time of supply of services and transitional input tax credit: `failed_checks` |

Three questions ended in `failed_checks`, which is the evaluator finding a citation or unsupported-claim problem. Several others were
`answered` while citing the wrong sections, so **`answered` does not mean correct**: the evaluator checks sources and omissions, not whether
the right section was found. Each question took about 100 seconds (28 minutes for the set). CLI, batch and API agree on the same question:
status, answer and sources were identical in the check. The new path should not be used to answer users until retrieval, the evidence filter and
the condition check improve and this set passes.

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
- **percentage**: Computed as `(total / 28) * 100`.

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
  - Subsequent lines: one result per question (`question`, `rewritten_queries`, `retrieved_chunks`, `output`).
- Builds the evaluator LLM and wiring:
  - `evaluator_parser = PydanticOutputParser(pydantic_object=RAGEvaluation)`
  - `evaluator_prompt = setup_evaluator_prompt(evaluator_parser)`
- For each result without error:
  - Normalizes `retrieved_chunks` into a `facts` string.
  - Extracts `model_answer` + `citations` from the RAG output.
  - Formats the evaluator prompt and calls the evaluator LLM.
  - Parses the JSON response into a `RAGEvaluation` instance and stores it as `layer2` (per-metric score + percentage breakdown).
- Additionally computes a simple `layer1` structural summary from the raw RAG output:
  - `explanation_provided`: `true` if `rag_output.explanation` exists and is non-empty.
  - `num_retrieved_chunks`: count of `[DOC n]` markers in `retrieved_chunks`.
  - `num_citations`: number of citation objects in `rag_output.citations`.
- Aggregates numeric scores across all evaluated questions:
  - Sums each field (`factual_existence`, `factual_faithfulness`, `query_relevance`, `legal_precision`, `clarity`, `citation_quality`, `explanation_from_citations`, `total`, `percentage`).
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
- `results`: Per-question entries with:
  - `question_id`, `question`
  - `rag_output`: the original model output object (or `null`)
  - `layer1`: structural summary (`explanation_provided`, `num_retrieved_chunks`, `num_citations`)
  - `layer2`: evaluator scores (per metric: `{ score, percentage }`, plus `total`)
