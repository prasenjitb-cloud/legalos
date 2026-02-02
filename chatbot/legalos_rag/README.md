# legalos_rag

**Local helper module** containing the structured parts of the RAG pipeline: retrieval, prompt templates, output schemas, and invocation. Used by `chatbot/main.py`; not run standalone.

---

### Directory structure

```text
legalos_rag/
├── README.md
├── __init__.py
├── factsRetriever.py 
├── ragInvoker.py     
└── prompt/
    ├── promptSchema.py    
    └── prompts.py        
```

---

## Module structure & execution flow

### `prompt/promptSchema.py`

Pydantic schemas for structured outputs: `LegalAnswer` (answer_found, act_name, section, explanation, citations) and `Citation` (pdf_number, page, file_name, quote).

### `prompt/prompts.py`

- Builds the **RAG prompt skeleton**:
  - Loads a versioned template from a JSON file (e.g. `ragPrompts.json`) using a **prompt version key** (like `"v1"`).
  - Injects `format_instructions` from the output parser.
  - Returns a LangChain `PromptTemplate` with:
    - `input_variables=["facts", "question"]`
    - `partial_variables={"format_instructions": parser.get_format_instructions()}`.

### `factsRetriever.py`

- **setup_vectorstore** — Build Qdrant vectorstore with HuggingFace embeddings.
- **format_docs** — Turn a list of documents into a single string for the prompt.
- **getFacts** — Retrieve top-k chunks for a query from the vector DB and return them formatted for the prompt.

### `ragInvoker.py`

- **invoker** — Glue function that:
  - Builds the output parser for `LegalAnswer` (from `prompt/promptSchema.py`).
  - Calls `prompt/prompts.setup_rag_prompt_skeleton(parser, prompt_version, templates_path)` to get the `PromptTemplate`.
  - Renders the final prompt text by formatting with:
    - `facts` (retrieved docs)
    - `question` (user query)
  - Invokes the SLM, parses the JSON-like output into `LegalAnswer`, logs the run, and returns the result.
- **log_rag_run** — Append one RAG run (query, final prompt text, parsed output, model) as a JSONL line to `rag_runs.jsonl`.

---

## How `chatbot/main.py` uses this module

`chatbot/main.py` is orchestration-only:

1. Parse config/CLI (`vectordbpath`, `prompt` version like `"v1"`, and `templatespath` to `ragPrompts.json`).
2. Initialize the SLM (Ollama, `qwen2.5:3b-instruct`).
3. Call **`factsRetriever.getFacts(db_path, query)`** to fetch relevant chunks from the vector DB.
4. Call **`ragInvoker.invoker(slm, retrieved_docs, query, model_name, prompt, templates_path)`** to build the prompt, invoke the SLM, and get a structured `LegalAnswer`.

Example:

```python
retrieved_docs = chatbot.legalos_rag.factsRetriever.getFacts(q=query, db_path=db_path)
result = chatbot.legalos_rag.ragInvoker.invoker(
    slm,
    retrieved_docs,
    query,
    SLM_MODEL_NAME,
    prompt,          # e.g. "v1"
    templates_path,  # e.g. "./ragPrompts.json"
)
```

This keeps retrieval and model invocation separate so you can swap:
- prompt versions (via `prompt` + `ragPrompts.json`)
- models (via `SLM_MODEL_NAME`)
without changing the retrieval logic.

---

## Prompt workflow

End-to-end prompt formation looks like this:

1. **Config / CLI**
   - `prompt`: version key, e.g. `"v1"`.
   - `templatespath`: path to `ragPrompts.json`, which looks like:

   ```json
   {
     "v1": {
       "template": "You are a legal document reader...\\n\\nOutput:\\n{format_instructions}\\n\\nFacts:\\n{facts}\\n\\nQuery:\\n{question}"
     }
   }
   ```

2. **`prompt/prompts.setup_rag_prompt_skeleton(...)`**
   - Loads the JSON from `templatespath`.
   - Picks `data[prompt]["template"]`.
   - Wraps it in a `PromptTemplate` with:
     - `{format_instructions}` filled from the output parser.
     - `{facts}` and `{question}` as runtime inputs.

3. **`ragInvoker.invoker(...)`**
   - Calls `prompt.format(facts=retrieved_docs, question=query)` to produce the final string sent to the SLM.

4. **SLM output → schema + logging**
   - SLM output is parsed into `LegalAnswer`.
   - The full prompt, query, model, and parsed output are logged to `rag_runs.jsonl`.

### Prompt workflow diagram

```mermaid
flowchart TD
    A[Config/CLI: prompt='v1', templatespath='./ragPrompts.json'] --> B[ragInvoker.invoker]
    B --> C[Create PydanticOutputParser with LegalAnswer schema]
    C --> D[prompt/prompts.setup_rag_prompt_skeleton]
    D --> E[Load ragPrompts.json]
    E --> F[Select data['v1']['template']]
    F --> G[Create PromptTemplate with format_instructions]
    G --> H[prompt.format with facts + question]
    H --> I[Final prompt string]
    I --> J[Invoke SLM]
    J --> K[Parse JSON output to LegalAnswer]
    K --> L[Log to rag_runs.jsonl]
    L --> M[Return LegalAnswer]
```

---

## Logging (`rag_runs.jsonl`)

Each RAG run is appended as one JSON line: timestamp, model, query, final_prompt, and parsed output. Used for prompt-engineering iteration and review without re-running experiments.
