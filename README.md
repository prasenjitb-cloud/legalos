# Legalos ⚖️

**Your one-stop solution to all your legal problems.**

---

### FILE STRUCTURE-

```text
.
├── vectorDB
│   ├── collection
│   │   └── central_acts
│   │
│   └── meta.json   # Qdrant DB metadata (vector size, collection config, etc.)
│
├── factsDB
│   └── centralActPdfs
│       ├── data
│       │   ├── 189008.pdf
│       │   ├── 189108.pdf
│       │   └── ...
│       └── failed_pdfs.txt
│
├── config/              # RAG configs (one JSON per prompt setup)
│   └── rag_v1.json      # Config: vectordbpath, promptTemplate, model.model_name, logging (logfile, exclude_*)
│
├── chatbot
│   ├── README.md
│   ├── centralActsDownloader.py
│   ├── main.py
│   ├── vectorDbSetup.py
│   └── legalos_rag
│       ├── README.md
│       ├── __init__.py
│       ├── runRag.py
│       └── prompt
│           ├── prompts.py
│           └── promptSchema.py
│
├── test
│   └── promptTester          # Batch prompt runner + evaluator
│       ├── README.md         # Detailed docs for batch + evaluator
│       ├── promptRunBatch.py # Batch runner entrypoint
│       ├── config/           # Batch configs (vectordbpath, promptTemplate, questionsetfile, outputpath)
│       ├── questionSet.json  # Example question set
│       ├── outputs/          # JSONL batch run outputs (one line per question)
│       └── evaluator/
│           ├── evaluate.py        # LLM-based evaluator for batch runs
│           ├── evaluatorPrompt.py # RAGEvaluation schema + evaluator prompt template
│           ├── config/            # Evaluator configs (batchResultFile, outputpath)
│           └── evaluationResults/ # Saved evaluation_<run_id>_*.json files
│
└── requirements.txt
```

## Getting Started

### Prerequisites

Before running **Legalos**, make sure you have:

- **Python 3.11**
  > Python 3.11 works best with LangChain. Newer versions may cause compatibility issues.
- **pip** (or any Python package manager)
- **Ollama** (for running local SLM)

> **Note**: Legalos works fully offline with no external API dependencies or environment variables required.

> **Running scripts:** This project uses the **`-m` module structure**. Run Python scripts from the **project root** (`legalos/`) with `python -m chatbot.<module>`, e.g.:
>
> - **Interactive RAG** (ask questions one at a time):
>   ```bash
>   python -m chatbot.main --config ./config/rag_v1.json
>   ```
> - **Batch prompt testing** (run a question set, write JSONL to `outputpath/`):
>   ```bash
>   python -m test.promptTester.promptRunBatch --config test/promptTester/config/v1.json
>   ```
> - **Evaluate a batch run** (LLM-based scoring into `evaluationResults/`):
>   ```bash
>   python -m test.promptTester.evaluator.evaluate --config test/promptTester/evaluator/config/eval1.json
>   ```
>   See `test/promptTester/README.md` for details on the RAGEvaluation schema, evaluator prompt, and evaluation JSON format.
>

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/prasenjitb-cloud/legalos.git
```

### 2. Navigate to the project directory

```bash
cd legalos
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## AWS Deployment

Want to deploy Legalos on AWS EC2? Check out the complete deployment guide:

📁 **[aws_deployment/README.md](./aws_deployment/README.md)** - Step-by-step AWS deployment guide with all terminal commands

---

## You're all set!
