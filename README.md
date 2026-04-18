# Legalos ⚖️

**High-level overview**

India’s legal ecosystem is fragmented: citizens struggle to find the right lawyers or even basic procedural information, while lawyers lack simple tools to organize case material and surface insights from their own documents. Legalos will be a legal operating system for this ecosystem by solving both sides of this problem.

Right now, this repo focuses on a **RAG-based legal chatbot** that will let users ask natural-language questions over Indian acts and get grounded, citation-rich answers from an offline vector database. Retrieval uses a **query rewriter**: informal questions are expanded into short legal search phrases so the vector store is queried from multiple angles; chunks are merged and deduplicated before generation.

In its full form, Legalos will have three pillars:

- **Client-facing legal assistant**: A chatbot that will help citizens understand procedures, rights, and basic legal concepts, and will guide them toward the right type of legal help.
- **Lawyer-facing copilot**: A chatbot that will sit on top of a lawyer’s own documents and workflows, will help retrieve case information quickly, assist with legal research, and support drafting and refining arguments.
- **Collaboration and consultation platform**: A shared layer where clients and lawyers will be able to connect, manage engagements, and use the assistants together during consultations.

The long-term vision is to become India’s premier Legal OS: the default, trusted layer where citizens discover legal help, lawyers manage their matters, and both can query courts, acts, and case history and get reliable, context-aware answers.

---

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
│       ├── queryRewriter.py   # Legal phrasing + variant queries for retrieval
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
├── benchmarking                 # Benchmark scores from evaluation results
│   ├── README.md
│   ├── calculateBenchmarkScore.py
│   └── benchmarks/
│       ├── config/              # Configs (batchResultFile, questionSetFile, outputpath)
│       ├── b1                   # Example benchmark report JSON
│       └── b2                   # Example benchmark report JSON
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
> - **Calculate benchmark score** (aggregate evaluation results by section):
>   ```bash
>   python -m benchmarking.calculateBenchmarkScore --config benchmarking/benchmarks/config/b1.json
>   ```
>   Additional benchmark configs live under `benchmarking/benchmarks/config/` (for example `b2.json`). See `test/promptTester/README.md` for the evaluator and `benchmarking/README.md` for benchmark scoring.
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
