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
│   └── rag_v1.json      # Config: vectordbpath, template, model.model_name, logging (logfile, exclude_*)
│
├── chatbot
│   ├── README.md
│   ├── centralActsDownloader.py
│   ├── main.py
│   ├── vectorDbSetup.py
│   └── legalos_rag
│       ├── __init__.py
│       ├── README.md
│       ├── factsRetriever.py
│       ├── logger.py
│       ├── ragInvoker.py
│       └── prompt
│           ├── prompts.py
│           └── promptSchema.py
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
> ```bash
> python -m chatbot.main --config ./config/rag_v1.json
> ```

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
