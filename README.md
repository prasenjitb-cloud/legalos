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
├── chatbot
│   ├── README.md
│   ├── centralActsDownloader.py
│   ├── main.py
│   ├── vectorDbSetup.py
│   └── legalos_rag
│       ├── __init__.py
│       ├── README.md
│       ├── factsRetriever.py
│       ├── promptSchema.py
│       ├── prompts.py
│       └── ragInvoker.py
│
└── requirements.txt
```

## Getting Started

### Prerequisites

Before running **Legalos**, make sure you have:

- **Python 3.11**
  > Python 3.11 works best with LangChain. Newer versions may cause compatibility issues.
- **pip** (or any Python package manager)

> **Running scripts:** This project uses the **`-m` module structure**. Run Python scripts from the **project root** (`legalos/`) with `python -m chatbot.<module>`, e.g. `python -m chatbot.main --vectordbpath ./DB`. 

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

## Environment Setup

### Create a `.env` file

```bash
touch .env
```

### Add the following variables to `.env`

> These are **optional** and used only for analytics/tracing.  
> The system works fully offline without them.

```env
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=LegalOs
```

---

## You're all set!
