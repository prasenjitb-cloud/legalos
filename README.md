# Legalos ⚖️

**Your one-stop solution to all your legal problems.**

---

### FILE STRUCTURE-

```text
.
├── DB
│   ├── collection
│   │   └── central_acts
│   │
│   └── meta.json
│
├── centralActPdfs
│   ├── data
│   │   ├── 189008.pdf
│   │   ├── 189108.pdf
│   │   └── ...
│   │
│   └── failed_pdfs.txt
│
├── legalos_rag
│   ├── README.md
│   ├── factsRetriever.py
│   ├── promptSchema.py
│   ├── prompts.py
│   └── ragInvoker.py
│
├── ragChatbot
│   ├── README.md
│   ├── centralActsDownloader.py
│   ├── main.py
│   └── vectorDbSetup.py
│
└── requirements.txt
```

## Getting Started

### Prerequisites

Before running **Legalos**, make sure you have:

- **Python 3.11**
  > Python 3.11 works best with LangChain. Newer versions may cause compatibility issues.
- **pip** (or any Python package manager)

> **Running scripts:** This project uses the **`-m` module structure**. Run Python scripts from the **project root** (`legalos/`) with `python -m ragChatbot.<module>`, e.g. `python -m ragChatbot.main --vectordbpath ./DB`. See `ragChatbot/README.md` for full commands.

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
