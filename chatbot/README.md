## Legal Data Setup

### Directory structure (after setup)

```text
chatbot/
├── README.md
├── centralActsDownloader.py
├── main.py
├── vectorDbSetup.py
└── legalos_rag/
    ├── README.md
    ├── __init__.py  
    ├── factsRetriever.py 
    ├── ragInvoker.py
    └── prompt
        ├── prompts.py
        └── promptSchema.py
```

---

### Step 1: Download Central Acts PDFs locally

This script **requires a mandatory named argument** specifying where PDFs should be stored.

**Recommended path (used throughout this project):**

```text
./factsDB/centralActPdfs/data
```

Run:

```bash
python -m chatbot.centralActsDownloader --outputDir ./factsDB/centralActPdfs/data
```

This will create (under legalos project root):

```
factsDB/
└── centralActPdfs/
    ├── data/               # Downloaded PDFs
    └── failed_pdfs.txt     # Download failures (if any)
```

---

### Step 2: Create Vector Database and Embed PDFs

This script **requires two mandatory named arguments**:

1. **PDF directory** (input)
2. **Vector DB directory** (output)

**Recommended paths:**

```text
PDFs : ./factsDB/centralActPdfs/data
DB   : ./vectorDB
```

Run:

```bash
python -m chatbot.vectorDbSetup \
  --pdfActsDirectory ./factsDB/centralActPdfs/data \
  --vectordbDirectory ./vectorDB
```

This will:

- Create a **local, on-disk Qdrant vector database**
- Embed all PDFs using **BAAI/bge-small-en**
- Store vectors persistently inside `./vectorDB`

---

## Local SLM Setup (Ollama)

We use **qwen2.5:3b-instruct** as the local Small Language Model (SLM).

### Step 1: Install Ollama

**macOS**

```bash
brew install ollama
```

**Windows**  
Download from:  
https://ollama.com/download

---

### Step 2: Start Ollama server

```bash
ollama serve
```

---

### Step 3: Pull the model (run once)

```bash
ollama pull qwen2.5:3b-instruct
```

---

## Run the RAG System

Once everything is set up:

The RAG CLI now reads **only a single JSON config file**, passed as the sole CLI argument.

**Config directory (from `legalos/`):**

```text
config/
  └─ rag_v1.json
       ├─ vectordbpath : \"./vectorDB\"
       └─ template     : \"You are a legal document reader...{format_instructions}...{facts}...{question}\"
```

**Run from the project root (`legalos/`):**

```bash
python -m chatbot.main --config ./config/rag_v1.json
```

---

## You're all set! ✅

**Ask legal questions. Get answers grounded strictly in law.**
