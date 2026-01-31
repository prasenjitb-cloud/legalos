## Legal Data Setup

### Directory structure (after setup)

```text
ragChatbot/
├── README.md
├── centralActsDownloader.py
├── main.py
└── vectorDbSetup.py
```

---

### Step 1: Download Central Acts PDFs locally

This script **requires a mandatory named argument** specifying where PDFs should be stored.

**Recommended path (used throughout this project):**

```text
./centralActPdfs/data
```

Run:

```bash
python -m ragChatbot.centralActsDownloader --outputDir ./centralActPdfs/data
```

This will create:

```
centralActPdfs/
├── data/               # Downloaded PDFs
├── failed_pdfs.txt     # Download failures (if any)
```

---

### Step 2: Create Vector Database and Embed PDFs

This script **requires two mandatory named arguments**:

1. **PDF directory** (input)
2. **Vector DB directory** (output)

**Recommended paths:**

```text
PDFs : ./centralActPdfs/data
DB   : ./DB
```

Run:

```bash
python -m ragChatbot.vectorDbSetup \
  --pdfActsDirectory ./centralActPdfs/data \
  --vectordbDirectory ./DB
```

This will:

- Create a **local, on-disk Qdrant vector database**
- Embed all PDFs using **BAAI/bge-small-en**
- Store vectors persistently inside `./DB`

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

This script **requires one mandatory named argument**:

1. **Vector DB directory**

**Recommended paths:**

```text
DB   : ./DB
```

```bash
python -m ragChatbot.main --vectordbpath ./DB
```

---

## You're all set! ✅

**Ask legal questions. Get answers grounded strictly in law.**
