# Legalos ⚖️

**Your one-stop solution to all your legal problems.**

---

## Getting Started

### Prerequisites

Before running **Legalos**, make sure you have:

- **Python 3.11**  
  > Python 3.11 works best with LangChain. Newer versions may cause compatibility issues.
- **pip** (or any Python package manager)

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/legalos.git
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

## Legal Data Setup

### Step 1: Download Central Acts PDFs locally

This script **requires a mandatory argument** specifying where PDFs should be stored.

**Recommended path (used throughout this project):**
```text
./stateActPdfs/data
```

Run:
```bash
python ragChatbot/stateActsDownloader.py ./stateActPdfs/data
```

This will create:
```
stateActPdfs/
├── data/               # Downloaded PDFs
├── failed_pdfs.txt     # Download failures (if any)
```

---

### Step 2: Create Vector Database and Embed PDFs

This script **requires two mandatory arguments**:
1. **PDF directory** (input)
2. **Vector DB directory** (output)

**Recommended paths:**
```text
PDFs : ./stateActPdfs/data
DB   : ./DB
```

Run:
```bash
python ragChatbot/vectorSetup.py ./stateActPdfs/data ./DB
```

This will:
- Create a **local, on-disk Qdrant vector database**
- Embed all PDFs using **BAAI/bge-small-en**
- Store vectors persistently inside `./DB`

> ⚠️ **Important:**  
> If you change the **DB path here**, you must update the same path in  
> **`ragChatbot/slmSetup.py`**, as it currently assumes the vector database  
> is located at `./DB`.

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

```bash
python ragChatbot/slmSetup.py
```

> ⚠️ This script assumes the vector database exists at:
```text
./DB
```
If you changed the DB location during ingestion, update it here as well.

---

## You're all set! ✅

**Ask legal questions. Get answers grounded strictly in law.**

