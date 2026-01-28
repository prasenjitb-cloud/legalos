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

## Legal Data Setup

### Step 1: Download Central Acts PDFs locally

This script **requires a mandatory named argument** specifying where PDFs should be stored.

**Recommended path (used throughout this project):**

```text
./stateActPdfs/data
```

Run:

```bash
python ragChatbot/stateActsDownloader.py --outputDir ./stateActPdfs/data
```

This will create:

```
stateActPdfs/
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
PDFs : ./stateActPdfs/data
DB   : ./DB
```

Run:

```bash
python ragChatbot/vectorSetup.py \
  --pdfActsDirectory ./stateActPdfs/data \
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
python ragChatbot/main.py --vectordbpath ./DB
```

---# legalos_rag Structure & Execution Flow

## 📂 `legalos_rag/` Directory

### `promptSchema.py`

Contains all prompt schemas used for structured outputs and validation.

### `prompts.py`

Stores all prompt templates in one place.

### `contextRetriever.py`

Contains functions to:

- Retrieve relevant context from the vector database
- Format the retrieved documents for model usage

### `chainInvoker.py`

Acts as the final execution layer. Responsible for:

- Invoking the LLM/SLM with all required inputs
- Managing prompt + context + query assembly
- Logging results in a structured and consistent way

---

## 🔄 Main File Workflow

The main file is intentionally kept lightweight and orchestration-only.

Steps:

1. Initialize LLM / SLM objects.
2. Retrieve relevant documents using `getContext()` from `contextRetriever.py` (given DB path and query).
3. Invoke the model using `chainInvoker()` by passing:
   - LLM object
   - Retrieved documents
   - User query
   - Model name (used for logging)

Example:

```python
result = chainInvoker(llm, retrieved_docs, query, SLM_MODEL_NAME)
```

This design allows easily plugging in multiple models for comparison after a single retrieval step.

---

## 📝 Logging (`rag_runs`)

For **task-3-prompt-engineering**, the `rag_runs` file is maintained as a **living README-style log**.

- All executions performed during this task are recorded in `rag_runs`.
- Each entry corresponds to a distinct **prompt iteration** during prompt tuning.
- Similar or duplicate outputs were intentionally removed.
- Only **representative and meaningful results** are retained to keep reviews faster and clearer.
- Every prompt change or refinement made in **task-3-prompt-engineering** is documented here.

This file serves as:

- an **iteration history** of prompt engineering experiments, and
- a **review artifact** to understand prompt evolution without re-running experiments.

---

## 🔀 Execution Flow

### ASCII Art Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main File Start                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Initialize LLM/SLM   │
                  │      Objects         │
                  └──────────┬───────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────┐
         │  getContext(db_path, query)               │
         │                                           │
         │  ┌─────────────────────────────────────┐  │
         │  │    contextRetriever.py              │  │
         │  │                                     │  │
         │  │  ┌───────────────────────────────┐  │  │
         │  │  │ Retrieve chunks from Vector   │  │  │
         │  │  │ Database                      │  │  │
         │  │  └───────────────┬───────────────┘  │  │
         │  │                  ▼                  │  │
         │  │  ┌───────────────────────────────┐  │  │
         │  │  │ Format documents for prompt   │  │  │
         │  │  └───────────────┬───────────────┘  │  │
         │  │                  ▼                  │  │
         │  │  ┌───────────────────────────────┐  │  │
         │  │  │ Return: retrieved_docs        │  │  │
         │  │  └───────────────────────────────┘  │  │
         │  └─────────────────────────────────────┘  │
         └───────────────────┬───────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│  chainInvoker(llm, retrieved_docs, query, model_name)         │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │               chainInvoker.py                            │ │
│  │                                                          │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │ Load prompt template (from prompts.py)             │  │ │
│  │  └─────────────────────┬──────────────────────────────┘  │ │
│  │                        ▼                                 │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │ Apply output schema (from promptSchema.py)         │  │ │
│  │  └─────────────────────┬──────────────────────────────┘  │ │
│  │                        ▼                                 │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │ Assemble: prompt + context + query                 │  │ │
│  │  └─────────────────────┬──────────────────────────────┘  │ │
│  │                        ▼                                 │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │ Invoke LLM/SLM                                     │  │ │
│  │  └─────────────────────┬──────────────────────────────┘  │ │
│  │                        ▼                                 │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │ Log results to rag_runs                            │  │ │
│  │  └─────────────────────┬──────────────────────────────┘  │ │
│  │                        ▼                                 │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │ Return: result                                     │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

## You're all set! ✅

**Ask legal questions. Get answers grounded strictly in law.**

```

```
