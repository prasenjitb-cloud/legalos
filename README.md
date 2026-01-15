# Legalos ⚖️

**Your one-stop solution to all your legal problems.**

---

## Getting Started

### Prerequisites

Before running **Legalos**, make sure you have:

- **Python 3.11**  
  > Python 3.11 works best with LangChain. Newer versions may cause issues.
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
#### These are for analytics. Putting LANGSMITH_TRACING_V2=true starts the tracing.
```env
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=LegalOs
```


---

## Legal Data Setup

### Step 1: Download Central Acts PDFs locally

Download Central Acts PDFs from India Code to your system:
```bash
python ragChatbot/stateActsDownloader.py
```

### Step 2: Create Vector Database and Embed PDFs

This will create a local Qdrant vector database and embed all downloaded PDFs.  
Embedding model used: **BAAI/bge-small-en**
```bash
python ragChatbot/vectorSetup.py
```

---

## Local SLM Setup (Ollama)

We use **qwen2.5:3b-instruct** as the local SLM.

### Step 1: Install Ollama

**macOS**
```bash
brew install ollama
```

**Windows**  
Download from: [https://ollama.com/download](https://ollama.com/download)

### Step 2: Start Ollama server
```bash
ollama serve
```

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

---

## You're all set! ✅

**Ask legal questions. Get answers grounded strictly in law.**