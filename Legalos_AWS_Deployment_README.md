# Legalos -- AWS Deployment (Local RAG + Local SLM)

This document describes the complete deployment architecture and
infrastructure decisions for running **Legalos** on AWS using:

-   Local Qdrant (file-based)
-   HuggingFace embeddings (CPU-only)
-   Ollama (qwen2.5:3b-instruct)
-   S3 for vector DB backup
-   EC2 for inference runtime

------------------------------------------------------------------------

# 🧠 Architecture Overview

    Local Machine (Ingestion Phase)
        ↓
    PDF Download + Embedding Creation
        ↓
    Local Qdrant Vector DB (~350MB)
        ↓
    Upload vectorDB → S3 (Backup Storage)
        ↓
    EC2 (Inference Runtime)
        ↓
    Download vectorDB from S3
        ↓
    Run:
    - Qdrant (local mode)
    - HuggingFace query embeddings
    - Ollama SLM (3B model)
    - Legalos RAG pipeline

------------------------------------------------------------------------

# 🔥 Key Decisions & Why

## 1️⃣ Ingestion vs Inference Separation

**Chosen:**\
- Perform crawling + embedding locally.\
- Deploy inference-only stack on EC2.

**Why:**\
- Avoid heavy embedding computation on EC2.\
- Reduce AWS compute costs.\
- Faster deployment cycle.\
- Cleaner architecture separation.

------------------------------------------------------------------------

## 2️⃣ Vector Database Storage Strategy

**Chosen:**\
Upload `vectorDB/` to S3 and download to EC2 disk at runtime.

**Why:**\
- S3 is durable and cheap.\
- Qdrant cannot operate directly on S3.\
- Clean backup strategy.\
- Avoid recomputation.

------------------------------------------------------------------------

## 3️⃣ S3 Bucket Type

**Chosen:**\
General Purpose Bucket

**Why:**\
- No need for S3 Express.\
- No high TPS workload.\
- Just storing \~350MB artifacts.\
- Multi-AZ durability.

------------------------------------------------------------------------

## 4️⃣ EC2 Instance Type

**Chosen:**\
m7i-flex.large

**Specs:**\
- 2 vCPU\
- 8GB RAM

**Why:**\
- 3B SLM requires significant RAM.\
- Smaller instances would crash.\
- 8GB sufficient for CPU inference.
- Best option available among the free credit options.

------------------------------------------------------------------------

## 5️⃣ Disk Resize

Initial root volume:\
8GB → insufficient

**Solution:**\
Resize EBS volume to 20GB.

**Why:**\
- 3B model alone ≈ 3--4GB\
- HuggingFace + vectorDB + system files require buffer\
- Prevent "No space left on device"

------------------------------------------------------------------------

## 6️⃣ Embedding Backend

**Chosen:**\
HuggingFaceEmbeddings (CPU-only torch)

**Why:**\
- Already integrated in codebase\
- No need to refactor\
- CPU-only torch installed\
- No CUDA packages

------------------------------------------------------------------------

## 7️⃣ Model Runtime

**Chosen:**\
Ollama + qwen2.5:3b-instruct

**Why:**\
- Fully local\
- No API cost\
- Full RAG control\
- No data leaving instance

Tradeoff:\
- Slower on CPU\
- Limited by 2 vCPU

------------------------------------------------------------------------

# 📦 Runtime Stack (On EC2)

-   Ubuntu 22.04 LTS\
-   Python 3.11\
-   Virtual environment\
-   HuggingFace embeddings (CPU torch)\
-   Qdrant (local mode)\
-   Ollama (port 11434)\
-   Legalos RAG system

------------------------------------------------------------------------

# 🗂 Storage Layout

## S3:

-   `vectorDB/` (backup only)

## EC2:

-   `/legalos/vectorDB` (working copy)\
-   `~/.ollama/models` (SLM model)\
-   `~/.cache/huggingface` (embedding model cache)

------------------------------------------------------------------------

# 💰 Cost Model

If running 24/7:

-   m7i-flex.large ≈ \~\$85/month\
-   20GB EBS ≈ \~\$1--2/month\
-   S3 negligible

**Best practice:**\
Stop instance when not in use.

------------------------------------------------------------------------

# 🔒 Security Setup

-   IAM Role attached to EC2\
-   AmazonS3ReadOnlyAccess policy\
-   No access keys stored on server\
-   Security group restricts SSH to personal IP

------------------------------------------------------------------------

# ⚡ Performance Notes

-   3B model on CPU = moderate latency\
-   Retrieval (Qdrant) is fast\
-   Generation dominates response time

Optimization options: - Switch to 1.5B model\
- Reduce retrieved chunks (k)\
- Use API-based LLM

------------------------------------------------------------------------

# 🛑 Operational Best Practice

After each session:

1.  Stop EC2 instance\
2.  Do not terminate unless done permanently\
3.  S3 remains as persistent backup

------------------------------------------------------------------------

# 🧩 Final Deployment State

You now have:

-   Fully local RAG pipeline\
-   Cloud-hosted inference runtime\
-   Persistent vector database backup\
-   Clean separation of ingestion vs serving\
-   Cost-aware infrastructure setup

------------------------------------------------------------------------

# 🚀 Live Demo

## Deployment Status: ✅ Deployed

The Legalos RAG system is now live on AWS EC2 with a web interface!

### 🌐 Access URL
**[http://3.6.205.42:5000/](http://3.6.205.42:5000/)**

### 🧪 Try It Out

**Sample Question:**\
*"What is the objective of the Right to Information Act, 2005?"*

### ⚠️ Performance Note
Response time is longer due to:
-   CPU-only inference (no GPU)\
-   3B parameter model running on 2 vCPU\
-   HuggingFace embeddings + Qdrant retrieval + LLM generation

**Be patient** – it works! ⏳

------------------------------------------------------------------------

# 🎯 What's Running

-   **Flask web server** (port 5000)\
-   **Ollama SLM** (qwen2.5:3b-instruct)\
-   **Qdrant vectorDB** (local file-based mode)\
-   **HuggingFace embeddings** (CPU torch)\
-   **Full RAG pipeline** with citation extraction
