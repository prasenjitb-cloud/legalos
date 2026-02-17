# Legalos AWS Deployment Guide

Complete step-by-step guide to deploy Legalos RAG system on AWS EC2 with terminal commands.

---

## 📑 Table of Contents

1. [Prerequisites](#-prerequisites)
2. [Architecture Overview](#️-architecture-overview)
3. [Step 1: Create EC2 Instance](#-step-1-create-ec2-instance)
4. [Step 2: Connect to EC2 Instance](#-step-2-connect-to-ec2-instance)
5. [Step 3: Setup EC2 Environment](#-step-3-setup-ec2-environment)
6. [Step 4: Upload Project Files](#-step-4-upload-project-files-to-ec2)
7. [Step 5: Install Python Dependencies](#️-step-5-install-python-dependencies-on-ec2)
8. [Step 6: Verify Setup](#-step-6-verify-setup)
9. [Step 7: Open Port 5000](#-step-7-open-port-5000-in-security-group)
10. [Step 8: Run Legalos RAG System](#-step-8-run-legalos-rag-system)
11. [Step 9: Run as Background Service](#-step-9-run-as-background-service-optional)
12. [Troubleshooting](#-troubleshooting)
13. [Managing Your EC2 Instance](#-managing-your-ec2-instance)
14. [Cost Estimation](#-cost-estimation)
15. [Quick Reference Commands](#-quick-reference-commands)

---

## 📋 Prerequisites

- AWS account with appropriate permissions
- Local machine with SSH client
- Legalos project with vectorDB already created locally
- Terminal access

**Note**: No environment variables or API keys needed - system works fully offline!

---

## 🏗️ Architecture Overview

```
Local Machine
    ↓
    PDF Download + Embedding Creation
    ↓
    Local Qdrant Vector DB (~350MB)
    ↓
    Upload to EC2 via SCP/RSYNC
    ↓
EC2 Instance
    ↓
    - Qdrant (local file-based)
    - HuggingFace embeddings (CPU)
    - Ollama (qwen2.5:3b-instruct)
    - Legalos RAG pipeline
    - Optional: Flask web interface
```

---

## 🚀 Step 1: Create EC2 Instance

### 1.1 Launch EC2 Instance via AWS Console

1. **Go to EC2 Dashboard**
   - Navigate to AWS Console → EC2 → Launch Instance

2. **Configure Instance**
   - **Name**: `legalos-rag-server`
   - **AMI**: Ubuntu Server 22.04 LTS (64-bit x86)
   - **Instance Type**: `m7i-flex.large` (2 vCPU, 8GB RAM)
   - **Key Pair**: Create new or select existing key pair (save `.pem` file)
   - **Network Settings**: Allow SSH traffic from your IP
   - **Storage**: **20 GB gp3 root volume** (⚠️ Important: Change from default 8GB to 20GB)

**Note:** We'll configure port 5000 access in Step 7 after setup is complete.

3. **Launch Instance**

### 1.2 Save Key Pair Locally

After downloading the `.pem` file:

```bash
# Move to .ssh directory and set permissions
mv ~/Downloads/legalos-key.pem ~/.ssh/
chmod 400 ~/.ssh/legalos-key.pem
```

### 1.3 Get Instance Public IP

```bash
# From AWS Console, note the Public IPv4 address
# Example: 3.6.205.42
```

---

## 🔌 Step 2: Connect to EC2 Instance

```bash
# SSH into your instance
ssh -i ~/.ssh/legalos-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP>

# Example:
# ssh -i ~/.ssh/legalos-key.pem ubuntu@3.6.205.42

# type yes
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
```

**Note**: Replace `<YOUR_EC2_PUBLIC_IP>` with your actual EC2 public IP address.

---

## 📦 Step 3: Setup EC2 Environment

Run these commands on your EC2 instance:

### 3.1 Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### 3.2 Install Python 3.11

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

## ⚠ Service Restart Prompt During Installation

While installing Python 3.11 or running `apt upgrade`, you may see a screen like:

> **Daemons using outdated libraries**  
> Which services should be restarted?

This is normal.

### ✅ What to do:

1. Press **Enter** (keep default selected services)
2. Press **Tab** to highlight `<Ok>`
3. Press **Enter** again

Installation will continue automatically.

### 3.3 Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3.4 Pull Ollama Model

```bash
# This will download the 3B parameter model (~2GB)
ollama pull qwen2.5:3b-instruct
```

### 3.5 Verify Ollama is Running

```bash
# Check Ollama service
systemctl status ollama
```

**If you see:**
```
● ollama.service - Ollama Service
     Active: active (running)
```
✅ Ollama is running!

**To exit the status screen:**
- Press: `q`

**If Ollama is NOT running:**

Start it:
```bash
sudo systemctl start ollama
```

Then check again:
```bash
systemctl status ollama
```

**Test the Model:**

```bash
ollama run qwen2.5:3b-instruct "Hello"
```

---

## 📤 Step 4: Upload Project Files to EC2

**Run these commands on your LOCAL machine** (in a new terminal):

### 4.1 Upload Files via SCP

**Run these commands on your LOCAL machine:**

```bash
# Navigate to your legalos directory
cd /path/to/your/legalos

# Create legalos directory on EC2
ssh -i ~/.ssh/legalos-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP> "mkdir -p ~/legalos"

# Upload required files and folders
scp -i ~/.ssh/legalos-key.pem -r chatbot config vectorDB browserRun.py aws_deployment/requirements.txt ubuntu@<EC2_PUBLIC_IP>:~/legalos/

```
**Files being uploaded:**
- `chatbot/` - RAG system code
- `config/` - Configuration files (rag_v1.json)
- `vectorDB/` - Pre-created vector database (~350MB)
- `browserRun.py` - Flask API server
- `aws_deployment/requirements.txt` - Python dependencies

---

## 🛠️ Step 5: Install Python Dependencies on EC2

**Back on your EC2 instance** (SSH terminal):

```bash
# Navigate to project directory
cd ~/legalos

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## ✅ Step 6: Verify Setup

```bash
# Check Python packages
pip list | grep -E "langchain|qdrant|torch|ollama"

# Check vectorDB
ls -lh ~/legalos/vectorDB

# Check config
cat ~/legalos/config/rag_v1.json
```

---

## 🔓 Step 7: Open Port 5000 in Security Group

To allow external access to your Flask API running on port 5000, configure the Security Group.

### 7.1 Go to EC2 Console

1. Navigate to **EC2 Dashboard**
2. Click **Instances**
3. Select your running instance

### 7.2 Open Security Group

1. Scroll down to the **Security** tab
2. Click the linked **Security Group** (e.g., sg-xxxxxx)
3. Go to the **Inbound rules** tab
4. Click **Edit inbound rules**

### 7.3 Add Rule for Port 5000

Click **Add rule** and configure:

| Field | Value |
|-------|-------|
| **Type** | Custom TCP |
| **Port Range** | 5000 |
| **Source** | 0.0.0.0/0 (public access) |
| **Description** | Flask API access |

Then click **Save rules**.

### 🔐 Recommended (Safer Option)

If this is only for personal testing, restrict access to your IP:

| Field | Value |
|-------|-------|
| **Type** | Custom TCP |
| **Port Range** | 5000 |
| **Source** | My IP |
| **Description** | Flask API access (restricted) |

This allows only your current IP address to access port 5000.

**Note:** If your IP changes (e.g., after reconnecting to WiFi), you'll need to update this rule.

---

## 🎯 Step 8: Run Legalos RAG System

### Option A: Flask API (Recommended for Browser Access)

Run the browserRun.py Flask API:

```bash
# Make sure you're in the legalos directory with venv activated
cd ~/legalos
source venv/bin/activate

# Run the Flask API server
python browserRun.py
```

**Access the API:**

The server will run on `http://<YOUR_EC2_PUBLIC_IP>:5000`

**Test the /chat endpoint:**

```bash
# Using curl (from any terminal)
curl -X POST http://<YOUR_EC2_PUBLIC_IP>:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the objective of the Right to Information Act, 2005?"}'

# Or using GET
curl "http://<YOUR_EC2_PUBLIC_IP>:5000/chat?message=What%20is%20the%20objective%20of%20the%20Right%20to%20Information%20Act%2C%202005%3F"
```

**API Response Format:**

```json
{
  "answer_found": true,
  "reply": "The objective of the Right to Information Act...",
  "citations": [
    {
      "page": "Page 1",
      "quote": "The Act aims to..."
    }
  ]
}
```

### Option B: Command-Line Interface (Optional)

```bash
# Make sure you're in the legalos directory with venv activated
cd ~/legalos
source venv/bin/activate

# Run the RAG system
python -m chatbot.main --config config/rag_v1.json
```

**Test with a query:**
```
Ask a legal question (type 'exit' to quit): What is the objective of the Right to Information Act, 2005?
```

---

## 🔄 Step 9: Run as Background Service (Optional)

To keep the service running even after disconnecting SSH:

### Using screen

```bash
# Install screen
sudo apt install -y screen

# Start a new screen session
screen -S legalos

# Run your application
cd ~/legalos
source venv/bin/activate
python browserRun.py

# Detach from screen: Press Ctrl+A, then D

# Reattach to screen later:
screen -r legalos
```

---

## 🔧 Troubleshooting

### Port 5000 Not Accessible

```bash
# Check if Flask is running
ps aux | grep python

# Check if port is listening
sudo netstat -tlnp | grep 5000

# Verify Security Group allows port 5000 from your IP
```

### Out of Memory

```bash
# Check memory usage
free -h

# If OOM, consider switching to smaller model or larger instance
```

### Ollama Not Responding

```bash
# Restart Ollama service
sudo systemctl restart ollama

# Check Ollama status
systemctl status ollama

# Test Ollama
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:3b-instruct",
  "prompt": "Hello"
}'
```

### VectorDB Issues

```bash
# Verify vectorDB exists
ls -lh ~/legalos/vectorDB

# Check directory permissions
chmod -R 755 ~/legalos/vectorDB
```

---

## 🛑 Managing Your EC2 Instance

### Stop Instance (Save Costs)

When you're not using the system, stop the instance to avoid charges:

**Via AWS Console:**
1. Go to EC2 Dashboard → Instances
2. Select your instance
3. Instance state → Stop instance

**Via AWS CLI:**
```bash
aws ec2 stop-instances --instance-ids <YOUR_INSTANCE_ID>
```

### Start Instance

**Via AWS Console:**
1. Go to EC2 Dashboard → Instances
2. Select your instance
3. Instance state → Start instance

**Via AWS CLI:**
```bash
aws ec2 start-instances --instance-ids <YOUR_INSTANCE_ID>
```

**⚠️ Important:** Public IP may change after stop/start. Consider using an Elastic IP for a persistent IP address.

### Terminate Instance (Permanent Deletion)

**Only do this when you're completely done:**

1. Go to EC2 Dashboard → Instances
2. Select your instance
3. Instance state → Terminate instance

**Warning:** This permanently deletes the instance and all data on it!

---

## 📊 Performance Notes

- **Response Time**: 10-30 seconds per query (CPU-only inference)
- **Concurrent Users**: Limited by 2 vCPU and 8GB RAM
- **Optimization Options**:
  - Switch to smaller model (1.5B)
  - Reduce retrieved chunks (modify `factsRetriever.py`)
  - Use GPU-enabled instance (g4dn family)

---

## 🔒 Security Best Practices

1. **SSH Access**
   - Restrict Security Group to your IP only
   - Use key-based authentication (no passwords)
   - Keep `.pem` file secure (chmod 400)

2. **API Access**
   - Restrict port 5000 to trusted IPs only in Security Group
   - Consider adding authentication to Flask app for production
   - Use HTTPS with Nginx reverse proxy (production)

3. **System Updates**
   ```bash
   # Regular system updates
   sudo apt update && sudo apt upgrade -y
   ```

4. **Instance Management**
   - Stop instance when not in use
   - Never share your `.pem` file
   - Regularly review Security Group rules

---

## 💰 Cost Estimation

**m7i-flex.large (2 vCPU, 8GB RAM)**
- On-Demand: ~$0.12/hour = ~$86/month (24/7)
- **Recommendation**: Stop instance when not in use

**20GB EBS gp3 Storage**
- ~$1.60/month

**Total (if running 24/7)**: ~$88/month

**Cost-Saving Tips**:
- Stop instance after each use
- Use AWS Free Tier credits if available
- Consider Reserved Instances for long-term use

---

## 🔒 Security Best Practices

1. **SSH Access**
   - Restrict Security Group to your IP only
   - Use key-based authentication (no passwords)
   - Keep `.pem` file secure (chmod 400)

2. **API Access**
   - Restrict port 5000 to trusted IPs only in Security Group
   - Consider adding authentication to Flask app for production
   - Use HTTPS with Nginx reverse proxy (production)

3. **System Updates**
   ```bash
   # Regular system updates
   sudo apt update && sudo apt upgrade -y
   ```

---

## 🎉 You're Done!

Your Legalos RAG system is now running on AWS EC2!

**Access your deployment:**
- API: `http://<YOUR_EC2_PUBLIC_IP>:5000/chat`
- CLI: SSH into EC2 and run `python -m chatbot.main --config config/rag_v1.json`

**Example API Query:**
```bash
curl -X POST http://<YOUR_EC2_PUBLIC_IP>:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the objective of the Right to Information Act, 2005?"}'
```

---

## 📝 Quick Reference Commands

```bash
# Connect to EC2
ssh -i ~/.ssh/legalos-key.pem ubuntu@<EC2_IP>

# Activate environment
cd ~/legalos && source venv/bin/activate

# Run Flask API
python browserRun.py

# Run CLI (alternative)
python -m chatbot.main --config config/rag_v1.json

# Test API
curl -X POST http://<EC2_IP>:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Your question here"}'

# View logs (if using systemd)
sudo journalctl -u legalos -f

# Stop service
sudo systemctl stop legalos

# Update files from local (run from your legalos directory)
rsync -avz -e "ssh -i ~/.ssh/legalos-key.pem" \
  chatbot/ config/ vectorDB/ browserRun.py aws_deployment/requirements.txt \
  ubuntu@<EC2_IP>:~/legalos/
```

---

## 📚 Additional Resources

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Ollama Documentation](https://ollama.com/docs)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [LangChain Documentation](https://python.langchain.com/)
