# Legalos AWS Deployment Guide

Complete step-by-step guide to deploy Legalos RAG system on AWS EC2 with terminal commands.

## 🌐 Live Demo

**Demo URL**: `http://13.235.62.240:5000/chat/`

⚠️ **Note**: The public IP changes every time the server is stopped and restarted, so this link may not be working when you view it.

**Test with curl:**
```bash
curl -X POST http://13.235.62.240:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the objective of the Right to Information Act, 2005?"}'
```

or

**GET Test**: "http://13.235.62.240:5000/chat?message=What%20is%20the%20objective%20of%20the%20Right%20to%20Information%20Act,%202005?"

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

### 1.1 Login to AWS Console

1. **Open your web browser**
2. **Go to**: https://console.aws.amazon.com/
3. **Sign in** with your AWS account credentials

### 1.2 Navigate to EC2

1. **In the AWS Console search bar** (top), type: `EC2`
2. **Click on**: `EC2` (Virtual Servers in the Cloud)
3. **You will see**: EC2 Dashboard page

### 1.3 Launch New Instance

1. **Click the orange button**: `Launch instance`
2. **You will see**: "Launch an instance" page

### 1.4 Configure Instance Settings

Fill in the following fields **EXACTLY as shown**:

#### Name and Tags
- **Name**: Type `legalos-rag-server`

#### Application and OS Images (AMI)
- **Click on**: `Ubuntu` (in Quick Start)
- **Select**: `Ubuntu Server 22.04 LTS (HVM), SSD Volume Type`
- **Architecture**: `64-bit (x86)`

#### Instance Type
- **Click the dropdown** next to "Instance type"
- **Search for**: `m7i-flex.large`
- **Select**: `m7i-flex.large` (2 vCPU, 8 GiB Memory)

#### Key Pair (login)
**If you don't have a key pair:**
1. **Click**: `Create new key pair`
2. **Key pair name**: Type `legalos-key`
3. **Key pair type**: Select `RSA`
4. **Private key file format**: Select `.pem`
5. **Click**: `Create key pair`
6. **Your browser will download**: `legalos-key.pem` (save it!)

**If you already have a key pair:**
- **Select your existing key pair** from the dropdown

#### Network Settings
1. **Check the box**: `Allow SSH traffic from` → Select `My IP`
2. ⚠️ **Note**: We'll add port 5000 later in Step 7

#### Configure Storage
1. **Look for**: "Configure storage" section
2. **You'll see**: `8 GiB gp3 Root volume` (default)
3. **⚠️ IMPORTANT - Click on**: `8 GiB` to edit it
4. **Change to**: `20` (GiB)
5. **Keep**: `gp3` as the volume type

#### Summary
- Review your settings on the right side panel
- **You should see**:
  - Number of instances: 1
  - Instance type: m7i-flex.large
  - Storage: 20 GiB

### 1.5 Launch Instance

1. **Scroll down** to the bottom
2. **Click the orange button**: `Launch instance`
3. **You will see**: "Successfully initiated launch of instance"
4. **Click**: `View all instances`

### 1.6 Wait for Instance to Start

1. **You will see**: Your instance in the list with `legalos-rag-server` name
2. **Wait until**:
   - **Instance State**: Changes from `Pending` to `Running` (green dot)
   - **Status check**: Shows `2/2 checks passed` (~2 minutes)

### 1.7 Get Your Instance Public IP

1. **Click on your instance** (`legalos-rag-server`)
2. **Look for**: "Public IPv4 address" in the Details section
3. **Copy this IP address** (Example: `3.6.205.42`)
4. **Save it somewhere** - you'll need it throughout this guide

### 1.8 Save Your Key Pair

**📍 On YOUR LOCAL MACHINE - Open Terminal:**

```bash
# Move the downloaded .pem file to .ssh directory
mv ~/Downloads/legalos-key.pem ~/.ssh/

# Set correct permissions (required for SSH)
chmod 400 ~/.ssh/legalos-key.pem

# Verify it's there
ls -l ~/.ssh/legalos-key.pem
```

**You should see**: `-r--------  1 yourusername  staff  ... legalos-key.pem`

---

## 🔌 Step 2: Connect to EC2 Instance

### 2.1 Open Terminal on Your Local Machine

**📍 On YOUR LOCAL MACHINE:**

- **Mac**: Press `Cmd + Space`, type `Terminal`, press Enter
- **Windows**: Use Git Bash or WSL
- **Linux**: Press `Ctrl + Alt + T`

### 2.2 SSH into EC2

**📍 In YOUR LOCAL MACHINE Terminal, type:**

```bash
ssh -i ~/.ssh/legalos-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP>
```

**Replace** `<YOUR_EC2_PUBLIC_IP>` with your actual IP from Step 1.7

**Example:**
```bash
ssh -i ~/.ssh/legalos-key.pem ubuntu@3.6.205.42
```

### 2.3 Confirm Connection

**You will see a message**:
```
The authenticity of host '3.6.205.42' can't be established.
ECDSA key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

**Type**: `yes` and press **Enter**

### 2.4 You're Connected!

**You should now see**:
```
Welcome to Ubuntu 22.04 LTS
...
ubuntu@ip-xxx-xx-xx-xx:~$
```

✅ **You are now inside your EC2 instance!**

**⚠️ From now until Step 4, all commands run on EC2 (SSH terminal)**

---

## 📦 Step 3: Setup EC2 Environment

**📍 ALL commands in Step 3 run in your EC2 SSH terminal**

### 3.1 Update System

**In your EC2 terminal, copy and paste:**

```bash
sudo apt update && sudo apt upgrade -y
```

**Press Enter**

This will take **2-3 minutes**. You'll see packages being updated.


## ⚠️ Service Restart Prompt During Installation

While installing Python 3.11 or running `apt upgrade`, you may see a screen like:

> **Daemons using outdated libraries**  
> Which services should be restarted?

This is normal.

### ✅ What to do:

1. Press **Enter** (keep default selected services)
2. Press **Tab** to highlight `<Ok>`
3. Press **Enter** again

Installation will continue automatically.

### 3.2 Install Python 3.11

**In your EC2 terminal, copy and paste these commands ONE BY ONE:**

```bash
sudo apt install -y software-properties-common
```
Press **Enter**, wait for it to complete (~30 seconds)

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
```
Press **Enter**, wait (~10 seconds)

```bash
sudo apt update
```
Press **Enter**, wait (~30 seconds)

```bash
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```
Press **Enter**, this will take **~2 minutes**

### 3.3 Install Ollama

**In your EC2 terminal:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Press **Enter**

This will take **~1 minute**. You'll see installation progress.

**When complete, you should see**:
```
>>> Ollama is installed!
```

### 3.4 Pull Ollama Model

**In your EC2 terminal:**

```bash
ollama pull qwen2.5:3b-instruct
```

Press **Enter**

⏳ **This downloads ~2GB and takes 3-5 minutes**

You'll see:
```
pulling manifest
pulling ...
...
success
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

Press **Enter**

**You should see**: The model generates a response to "Hello"

**To exit**: Press `Ctrl+D`

✅ **Ollama is working!**

---

## 📤 Step 4: Upload Project Files to EC2

### 4.1 Open New Terminal on Your Local Machine

**📍 IMPORTANT: Keep your EC2 SSH terminal open, but open a NEW terminal window**

**On your LOCAL machine:**
- **Mac**: Press `Cmd + N` for new terminal window
- **Windows**: Open new Git Bash window
- **Linux**: Press `Ctrl + Shift + N`

### 4.2 Navigate to Your Legalos Directory

**📍 In your NEW LOCAL terminal (NOT the EC2 one):**

```bash
cd /path/to/your/legalos
```

**Replace** `/path/to/your/legalos` with your actual path

**Example:**
```bash
cd ~/BTP/legalos
```

**Verify you're in the right place:**
```bash
ls
```

**You should see**:
- `chatbot/`
- `config/`
- `vectorDB/`
- `deployRun.py`
- `aws_deployment/`

### 4.3 Create Directory on EC2

**📍 Still in your LOCAL terminal:**

```bash
ssh -i ~/.ssh/legalos-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP> "mkdir -p ~/legalos"
```

**Replace** `<YOUR_EC2_PUBLIC_IP>` with your IP

**Example:**
```bash
ssh -i ~/.ssh/legalos-key.pem ubuntu@3.6.205.42 "mkdir -p ~/legalos"
```

Press **Enter** - this runs quickly

### 4.4 Upload Files to EC2

**📍 Still in your LOCAL terminal, in your legalos directory:**

```bash
scp -i ~/.ssh/legalos-key.pem -r chatbot config vectorDB deployRun.py aws_deployment/requirements.txt ubuntu@<YOUR_EC2_PUBLIC_IP>:~/legalos/
```

**Replace** `<YOUR_EC2_PUBLIC_IP>` with your IP

**Example:**
```bash
scp -i ~/.ssh/legalos-key.pem -r chatbot config vectorDB deployRun.py aws_deployment/requirements.txt ubuntu@3.6.205.42:~/legalos/
```

Press **Enter**

⏳ **This uploads ~350MB and takes 2-5 minutes depending on your internet speed**

You'll see:
```
chatbot/...
config/...
vectorDB/...
deployRun.py
requirements.txt
```

✅ **When complete, all files are on EC2!**

**Files uploaded:**
- `chatbot/` - RAG system code
- `config/` - Configuration files (rag_v1.json)
- `vectorDB/` - Pre-created vector database (~350MB)
- `deployRun.py` - Flask API server
- `aws_deployment/requirements.txt` - Python dependencies

---

## 🛠️ Step 5: Install Python Dependencies on EC2

### 5.1 Go Back to EC2 Terminal

**📍 Switch to your EC2 SSH terminal window** (the first terminal you opened in Step 2)

You should still see: `ubuntu@ip-xxx-xx-xx-xx:~$`

### 5.2 Navigate to Project Directory

**In your EC2 terminal:**

```bash
cd ~/legalos
```

Press **Enter**

**Verify files are there:**
```bash
ls
```

**You should see**:
```
chatbot  config  deployRun.py  requirements.txt  vectorDB
```

### 5.3 Create Virtual Environment

**In your EC2 terminal:**

```bash
python3.11 -m venv venv
```

Press **Enter** - takes ~30 seconds

### 5.4 Activate Virtual Environment

**In your EC2 terminal:**

```bash
source venv/bin/activate
```

Press **Enter**

**Your prompt should change to**:
```
(venv) ubuntu@ip-xxx-xx-xx-xx:~/legalos$
```

✅ **The `(venv)` means it's activated!**

### 5.5 Upgrade Pip

**In your EC2 terminal:**

```bash
pip install --upgrade pip
```

Press **Enter** - takes ~10 seconds

### 5.6 Install Dependencies

**In your EC2 terminal:**

```bash
pip install -r requirements.txt
```

Press **Enter**

⏳ **This takes 10-15 minutes** (installing PyTorch, sentence-transformers, etc.)

You'll see:
```
Collecting flask...
Downloading...
Installing...
Successfully installed ...
```

✅ **When you see your prompt again, installation is complete!**

---

## ✅ Step 6: Verify Setup

**📍 In your EC2 terminal (with venv activated):**

### 6.1 Check Python Packages

```bash
pip list | grep -E "langchain|qdrant|torch|ollama"
```

Press **Enter**

**You should see**:
```
langchain-community    ...
langchain-core         ...
langchain-huggingface  ...
langchain-ollama       ...
langchain-qdrant       ...
qdrant-client          ...
torch                  ...
```

### 6.2 Check VectorDB

```bash
ls -lh ~/legalos/vectorDB
```

Press **Enter**

**You should see**:
```
total ...
drwxr-xr-x ... collection
-rw-r--r-- ... meta.json
```

### 6.3 Check Config

```bash
cat ~/legalos/config/rag_v1.json
```

Press **Enter**

**You should see**: JSON configuration file content

✅ **Everything is set up correctly!**

---

## 🔓 Step 7: Open Port 5000 in Security Group

**📍 This step is done in your web browser, not terminal**

### 7.1 Go to EC2 Console in Browser

1. **Open browser** and go to: https://console.aws.amazon.com/ec2/
2. **You should see**: EC2 Dashboard

### 7.2 Navigate to Your Instance

1. **Click on**: `Instances (running)` in the left sidebar
2. **You should see**: Your `legalos-rag-server` instance
3. **Click on the checkbox** next to your instance (to select it)
4. **Click on the instance name** (`legalos-rag-server`) to open details

### 7.3 Find Security Group

1. **Scroll down** to the bottom half of the page
2. **Click on the tab**: `Security`
3. **You will see**: "Security groups" section
4. **Click on the blue link**: `sg-xxxxxxxxx` (your security group ID)
   - It will open in a new page

### 7.4 Edit Inbound Rules

1. **You are now on**: Security Groups page
2. **Click on the tab**: `Inbound rules` (should already be selected)
3. **You should see**: At least one rule for SSH (port 22)
4. **Click the button**: `Edit inbound rules` (top right)

### 7.5 Add Port 5000 Rule

1. **You are now on**: "Edit inbound rules" page
2. **Click the button**: `Add rule` (bottom left)
3. **A new row appears** - Fill it in:

**Type dropdown:**
- Click on `Type` dropdown
- Scroll down and select: `Custom TCP`

**Port range:**
- Click on `Port range` field
- Type: `5000`

**Source:**
- Click on `Source` dropdown
- Select: `My IP` (recommended for testing)
- OR select: `Anywhere-IPv4` (for public access)
- Your IP will be filled automatically

**Description (optional):**
- Click on `Description` field
- Type: `Flask API access`

4. **Click the orange button**: `Save rules` (bottom right)

✅ **You should see**: "Successfully modified rules"

### 7.6 Verify Rule is Added

**You should now see** in your Inbound rules table:
```
Port range: 5000
Source: Your IP / 0.0.0.0/0
```

**Note:** If you selected "My IP" and your IP changes later (after reconnecting WiFi), you'll need to repeat Step 7 to update the rule.

---

## 🎯 Step 8: Run Legalos RAG System

### 8.1 Start the Flask API Server

**📍 In your EC2 terminal (make sure you're in ~/legalos with venv activated):**

**Verify you're in the right place:**
```bash
pwd
```

**Should show**: `/home/ubuntu/legalos`

**Verify venv is activated (you should see `(venv)` in your prompt):**

If not activated, run:
```bash
source venv/bin/activate
```

**Now start the server:**
```bash
python deployRun.py
```

Press **Enter**

**You should see**:
```
 * Serving Flask app 'deployRun'
 * Debug mode: off
WARNING: This is a development server. Do not use it in production.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://xxx.xxx.xxx.xxx:5000
Press CTRL+C to quit
```

✅ **Server is running!**

**⚠️ DO NOT PRESS CTRL+C** - leave this terminal running

### 8.2 Test the API

**📍 Open a NEW terminal window on your LOCAL machine**

**Test with curl:**

```bash
curl -X POST http://<YOUR_EC2_PUBLIC_IP>:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the objective of the Right to Information Act, 2005?"}'
```

**Replace** `<YOUR_EC2_PUBLIC_IP>` with your actual IP

**Example:**
```bash
curl -X POST http://3.6.205.42:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the objective of the Right to Information Act, 2005?"}'
```

Press **Enter**

⏳ **This will take 10-30 seconds** (first query is slower)

**You should see** a JSON response:

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

✅ **Your API is working!**

### 8.3 Access from Browser

**You can also test in your browser:**

1. **Open browser**
2. **Go to**: `http://<YOUR_EC2_PUBLIC_IP>:5000/chat?message=hello`

**Replace** `<YOUR_EC2_PUBLIC_IP>` with your IP

**Example**: http://3.6.205.42:5000/chat?message=hello

**You should see**: JSON response in browser

### 8.4 Stop the Server

**To stop the Flask server:**

1. **Go to your EC2 terminal** (where Flask is running)
2. **Press**: `Ctrl+C`

**You should see**:
```
^C
(venv) ubuntu@ip-xxx-xx-xx-xx:~/legalos$
```

Server is stopped.

---

## 🔄 Step 9: Run as Background Service (Optional)

**This step keeps your Flask API running even after you close your SSH connection**

**📍 In your EC2 terminal:**

### 9.1 Install Screen

```bash
sudo apt install -y screen
```

Press **Enter** - takes ~20 seconds

### 9.2 Start Screen Session

```bash
screen -S legalos
```

Press **Enter**

**Your screen will clear** - this is normal!

### 9.3 Navigate and Run Server

**You're now inside a screen session:**

```bash
cd ~/legalos
source venv/bin/activate
python deployRun.py
```

**You should see** Flask server starting

✅ **Server is running in screen!**

### 9.4 Detach from Screen

**To leave the server running and return to your normal terminal:**

1. **Press and hold**: `Ctrl + A`
2. **Then press**: `D` 

**You should see**:
```
[detached from 12345.legalos]
ubuntu@ip-xxx-xx-xx-xx:~$
```

✅ **Server is still running in background!**

### 9.5 Test It's Working

**In a new terminal on your LOCAL machine:**

```bash
curl -X POST http://<YOUR_EC2_PUBLIC_IP>:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'
```

**Should get a response** - server is running!

### 9.6 Reattach to Screen (Later)

**When you want to see the server logs again:**

```bash
screen -r legalos
```

**To detach again**: Press `Ctrl+A`, then `D`

### 9.7 Stop the Server

**To stop the server running in screen:**

1. **Reattach**: `screen -r legalos`
2. **Press**: `Ctrl+C` to stop Flask
3. **Type**: `exit` to close the screen session

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

**When you're done using Legalos for the day, STOP the instance to avoid charges**

**📍 In your web browser:**

1. **Go to**: https://console.aws.amazon.com/ec2/
2. **Click on**: `Instances (running)` in left sidebar
3. **Find**: Your `legalos-rag-server` instance
4. **Click the checkbox** next to your instance
5. **Click dropdown**: `Instance state` (top right, orange button)
6. **Click**: `Stop instance`
7. **Confirm**: Click `Stop` in the popup

**You should see**:
- Instance state changes from `Running` to `Stopping` to `Stopped`
- Status: Red square icon

✅ **Instance is stopped - you're no longer being charged for compute!**

💰 **You only pay for storage now (~$1.60/month for 20GB)**

### Start Instance (Resume Later)

**When you want to use Legalos again:**

**📍 In your web browser:**

1. **Go to**: https://console.aws.amazon.com/ec2/
2. **Click on**: `Instances` in left sidebar
3. **Find**: Your `legalos-rag-server` instance (shows "Stopped")
4. **Click the checkbox** next to your instance
5. **Click dropdown**: `Instance state` (top right)
6. **Click**: `Start instance`

**Wait 1-2 minutes**

**You should see**:
- Instance state changes to `Running` (green dot)
- Status check: `2/2 checks passed`

⚠️ **IMPORTANT: Check your NEW Public IP!**

1. **Click on your instance name**
2. **Look for**: "Public IPv4 address"
3. **Note the NEW IP** (it may have changed!)

**Use this new IP** to connect via SSH and access your API.

### Terminate Instance (⚠️ Permanent Deletion)

**ONLY do this when you're completely done with the project forever:**

**📍 In your web browser:**

1. **Go to**: https://console.aws.amazon.com/ec2/
2. **Click on**: `Instances` in left sidebar
3. **Find**: Your `legalos-rag-server` instance
4. **Click the checkbox** next to your instance
5. **Click dropdown**: `Instance state` (top right)
6. **Click**: `Terminate instance`
7. **Type**: `terminate` in the confirmation box
8. **Click**: `Terminate`

⚠️ **WARNING: This PERMANENTLY deletes:**
- The instance
- All code and data on it
- Your vectorDB
- Everything!

**This action CANNOT be undone!**

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

✅ **Your Legalos RAG system is now running on AWS EC2!**

### 📍 Quick Access Guide

**Your API URL:**
```
http://<YOUR_EC2_PUBLIC_IP>:5000/chat
```

**Replace** `<YOUR_EC2_PUBLIC_IP>` with your actual IP from EC2 console

**Example:**
```
http://3.6.205.42:5000/chat
```

### 🧪 Test Your API

**From terminal:**
```bash
curl -X POST http://<YOUR_EC2_PUBLIC_IP>:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the objective of the Right to Information Act, 2005?"}'
```

**From browser:**
```
http://<YOUR_EC2_PUBLIC_IP>:5000/chat?message=hello
```

### 🔗 Important URLs You Need

| Purpose | URL |
|---------|-----|
| AWS Console | https://console.aws.amazon.com/ |
| EC2 Dashboard | https://console.aws.amazon.com/ec2/ |
| Your API (GET) | http://YOUR_IP:5000/chat?message=hello |
| Your API (POST) | http://YOUR_IP:5000/chat |

### 💡 Remember

- **After each use**: Stop your EC2 instance to save money
- **Before using again**: Start instance and note the NEW IP address
- **Your IP may change**: After stop/start, check the new Public IPv4 address

---

## 📝 Quick Reference Commands

```bash
# Connect to EC2
ssh -i ~/.ssh/legalos-key.pem ubuntu@<EC2_IP>

# Activate environment
cd ~/legalos && source venv/bin/activate

# Run Flask API
python deployRun.py

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
  chatbot/ config/ vectorDB/ deployRun.py aws_deployment/requirements.txt \
  ubuntu@<EC2_IP>:~/legalos/
```

---

## 📚 Additional Resources

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Ollama Documentation](https://ollama.com/docs)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [LangChain Documentation](https://python.langchain.com/)
