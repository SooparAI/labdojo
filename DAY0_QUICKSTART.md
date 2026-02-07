# Lab Dojo v10 - Quick Start Guide

## Step 1: Install & Run

### Windows
Double-click `LabDojo_Installer.bat`

### Mac
Double-click `LabDojo_Installer.command`

(If macOS blocks it: right-click → Open → click "Open")

### Manual
```bash
pip install aiohttp fastapi uvicorn pydantic
ollama pull llama3:8b
python3 labdojo.py
```

**Dashboard opens at:** http://localhost:8080

## Step 2: Configure AI Backend

Lab Dojo works with just Ollama (free, local). For more powerful models, configure additional backends in Settings:

| Backend | Setup |
|---------|-------|
| **Ollama** (default) | Install Ollama, pull a model. Auto-detected. |
| **Vast.ai Serverless** | Add API key + endpoint ID in Settings → Compute |
| **OpenAI** | Add API key in Settings → API Keys |
| **Anthropic** | Add API key in Settings → API Keys |

## Step 3: Start Researching

### Chat
Ask science questions. Lab Dojo automatically searches PubMed, UniProt, PDB, and other databases to ground its answers with real citations.

### Projects
Create research projects to maintain persistent context across sessions. Add decision logs and literature matrices.

### Papers
Search PubMed directly. Export citations as BibTeX or RIS.

### Pipelines
Run multi-step workflows: literature reviews, protein analysis.

### Monitor
Track topics for new publications and get alerts.

## How Routing Works

| Question Type | Backend Used | Cost |
|--------------|-------------|------|
| Any question with Ollama available | Ollama (local) | Free |
| Ollama unavailable, serverless configured | Vast.ai Serverless | ~$0.001/query |
| Neither available, OpenAI key set | ChatGPT | Per-token |
| No AI backend available | Raw API data returned | Free |

## Cost Control

| Setting | Default |
|---------|---------|
| Daily Budget | $5.00 |
| When Exceeded | Routes to local only |
| Reset | Midnight daily |

## Files Included

| File | Purpose |
|------|---------|
| `labdojo.py` | Main application (~3,800 lines) |
| `LabDojo_Installer.bat` | Windows one-click launcher |
| `LabDojo_Installer.command` | Mac one-click launcher |
| `.env.example` | Environment variable template |
| `docker/` | Dockerfile for serverless deployment |
| `scripts/` | Serverless setup scripts |
