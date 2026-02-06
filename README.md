# Lab Dojo v10 — AI Research Agent

A local-first AI research agent for scientists. Connects to PubMed, UniProt, PDB, ChEMBL, STRING, KEGG, and ClinicalTrials.gov. Runs Ollama locally for free inference, with serverless/ChatGPT/Claude fallbacks.

## What's New in v10

**Citation Verification Engine** — Every PMID verified via NCBI API before display. DOI links, passage-level provenance tracking.

**Persistent Project Memory** — Create research projects that survive sessions. Domain context, decision logs, evolving hypotheses, literature matrices.

**Provenance Tracker** — Links each AI claim to the specific abstract passage that supports it.

**Pipeline Engine** — Multi-step workflows (find papers → extract → compare → synthesize) in one command.

**Export System** — BibTeX, RIS, Markdown, LaTeX-ready output. One-click citation export.

**Verbosity Control** — Concise / Detailed / Comprehensive toggle. Get the depth you need.

**Deterministic Mode** — Temperature=0 toggle for reproducible outputs.

**Agentic Monitor** — Background publication tracking, retraction alerts, contradiction detection.

**Literature Matrix** — Organize papers by methods, findings, limitations across your project.

**Decision Log** — Track reasoning behind research decisions with full context.

## Quick Start

### Windows
Double-click `LabDojo_Installer.bat`

### Mac
```bash
chmod +x LabDojo_Installer.command
./LabDojo_Installer.command
```

### Manual
```bash
pip install fastapi uvicorn aiohttp pydantic
python labdojo.py
```

Open http://localhost:8080

## Requirements

- Python 3.11+
- Ollama (recommended, free local inference)
- Optional: Vast.ai API key, OpenAI API key, Anthropic API key

## Architecture

```
labdojo.py (single file, ~3600 lines)
├── Config & Logging
├── KnowledgeBase (SQLite, 14 tables)
│   ├── Projects, Decisions, Conversations
│   ├── Verified Citations, Provenance
│   ├── Literature Matrix, Pipeline Runs
│   ├── Monitored Topics, Alerts
│   └── Learned Q&A, Papers, Hypotheses
├── ScienceAPIs (7 databases)
│   ├── PubMed (efetch XML parsing)
│   ├── UniProt, PDB, ChEMBL
│   ├── STRING, KEGG, ClinicalTrials.gov
│   └── Citation Verification
├── AI Clients (4 backends)
│   ├── Ollama (primary, free)
│   ├── Vast.ai Serverless (fallback)
│   ├── ChatGPT (fallback)
│   └── Claude (fallback)
├── Export System (BibTeX, RIS, Markdown)
├── Pipeline Engine (multi-step workflows)
├── FastAPI Application (30+ endpoints)
└── Dashboard UI (Apple dark theme)
```

## License

Private research tool.
