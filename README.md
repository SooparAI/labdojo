# Lab Dojo v10 — AI Research Agent for Scientists

An open-source AI-powered research assistant built for principal investigators, postdocs, and graduate researchers. Lab Dojo runs locally on your machine, connects to 7 science databases, and uses your choice of LLM to provide grounded, citation-verified answers.

## Why Lab Dojo?

Lab Dojo was built to address the specific pain points scientists face with general-purpose AI tools:

- **No hallucinated citations** — Every PMID is verified against PubMed before display
- **Treats you as a peer** — No "consult a professional" or "form a hypothesis first"
- **Persistent project memory** — Context survives across sessions, not just one chat
- **Provenance tracking** — Links each claim to the specific abstract passage that supports it
- **Reproducible outputs** — Deterministic mode with temperature=0 for version-controlled results
- **Your data stays local** — Runs on localhost, SQLite database on your machine

## Quick Start

### Windows
Double-click `LabDojo_Installer.bat`

### Mac
Double-click `LabDojo_Installer.command`

(If macOS blocks it: right-click → Open → click "Open")

### Manual Setup

```bash
# Install Python 3.11+
pip install aiohttp fastapi uvicorn pydantic

# Install Ollama (https://ollama.ai)
ollama pull llama3:8b

# Run Lab Dojo
python3 labdojo.py
# Open http://localhost:8080
```

## Features

### Science Databases (7 APIs, no keys needed)

| Database | What it provides |
|----------|-----------------|
| **PubMed** | Papers, abstracts, citations (NCBI E-utilities) |
| **UniProt** | Protein sequences, functions, annotations |
| **PDB** | 3D protein structures |
| **ChEMBL** | Drug targets, bioactivity data |
| **STRING** | Protein-protein interactions |
| **KEGG** | Metabolic and signaling pathways |
| **ClinicalTrials.gov** | Active clinical trials |

### AI Backends (choose one or more)

| Backend | Cost | Setup |
|---------|------|-------|
| **Ollama** (recommended) | Free | `ollama pull llama3:8b` |
| **Vast.ai Serverless** | ~$0.001/query | Configure in Settings |
| **OpenAI** | Per-token | Add API key in Settings |
| **Anthropic** | Per-token | Add API key in Settings |

Lab Dojo tries backends in order: Ollama → Serverless → ChatGPT → Claude. If all fail, it returns raw API data directly.

### Core Systems

- **Citation Verification Engine** — PMIDs verified via PubMed API before display
- **Project Memory** — Persistent context across sessions with decision logs
- **Literature Matrix** — Compare findings across papers
- **Pipeline Engine** — Multi-step workflows (literature review, protein analysis)
- **Export System** — BibTeX, RIS, Markdown output
- **Agentic Monitor** — Background publication tracking and alerts
- **Verbosity Control** — Concise / Detailed / Comprehensive toggle
- **Deterministic Mode** — Temperature=0 for reproducible outputs

## Configuration

All configuration is done through the Settings page in the UI, or via environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export VASTAI_API_KEY=your_key
export VASTAI_ENDPOINT_ID=12345
```

See `.env.example` for all available options.

## Architecture

Lab Dojo is a single Python file (~3,800 lines) with no external database dependencies:

```
labdojo.py (single file)
├── Config & Logging
├── KnowledgeBase (SQLite, 14 tables, WAL mode, thread-safe)
│   ├── Projects, Decisions, Conversations
│   ├── Verified Citations, Provenance
│   ├── Literature Matrix, Pipeline Runs
│   ├── Monitored Topics, Alerts
│   └── Learned Q&A, Papers, Hypotheses
├── ScienceAPIs (7 databases, async, cached)
├── AI Clients (4 backends with automatic fallback)
├── Export System (BibTeX, RIS, Markdown)
├── Pipeline Engine (multi-step workflows)
├── FastAPI Application (30+ REST endpoints)
└── Dashboard UI (Apple-style dark theme)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/status` | GET | System status |
| `/chat` | POST | Send a message |
| `/papers/search` | GET | Search PubMed |
| `/projects` | GET/POST | List/create projects |
| `/projects/{id}` | GET/DELETE | Get/delete project |
| `/projects/{id}/decisions` | GET/POST | Decision log |
| `/hypothesis` | POST | Generate hypothesis |
| `/pipeline/run` | POST | Run a pipeline |
| `/export/bibtex` | GET | Export BibTeX |
| `/export/ris` | GET | Export RIS |
| `/export/markdown` | GET | Export Markdown |
| `/monitor/topics` | GET/POST | Monitored topics |
| `/settings` | GET/POST | Configuration |
| `/learning/stats` | GET | Learning statistics |

## License

MIT — see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
