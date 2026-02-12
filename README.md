# Lab Dojo — AI Research Workstation for Pathology

An open-source research workstation that connects 20 free science APIs to local AI models. Built for principal investigators and senior researchers who need grounded, citation-verified answers without sending data to the cloud.

**Lab Dojo Pathology v0.1.2** — Released February 2026 by [JuiceVendor Labs Inc.](https://labdojo.org)

## Quick Start

### One-Line Install (Recommended)

**Windows:**
```cmd
pip install labdojo && labdojo
```

**macOS/Linux:**
```bash
pip3 install labdojo && labdojo
```

That's it! Lab Dojo will open at http://localhost:8080

**Why pip?** No virus warnings, no code signing needed, trusted by millions of developers.

### Alternative: Run from Source

```bash
git clone https://github.com/SooparAI/labdojo.git
cd labdojo
pip install -e .
labdojo
```

## Science APIs (20 databases, no keys required)

| Category | APIs |
|----------|------|
| **Literature** | PubMed, ArXiv, bioRxiv, Europe PMC, Semantic Scholar, OpenAlex, Crossref, ORCID |
| **Proteins** | UniProt, PDB, STRING |
| **Chemistry** | ChEMBL, PubChem |
| **Pathways** | KEGG, Reactome |
| **Genomics** | NCBI Gene, OMIM |
| **Clinical** | ClinicalTrials.gov, RxNorm, DrugBank |

All APIs are free, public, and require no authentication. Lab Dojo routes questions to the appropriate databases automatically based on context.

## AI Backends

| Backend | Cost | Setup |
|---------|------|-------|
| **Ollama** (recommended) | Free | `ollama pull llama3:8b` |
| **Vast.ai Serverless** | ~$0.001/query | Configure in Settings |
| **OpenAI** | Per-token | Add API key in Settings |
| **Anthropic** | Per-token | Add API key in Settings |

Backends are tried in order: Ollama → Serverless → OpenAI → Anthropic. If all fail, raw API data is returned directly.

## Core Systems

| System | Description |
|--------|-------------|
| **Citation Verification** | Every PMID verified against PubMed before display |
| **Project Memory** | Persistent context across sessions with decision logs |
| **Provenance Tracking** | Each claim linked to the specific abstract passage that supports it |
| **Pipeline Engine** | Multi-step workflows: literature review, protein analysis |
| **Export** | BibTeX, RIS, Markdown output |
| **Background Monitor** | Publication tracking and alerts for your topics |
| **Verbosity Control** | Concise / Detailed / Comprehensive toggle |
| **Deterministic Mode** | Temperature=0 for reproducible outputs |

## Architecture

Lab Dojo is a single Python file (~2,750 lines) with no external database dependencies:

```
labdojo.py
├── Config & Logging
├── KnowledgeBase (SQLite, 14 tables, WAL mode, thread-safe)
│   ├── Projects, Decisions, Conversations
│   ├── Verified Citations, Provenance
│   ├── Pipeline Runs, Decision Logs
│   └── Monitored Topics, Alerts
├── ScienceAPIs (20 databases, async parallel, cached)
├── Intent Classifier (casual vs research routing)
├── Conversation Memory (10-turn context window)
├── AI Clients (4 backends with automatic fallback)
├── Export System (BibTeX, RIS, Markdown)
├── Pipeline Engine (multi-step workflows)
├── FastAPI Application (30+ REST endpoints)
└── Dashboard UI (dark theme)
```

## What's New in v0.1.2

- All version strings unified across `labdojo.py`, installers, and Docker inference server
- Website changelog system now live at [labdojo.org](https://labdojo.org) with AI-generated developer summaries
- Architecture documentation and roadmap published for contributors

See the full release notes on [GitHub Releases](https://github.com/SooparAI/labdojo/releases).

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
| `/pipeline/run` | POST | Run a pipeline |
| `/export/bibtex` | GET | Export BibTeX |
| `/export/ris` | GET | Export RIS |
| `/export/markdown` | GET | Export Markdown |
| `/monitor/topics` | GET/POST | Monitored topics |
| `/monitor/alerts` | GET | Get alerts |
| `/settings` | GET/POST | Configuration |
| `/learning/stats` | GET | Learning statistics |

## Configuration

All configuration is done through the Settings page in the UI, or via environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export VASTAI_API_KEY=your_key
export VASTAI_ENDPOINT_ID=12345
```

See `.env.example` for all available options.

## License

MIT — see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

---

Built by [JuiceVendor Labs Inc.](https://labdojo.org) — Lab Dojo Pathology is the first in a series of discipline-specific editions.
