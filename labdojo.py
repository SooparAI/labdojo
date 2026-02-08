"""
Lab Dojo Pathology v0.1.1
AI-powered research workstation for pathology laboratories.

Copyright (c) 2025-2026 JuiceVendor Labs Inc.
Released under the MIT License. See LICENSE for details.

Requires: Python 3.10+, aiohttp, fastapi, uvicorn, pydantic
Optional: Ollama (local LLM), Vast.ai serverless, OpenAI/Anthropic keys
"""

__version__ = "0.1.1"
__author__ = "JuiceVendor Labs Inc."
__license__ = "MIT"

import asyncio
import hashlib
import json
import logging
import os
import platform
import re
import sqlite3
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_DIR = Path.home() / ".labdojo"
_DB_FILENAME = "labdojo.db"
_CONFIG_FILENAME = "config.json"
_SECRETS_FILENAME = "secrets.json"

_SENSITIVE_KEYS = frozenset({
    "vastai_api_key", "openai_api_key", "anthropic_api_key", "ncbi_api_key",
})


@dataclass
class Config:
    config_dir: str = str(_DEFAULT_CONFIG_DIR)
    host: str = "0.0.0.0"
    port: int = 8080
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = ""
    vastai_api_key: str = ""
    serverless_endpoint_id: int = 0
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-haiku-20240307"
    ncbi_api_key: str = ""
    daily_budget: float = 5.0
    verbosity: str = "detailed"
    deterministic: bool = False

    def __post_init__(self):
        for env_key, attr in (
            ("VASTAI_API_KEY", "vastai_api_key"),
            ("OPENAI_API_KEY", "openai_api_key"),
            ("ANTHROPIC_API_KEY", "anthropic_api_key"),
            ("NCBI_API_KEY", "ncbi_api_key"),
        ):
            env_val = os.environ.get(env_key, "")
            if env_val and not getattr(self, attr):
                setattr(self, attr, env_val)

        self._load_secrets()

    def _load_secrets(self):
        path = Path(self.config_dir) / _SECRETS_FILENAME
        if path.exists():
            try:
                with open(path) as fh:
                    secrets = json.load(fh)
                for key in _SENSITIVE_KEYS:
                    if key in secrets and secrets[key] and not getattr(self, key):
                        setattr(self, key, secrets[key])
            except Exception:
                pass

    def _save_secrets(self):
        Path(self.config_dir).mkdir(parents=True, exist_ok=True)
        secrets = {k: getattr(self, k) for k in _SENSITIVE_KEYS if getattr(self, k, "")}
        path = Path(self.config_dir) / _SECRETS_FILENAME
        with open(path, "w") as fh:
            json.dump(secrets, fh, indent=2)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    @classmethod
    def load(cls) -> "Config":
        path = _DEFAULT_CONFIG_DIR / _CONFIG_FILENAME
        if path.exists():
            try:
                with open(path) as fh:
                    data = json.load(fh)
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()

    def save(self):
        Path(self.config_dir).mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in asdict(self).items() if k not in _SENSITIVE_KEYS}
        with open(Path(self.config_dir) / _CONFIG_FILENAME, "w") as fh:
            json.dump(data, fh, indent=2)
        self._save_secrets()


def setup_logging(config: Config) -> logging.Logger:
    Path(config.config_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("labdojo")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(Path(config.config_dir) / "labdojo.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# KnowledgeBase  (SQLite, WAL mode, thread-safe)
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """Persistent storage for learned Q&A, projects, citations, and usage."""

    def __init__(self, config: Config):
        db_dir = Path(config.config_dir) / "knowledge"
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_dir / _DB_FILENAME)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        c = conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS learned_qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT DEFAULT '',
                api_used TEXT DEFAULT '',
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                quality_score REAL DEFAULT 0.5
            );
            CREATE TABLE IF NOT EXISTS usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                local_calls INTEGER DEFAULT 0,
                serverless_calls INTEGER DEFAULT 0,
                api_calls INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT DEFAULT '',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                key_terms TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                reasoning TEXT DEFAULT '',
                alternatives TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS literature_matrix (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                pmid TEXT DEFAULT '',
                title TEXT DEFAULT '',
                methods TEXT DEFAULT '',
                sample_size TEXT DEFAULT '',
                key_findings TEXT DEFAULT '',
                limitations TEXT DEFAULT '',
                relevance_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS verified_citations (
                pmid TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                authors TEXT DEFAULT '',
                journal TEXT DEFAULT '',
                year TEXT DEFAULT '',
                doi TEXT DEFAULT '',
                abstract TEXT DEFAULT '',
                verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                verification_source TEXT DEFAULT 'pubmed'
            );
            CREATE TABLE IF NOT EXISTS provenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim TEXT NOT NULL,
                pmid TEXT DEFAULT '',
                passage TEXT DEFAULT '',
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id TEXT PRIMARY KEY,
                pipeline_type TEXT NOT NULL,
                params TEXT DEFAULT '{}',
                status TEXT DEFAULT 'running',
                results TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS monitored_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_qa_question ON learned_qa(question);
            CREATE INDEX IF NOT EXISTS idx_qa_quality ON learned_qa(quality_score);
            CREATE INDEX IF NOT EXISTS idx_conv_project ON conversations(project_id);
            CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project_id);
            CREATE INDEX IF NOT EXISTS idx_matrix_project ON literature_matrix(project_id);
            CREATE INDEX IF NOT EXISTS idx_provenance_claim ON provenance(claim);
            CREATE INDEX IF NOT EXISTS idx_pipeline_status ON pipeline_runs(status);
            CREATE INDEX IF NOT EXISTS idx_alerts_topic ON alerts(topic);
        """)
        conn.commit()

    # -- Q&A Memory --

    def learn_qa(self, question: str, answer: str, source: str = "",
                 api_used: str = "", confidence: float = 0.5):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO learned_qa (question, answer, source, api_used, confidence, quality_score) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (question[:500], answer[:5000], source, api_used, confidence, confidence),
            )
            conn.commit()

    def recall_similar(self, question: str, threshold: float = 0.85) -> Optional[dict]:
        """Find a previously answered question that closely matches.

        Returns a dict with keys: question, answer, source, api_used, confidence.
        Returns None if no match exceeds the threshold.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT question, answer, source, api_used, confidence FROM learned_qa "
            "WHERE quality_score > 0.3 ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        q_lower = question.lower().strip()
        q_words = set(q_lower.split())
        best, best_score = None, 0.0
        for row in rows:
            stored = row["question"].lower().strip()
            if q_lower == stored:
                return dict(row)
            s_words = set(stored.split())
            if not q_words or not s_words:
                continue
            overlap = len(q_words & s_words) / max(len(q_words | s_words), 1)
            if overlap > best_score:
                best_score = overlap
                best = dict(row)
        return best if best_score >= threshold else None

    # -- Usage Tracking --

    def record_usage(self, local: int = 0, serverless: int = 0,
                     api: int = 0, cost: float = 0.0):
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT id FROM usage_stats WHERE date = ?", (today,)
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE usage_stats SET local_calls = local_calls + ?, "
                    "serverless_calls = serverless_calls + ?, "
                    "api_calls = api_calls + ?, total_cost = total_cost + ? "
                    "WHERE date = ?",
                    (local, serverless, api, cost, today),
                )
            else:
                conn.execute(
                    "INSERT INTO usage_stats (date, local_calls, serverless_calls, api_calls, total_cost) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (today, local, serverless, api, cost),
                )
            conn.commit()

    def get_usage(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM usage_stats WHERE date = ?", (today,)
        ).fetchone()
        if row:
            return dict(row)
        return {"date": today, "local_calls": 0, "serverless_calls": 0,
                "api_calls": 0, "total_cost": 0.0}

    # -- Conversations --

    def add_conversation(self, project_id: str, role: str, content: str):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO conversations (project_id, role, content) VALUES (?, ?, ?)",
                (project_id, role, content[:5000]),
            )
            conn.commit()

    def get_conversations(self, project_id: str, limit: int = 50) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT role, content, created_at FROM conversations "
            "WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def clear_conversations(self):
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM conversations")
            conn.commit()

    # -- Projects --

    def create_project(self, name: str, description: str = "",
                       key_terms: str = "") -> str:
        pid = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO projects (id, name, description, key_terms) VALUES (?, ?, ?, ?)",
                (pid, name, description, key_terms),
            )
            conn.commit()
        return pid

    def get_projects(self) -> list:
        conn = self._get_conn()
        return [dict(r) for r in conn.execute(
            "SELECT * FROM projects ORDER BY updated_at DESC"
        ).fetchall()]

    def get_project(self, pid: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
        return dict(row) if row else None

    def delete_project(self, pid: str):
        with self._lock:
            conn = self._get_conn()
            for tbl in ("projects", "conversations", "decisions", "literature_matrix"):
                col = "id" if tbl == "projects" else "project_id"
                conn.execute(f"DELETE FROM {tbl} WHERE {col} = ?", (pid,))
            conn.commit()

    # -- Decisions --

    def add_decision(self, project_id: str, decision: str,
                     reasoning: str = "", alternatives: str = ""):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO decisions (project_id, decision, reasoning, alternatives) "
                "VALUES (?, ?, ?, ?)",
                (project_id, decision, reasoning, alternatives),
            )
            conn.commit()

    def get_decisions(self, project_id: str, limit: int = 50) -> list:
        conn = self._get_conn()
        return [dict(r) for r in conn.execute(
            "SELECT * FROM decisions WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()]

    # -- Literature Matrix --

    def add_to_matrix(self, project_id: str, pmid: str, title: str,
                      methods: str = "", sample_size: str = "",
                      key_findings: str = "", limitations: str = "",
                      relevance_score: float = 0.0):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO literature_matrix "
                "(project_id, pmid, title, methods, sample_size, key_findings, limitations, relevance_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (project_id, pmid, title, methods, sample_size, key_findings, limitations, relevance_score),
            )
            conn.commit()

    def get_literature_matrix(self, project_id: str) -> list:
        conn = self._get_conn()
        return [dict(r) for r in conn.execute(
            "SELECT * FROM literature_matrix WHERE project_id = ? ORDER BY relevance_score DESC",
            (project_id,),
        ).fetchall()]

    # -- Citations --

    def verify_citation(self, pmid: str, title: str = "", authors: str = "",
                        journal: str = "", year: str = "", doi: str = "",
                        abstract: str = "", source: str = "pubmed"):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO verified_citations "
                "(pmid, title, authors, journal, year, doi, abstract, verification_source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (pmid, title, authors, journal, year, doi, abstract, source),
            )
            conn.commit()

    def get_verified_citation(self, pmid: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM verified_citations WHERE pmid = ?", (pmid,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_verified_citations(self) -> list:
        conn = self._get_conn()
        return [dict(r) for r in conn.execute(
            "SELECT * FROM verified_citations ORDER BY verified_at DESC"
        ).fetchall()]

    # -- Provenance --

    def add_provenance(self, claim: str, pmid: str = "",
                       passage: str = "", confidence: float = 0.0):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO provenance (claim, pmid, passage, confidence) VALUES (?, ?, ?, ?)",
                (claim, pmid, passage, confidence),
            )
            conn.commit()

    # -- Pipeline Runs --

    def create_pipeline_run(self, pipeline_type: str, params: dict) -> str:
        rid = hashlib.md5(f"{pipeline_type}{time.time()}".encode()).hexdigest()[:12]
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO pipeline_runs (id, pipeline_type, params) VALUES (?, ?, ?)",
                (rid, pipeline_type, json.dumps(params)),
            )
            conn.commit()
        return rid

    def update_pipeline_run(self, run_id: str, status: str, results: dict):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE pipeline_runs SET status = ?, results = ?, completed_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (status, json.dumps(results), run_id),
            )
            conn.commit()

    def get_pipeline_runs(self) -> list:
        conn = self._get_conn()
        return [dict(r) for r in conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT 20"
        ).fetchall()]

    # -- Monitoring --

    def add_monitored_topic(self, topic: str):
        with self._lock:
            conn = self._get_conn()
            conn.execute("INSERT INTO monitored_topics (topic) VALUES (?)", (topic,))
            conn.commit()

    def get_monitored_topics(self) -> list:
        conn = self._get_conn()
        return [dict(r) for r in conn.execute(
            "SELECT * FROM monitored_topics ORDER BY created_at DESC"
        ).fetchall()]

    def remove_monitored_topic(self, topic_id: int):
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM monitored_topics WHERE id = ?", (topic_id,))
            conn.commit()

    def add_alert(self, topic: str, message: str, data: str = ""):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO alerts (topic, message, data) VALUES (?, ?, ?)",
                (topic, message, data),
            )
            conn.commit()

    def get_alerts(self) -> list:
        conn = self._get_conn()
        return [dict(r) for r in conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT 50"
        ).fetchall()]

    # -- Admin --

    def clear_bad_data(self):
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM learned_qa WHERE quality_score < 0.3")
            conn.execute("DELETE FROM learned_qa WHERE length(answer) < 20")
            conn.commit()

    def reset_learning(self):
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM learned_qa")
            conn.execute("DELETE FROM usage_stats")
            conn.commit()


# ---------------------------------------------------------------------------
# Intent Classification (casual vs research)
# ---------------------------------------------------------------------------

_CASUAL_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings|yo|sup|hola|good\s*(morning|afternoon|evening|day|night))(\s+\w+)*[\s!.,?]*$",
    r"^(thanks|thank\s*you(\s+\w+)*|thx|ty|cheers|appreciated|great|awesome|cool|nice|ok|okay|got\s*it|understood)[\s!.,?]*$",
    r"^(bye|goodbye|see\s*you(\s+\w+)*|later|cya|take\s*care|peace)[\s!.,?]*$",
    r"^(what\s*(can|do)\s*you\s*do|help|how\s*do\s*(i|you)\s*use|what\s*is\s*this|who\s*are\s*you|what\s*are\s*you)[\s?!.,]*$",
    r"^(test|testing|ping|are\s*you\s*(there|working|alive|online))[\s?!.,]*$",
    r"^(how\s*are\s*you|how('s|\s*is)\s*it\s*going|what('s|\s*is)\s*up)[\s?!.,]*$",
]

_CASUAL_RESPONSES = {
    "greeting": "Welcome to Lab Dojo. I am your research assistant with access to 20 biomedical databases including PubMed, UniProt, PDB, ChEMBL, and more. Ask me any research question and I will ground my response in real data with PMID citations. For example, try asking about a specific protein, pathway, drug target, or gene.",
    "thanks": "You are welcome. Let me know if you have another question.",
    "farewell": "Take care. Your conversation history is saved locally.",
    "help": "Lab Dojo connects a local AI model to 20 free biomedical APIs. You can:\n\n- **Ask research questions** in the Chat tab (grounded in PubMed, UniProt, PDB, etc.)\n- **Search papers** in the Papers tab with BibTeX/RIS export\n- **Run pipelines** for literature review, protein analysis, drug/target profiling, pathway analysis, or cancer genomics\n- **Create projects** to organize research context and decisions\n- **Monitor topics** for new publications\n\nAll data stays on your machine. No cloud dependency.",
    "test": "Lab Dojo is running. All systems operational.",
    "meta": "I am Lab Dojo, a research workstation built for principal investigators and research labs. I run locally on your machine with access to 20 biomedical databases. Ask me a research question to get started.",
}


def classify_intent(message: str) -> tuple[str, str]:
    """Classify whether a message is casual or a research query.

    Returns (intent_type, response) where intent_type is 'casual' or 'research'.
    For casual messages, response contains the appropriate reply.
    For research messages, response is empty.
    """
    msg = message.strip()
    if len(msg) < 3:
        return "casual", _CASUAL_RESPONSES["greeting"]

    msg_lower = msg.lower().strip()

    for pattern in _CASUAL_PATTERNS:
        if re.match(pattern, msg_lower, re.IGNORECASE):
            if re.match(r"^(hi|hello|hey|howdy|greetings|yo|sup|hola|good\s*(morning|afternoon|evening|day|night))", msg_lower):
                return "casual", _CASUAL_RESPONSES["greeting"]
            elif re.match(r"^(thanks|thank\s*you|thx|ty|cheers|appreciated|great|awesome|cool|nice|ok|okay|got\s*it|understood)", msg_lower):
                return "casual", _CASUAL_RESPONSES["thanks"]
            elif re.match(r"^(bye|goodbye|see\s*you|later|cya|take\s*care|peace)", msg_lower):
                return "casual", _CASUAL_RESPONSES["farewell"]
            elif re.match(r"^(what\s*(can|do)\s*you\s*do|help|how\s*do)", msg_lower):
                return "casual", _CASUAL_RESPONSES["help"]
            elif re.match(r"^(who\s*are\s*you|what\s*are\s*you|what\s*is\s*this)", msg_lower):
                return "casual", _CASUAL_RESPONSES["meta"]
            elif re.match(r"^(test|testing|ping|are\s*you)", msg_lower):
                return "casual", _CASUAL_RESPONSES["test"]
            elif re.match(r"^(how\s*are\s*you|how('s|\s*is)\s*it)", msg_lower):
                return "casual", _CASUAL_RESPONSES["meta"]
            return "casual", _CASUAL_RESPONSES["greeting"]

    return "research", ""


# ---------------------------------------------------------------------------
# Science API Gateway  (20 databases, zero keys required)
# ---------------------------------------------------------------------------

_API_CATALOG = {
    "pubmed": {"name": "PubMed", "desc": "Biomedical literature", "base": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/", "free": True},
    "europe_pmc": {"name": "Europe PMC", "desc": "Biomedical literature + grants", "base": "https://www.ebi.ac.uk/europepmc/webservices/rest/", "free": True},
    "openalex": {"name": "OpenAlex", "desc": "Scholarly works and citations", "base": "https://api.openalex.org/", "free": True},
    "crossref": {"name": "Crossref", "desc": "DOI metadata for journals", "base": "https://api.crossref.org/", "free": True},
    "uniprot": {"name": "UniProt", "desc": "Protein sequences and function", "base": "https://rest.uniprot.org/", "free": True},
    "pdb": {"name": "RCSB PDB", "desc": "3D macromolecular structures", "base": "https://data.rcsb.org/rest/v1/", "free": True},
    "alphafold": {"name": "AlphaFold DB", "desc": "Predicted protein structures", "base": "https://alphafold.ebi.ac.uk/api/", "free": True},
    "ensembl": {"name": "Ensembl", "desc": "Genes, transcripts, variants", "base": "https://rest.ensembl.org/", "free": True},
    "chembl": {"name": "ChEMBL", "desc": "Bioactive molecules and assays", "base": "https://www.ebi.ac.uk/chembl/api/data/", "free": True},
    "pubchem": {"name": "PubChem", "desc": "Compounds and bioassays", "base": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/", "free": True},
    "gene_ontology": {"name": "Gene Ontology", "desc": "GO terms and annotations", "base": "https://api.geneontology.org/api/", "free": True},
    "reactome": {"name": "Reactome", "desc": "Curated biological pathways", "base": "https://reactome.org/ContentService/", "free": True},
    "string": {"name": "STRING", "desc": "Protein interaction networks", "base": "https://string-db.org/api/", "free": True},
    "kegg": {"name": "KEGG", "desc": "Pathway and genome databases", "base": "https://rest.kegg.jp/", "free": True},
    "clinicaltrials": {"name": "ClinicalTrials.gov", "desc": "Clinical trial metadata", "base": "https://clinicaltrials.gov/api/v2/", "free": True},
    "openfda": {"name": "openFDA", "desc": "FDA drugs, devices, adverse events", "base": "https://api.fda.gov/", "free": True},
    "rxnorm": {"name": "RxNorm", "desc": "Normalized drug names", "base": "https://rxnav.nlm.nih.gov/REST/", "free": True},
    "gdc": {"name": "GDC", "desc": "Cancer genomics (TCGA)", "base": "https://api.gdc.cancer.gov/", "free": True},
    "cbioportal": {"name": "cBioPortal", "desc": "Cancer mutations and CNVs", "base": "https://www.cbioportal.org/api/", "free": True},
}

_ROUTE_MAP = {
    "literature": ["pubmed", "europe_pmc", "openalex", "crossref"],
    "protein": ["uniprot", "pdb", "alphafold", "string"],
    "gene": ["ensembl", "gene_ontology", "kegg"],
    "drug": ["chembl", "pubchem", "openfda", "rxnorm"],
    "pathway": ["reactome", "kegg", "gene_ontology"],
    "cancer": ["gdc", "cbioportal", "pubmed"],
    "clinical": ["clinicaltrials", "openfda", "rxnorm"],
}

_TOPIC_KEYWORDS = {
    "literature": {"paper", "papers", "study", "studies", "review", "publication", "cite", "citation", "literature", "journal", "article"},
    "protein": {"protein", "kinase", "phosphorylation", "domain", "binding", "receptor", "enzyme", "substrate", "structure", "fold", "uniprot", "pdb"},
    "gene": {"gene", "transcript", "expression", "variant", "mutation", "snp", "allele", "chromosome", "genome", "ensembl", "exon", "intron"},
    "drug": {"drug", "compound", "inhibitor", "agonist", "antagonist", "pharmacology", "therapeutic", "dose", "toxicity", "fda", "clinical trial"},
    "pathway": {"pathway", "signaling", "cascade", "metabolic", "reactome", "kegg", "mapk", "wnt", "notch", "nfkb", "nf-kb", "jak", "stat"},
    "cancer": {"cancer", "tumor", "tumour", "oncogene", "carcinoma", "melanoma", "lymphoma", "leukemia", "metastasis", "tcga", "mutation"},
    "clinical": {"clinical", "trial", "patient", "treatment", "therapy", "adverse", "efficacy", "dosage", "prescription", "diagnosis"},
}


class ScienceAPIs:
    """Unified gateway to 20 biomedical databases with caching and smart routing."""

    def __init__(self, config: Config, kb: KnowledgeBase):
        self.config = config
        self.kb = kb
        self.logger = logging.getLogger("labdojo.api")
        self._cache: dict = {}
        self._cache_ttl = 300
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _cache_key(self, method: str, query: str) -> str:
        return hashlib.md5(f"{method}:{query}".encode()).hexdigest()

    def _cache_get(self, key: str):
        entry = self._cache.get(key)
        if entry and time.time() - entry["ts"] < self._cache_ttl:
            return entry["data"]
        return None

    def _cache_set(self, key: str, data):
        self._cache[key] = {"data": data, "ts": time.time()}
        if len(self._cache) > 500:
            oldest = sorted(self._cache, key=lambda k: self._cache[k]["ts"])[:100]
            for k in oldest:
                del self._cache[k]

    def classify_question(self, question: str) -> list[str]:
        """Determine which API categories are relevant to a question."""
        q = question.lower()
        matched = []
        for topic, keywords in _TOPIC_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                matched.append(topic)
        if not matched:
            matched = ["literature"]
        return matched

    def get_apis_for_question(self, question: str) -> list[str]:
        """Return ordered list of API IDs to query for a given question."""
        topics = self.classify_question(question)
        apis = []
        for topic in topics:
            for api_id in _ROUTE_MAP.get(topic, []):
                if api_id not in apis:
                    apis.append(api_id)
        if "pubmed" not in apis:
            apis.append("pubmed")
        return apis[:8]

    def get_api_status(self) -> dict:
        return {
            k: {"name": v["name"], "description": v["desc"],
                "rate_limit": "Free, no key" if v["free"] else "Key required",
                "has_key": True, "requires_key": not v["free"]}
            for k, v in _API_CATALOG.items()
        }

    async def fetch_grounding_data(self, question: str) -> tuple[str, list[str]]:
        """Fetch data from relevant APIs in parallel and return (context_string, sources_list)."""
        api_ids = self.get_apis_for_question(question)
        self.logger.info(f"Routing to APIs: {api_ids}")
        search_terms = self._extract_search_terms(question)

        async def _query_one(api_id: str) -> tuple[str, str]:
            try:
                result = await self._query_api(api_id, search_terms, question)
                if result:
                    self.kb.record_usage(api=1)
                    return api_id, result
            except Exception as exc:
                self.logger.debug(f"{api_id} query failed: {exc}")
            return api_id, ""

        results = await asyncio.gather(*[_query_one(aid) for aid in api_ids])

        context_parts = []
        sources = []
        for api_id, result in results:
            if result:
                context_parts.append(result)
                sources.append(_API_CATALOG[api_id]["name"])

        return "\n\n".join(context_parts), sources

    def _extract_search_terms(self, question: str) -> str:
        stop = {"what", "is", "the", "role", "of", "in", "how", "does", "do",
                "can", "why", "are", "was", "were", "been", "being", "have",
                "has", "had", "will", "would", "could", "should", "may",
                "might", "shall", "a", "an", "and", "or", "but", "for",
                "with", "about", "between", "through", "during", "before",
                "after", "above", "below", "to", "from", "up", "down", "on",
                "off", "over", "under", "again", "further", "then", "once",
                "this", "that", "these", "those", "it", "its", "they",
                "them", "their", "which", "who", "whom", "where", "when",
                "there", "here", "all", "each", "every", "both", "few",
                "more", "most", "other", "some", "such", "no", "not",
                "only", "own", "same", "so", "than", "too", "very",
                "tell", "me", "explain", "describe", "discuss"}
        words = re.sub(r"[^\w\s-]", "", question).split()
        terms = [w for w in words if w.lower() not in stop and len(w) > 2]
        return " ".join(terms[:6]) if terms else question[:60]

    async def _query_api(self, api_id: str, terms: str, question: str) -> str:
        ck = self._cache_key(api_id, terms)
        cached = self._cache_get(ck)
        if cached:
            return cached

        dispatch = {
            "pubmed": self._search_pubmed,
            "europe_pmc": self._search_europe_pmc,
            "openalex": self._search_openalex,
            "crossref": self._search_crossref,
            "uniprot": self._search_uniprot,
            "pdb": self._search_pdb,
            "alphafold": self._search_alphafold,
            "ensembl": self._search_ensembl,
            "chembl": self._search_chembl,
            "pubchem": self._search_pubchem,
            "gene_ontology": self._search_gene_ontology,
            "reactome": self._search_reactome,
            "string": self._search_string,
            "kegg": self._search_kegg,
            "clinicaltrials": self._search_clinicaltrials,
            "openfda": self._search_openfda,
            "rxnorm": self._search_rxnorm,
            "gdc": self._search_gdc,
            "cbioportal": self._search_cbioportal,
        }
        fn = dispatch.get(api_id)
        if not fn:
            return ""
        result = await fn(terms)
        if result:
            self._cache_set(ck, result)
        return result

    async def _http_get(self, url: str, params: dict = None,
                        headers: dict = None, timeout: int = 15) -> dict | str | None:
        try:
            session = await self._get_session()
            async with session.get(url, params=params, headers=headers,
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    return None
                ct = resp.content_type or ""
                if "json" in ct:
                    return await resp.json()
                return await resp.text()
        except Exception:
            return None

    async def _http_post(self, url: str, json_data: dict = None,
                         headers: dict = None, timeout: int = 15) -> dict | None:
        try:
            session = await self._get_session()
            async with session.post(url, json=json_data, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception:
            return None

    # -- PubMed (efetch XML) --

    async def _search_pubmed(self, terms: str) -> str:
        params = {"db": "pubmed", "term": terms, "retmax": "8", "retmode": "json", "sort": "relevance"}
        if self.config.ncbi_api_key:
            params["api_key"] = self.config.ncbi_api_key
        data = await self._http_get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params)
        if not data or not isinstance(data, dict):
            return ""
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return ""

        fetch_params = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"}
        if self.config.ncbi_api_key:
            fetch_params["api_key"] = self.config.ncbi_api_key
        xml = await self._http_get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=fetch_params)
        if not xml or not isinstance(xml, str):
            return ""

        return self._parse_pubmed_xml(xml)

    def _parse_pubmed_xml(self, xml: str) -> str:
        papers = []
        for article in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL):
            pmid = _xml_tag(article, "PMID")
            title = _xml_tag(article, "ArticleTitle")
            journal = _xml_tag(article, "Title")
            year = _xml_tag(article, "Year")
            abstract = _xml_tag(article, "AbstractText")
            authors = re.findall(r"<LastName>(.*?)</LastName>", article)
            doi_match = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', article)
            doi = doi_match.group(1) if doi_match else ""

            if pmid and title:
                auth_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    auth_str += " et al."
                entry = f"[PMID:{pmid}] {title}\n  Authors: {auth_str}\n  Journal: {journal} ({year})"
                if doi:
                    entry += f"\n  DOI: https://doi.org/{doi}"
                if abstract:
                    entry += f"\n  Abstract: {abstract[:500]}"
                papers.append(entry)

                self.kb.verify_citation(pmid, title, ", ".join(authors), journal, year, doi, abstract or "")

        if not papers:
            return ""
        self.logger.info(f"PubMed: {len(papers)} papers")
        return "PUBMED RESULTS:\n" + "\n\n".join(papers)

    async def search_pubmed(self, query: str, max_results: int = 10) -> list[dict]:
        """Public method for the papers search endpoint."""
        params = {"db": "pubmed", "term": query, "retmax": str(max_results),
                  "retmode": "json", "sort": "relevance"}
        if self.config.ncbi_api_key:
            params["api_key"] = self.config.ncbi_api_key
        data = await self._http_get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params)
        if not data or not isinstance(data, dict):
            return []
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        fetch_params = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"}
        if self.config.ncbi_api_key:
            fetch_params["api_key"] = self.config.ncbi_api_key
        xml = await self._http_get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=fetch_params)
        if not xml or not isinstance(xml, str):
            return []

        results = []
        for article in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL):
            pmid = _xml_tag(article, "PMID")
            title = _xml_tag(article, "ArticleTitle")
            journal = _xml_tag(article, "Title")
            year = _xml_tag(article, "Year")
            abstract = _xml_tag(article, "AbstractText")
            authors = re.findall(r"<LastName>(.*?)</LastName>", article)
            doi_match = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', article)
            doi = doi_match.group(1) if doi_match else ""
            if pmid:
                results.append({
                    "pmid": pmid, "title": title or "", "authors": authors,
                    "journal": journal or "", "pub_date": year or "",
                    "abstract": abstract or "",
                    "doi_url": f"https://doi.org/{doi}" if doi else "",
                })
        return results

    # -- Europe PMC --

    async def _search_europe_pmc(self, terms: str) -> str:
        data = await self._http_get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": terms, "format": "json", "pageSize": "5"})
        if not data or not isinstance(data, dict):
            return ""
        results = data.get("resultList", {}).get("result", [])
        if not results:
            return ""
        lines = []
        for r in results[:5]:
            pmid = r.get("pmid", "")
            title = r.get("title", "")
            src = r.get("source", "")
            year = r.get("pubYear", "")
            cited = r.get("citedByCount", 0)
            lines.append(f"[{src}:{pmid}] {title} ({year}, cited {cited}x)")
        return "EUROPE PMC:\n" + "\n".join(lines)

    # -- OpenAlex --

    async def _search_openalex(self, terms: str) -> str:
        data = await self._http_get(
            "https://api.openalex.org/works",
            params={"search": terms, "per_page": "5", "sort": "relevance_score:desc"},
            headers={"User-Agent": "LabDojo/0.1.1 (mailto:dev@juicevendorlabs.com)"})
        if not data or not isinstance(data, dict):
            return ""
        results = data.get("results", [])
        if not results:
            return ""
        lines = []
        for w in results[:5]:
            title = w.get("display_name", "")
            year = w.get("publication_year", "")
            cited = w.get("cited_by_count", 0)
            doi = w.get("doi", "")
            lines.append(f"{title} ({year}, cited {cited}x) {doi}")
        return "OPENALEX:\n" + "\n".join(lines)

    # -- Crossref --

    async def _search_crossref(self, terms: str) -> str:
        data = await self._http_get(
            "https://api.crossref.org/works",
            params={"query": terms, "rows": "5", "sort": "relevance"})
        if not data or not isinstance(data, dict):
            return ""
        items = data.get("message", {}).get("items", [])
        if not items:
            return ""
        lines = []
        for item in items[:5]:
            title = item.get("title", [""])[0] if item.get("title") else ""
            doi = item.get("DOI", "")
            ref_count = item.get("is-referenced-by-count", 0)
            lines.append(f"{title} (DOI:{doi}, refs:{ref_count})")
        return "CROSSREF:\n" + "\n".join(lines)

    # -- UniProt --

    async def _search_uniprot(self, terms: str) -> str:
        data = await self._http_get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": terms, "format": "json", "size": "3"})
        if not data or not isinstance(data, dict):
            return ""
        results = data.get("results", [])
        if not results:
            return ""
        lines = []
        for entry in results[:3]:
            acc = entry.get("primaryAccession", "")
            name = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
            org = entry.get("organism", {}).get("scientificName", "")
            func_comments = [c for c in entry.get("comments", []) if c.get("commentType") == "FUNCTION"]
            func_text = func_comments[0].get("texts", [{}])[0].get("value", "") if func_comments else ""
            lines.append(f"[{acc}] {name} ({org})")
            if func_text:
                lines.append(f"  Function: {func_text[:300]}")
        return "UNIPROT:\n" + "\n".join(lines)

    # -- PDB --

    async def _search_pdb(self, terms: str) -> str:
        query = {"query": {"type": "terminal", "service": "full_text",
                           "parameters": {"value": terms}},
                 "return_type": "entry", "request_options": {"results_content_type": ["experimental"],
                                                              "paginate": {"start": 0, "rows": 3}}}
        data = await self._http_post("https://search.rcsb.org/rcsbsearch/v2/query", json_data=query)
        if not data or not isinstance(data, dict):
            return ""
        hits = data.get("result_set", [])
        if not hits:
            return ""
        lines = []
        for h in hits[:3]:
            pdb_id = h.get("identifier", "")
            lines.append(f"PDB: {pdb_id} (https://www.rcsb.org/structure/{pdb_id})")
        return "PDB STRUCTURES:\n" + "\n".join(lines)

    # -- AlphaFold --

    async def _search_alphafold(self, terms: str) -> str:
        first_term = terms.split()[0] if terms.split() else terms
        data = await self._http_get(
            f"https://alphafold.ebi.ac.uk/api/prediction/{first_term}",
            headers={"Accept": "application/json"})
        if not data:
            acc_data = await self._http_get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={"query": terms, "format": "json", "size": "1"})
            if acc_data and isinstance(acc_data, dict):
                results = acc_data.get("results", [])
                if results:
                    acc = results[0].get("primaryAccession", "")
                    if acc:
                        data = await self._http_get(
                            f"https://alphafold.ebi.ac.uk/api/prediction/{acc}")
        if not data:
            return ""
        if isinstance(data, list) and data:
            entry = data[0]
            return f"ALPHAFOLD: {entry.get('uniprotAccession', '')} pLDDT={entry.get('plddt', 'N/A')} URL={entry.get('cifUrl', '')}"
        return ""

    # -- Ensembl --

    async def _search_ensembl(self, terms: str) -> str:
        first_term = terms.split()[0] if terms.split() else terms
        data = await self._http_get(
            f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{first_term}",
            headers={"Content-Type": "application/json"})
        if not data or not isinstance(data, dict) or "error" in data:
            return ""
        gene_id = data.get("id", "")
        desc = data.get("description", "")
        biotype = data.get("biotype", "")
        loc = f"{data.get('seq_region_name', '')}:{data.get('start', '')}-{data.get('end', '')}"
        return f"ENSEMBL: {gene_id} {desc} [{biotype}] Location: {loc}"

    # -- ChEMBL --

    async def _search_chembl(self, terms: str) -> str:
        data = await self._http_get(
            "https://www.ebi.ac.uk/chembl/api/data/molecule/search",
            params={"q": terms, "format": "json", "limit": "3"})
        if not data or not isinstance(data, dict):
            return ""
        mols = data.get("molecules", [])
        if not mols:
            return ""
        lines = []
        for m in mols[:3]:
            cid = m.get("molecule_chembl_id", "")
            name = m.get("pref_name", "") or ""
            mtype = m.get("molecule_type", "")
            phase = m.get("max_phase", "")
            lines.append(f"[{cid}] {name} (type:{mtype}, phase:{phase})")
        return "CHEMBL:\n" + "\n".join(lines)

    # -- PubChem --

    async def _search_pubchem(self, terms: str) -> str:
        first_term = terms.split()[0] if terms.split() else terms
        data = await self._http_get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{first_term}/property/MolecularFormula,MolecularWeight,IUPACName/JSON")
        if not data or not isinstance(data, dict):
            return ""
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return ""
        p = props[0]
        cid = p.get("CID", "")
        formula = p.get("MolecularFormula", "")
        weight = p.get("MolecularWeight", "")
        iupac = p.get("IUPACName", "")
        return f"PUBCHEM: CID:{cid} {iupac} (Formula:{formula}, MW:{weight})"

    # -- Gene Ontology --

    async def _search_gene_ontology(self, terms: str) -> str:
        data = await self._http_get(
            "https://api.geneontology.org/api/search/entity/autocomplete/" + terms.split()[0],
            params={"rows": "5"})
        if not data or not isinstance(data, dict):
            return ""
        docs = data.get("docs", [])
        if not docs:
            return ""
        lines = []
        for d in docs[:5]:
            go_id = d.get("id", "")
            label = d.get("label", "")
            cat = d.get("category", "")
            lines.append(f"[{go_id}] {label} ({cat})")
        return "GENE ONTOLOGY:\n" + "\n".join(lines)

    # -- Reactome --

    async def _search_reactome(self, terms: str) -> str:
        data = await self._http_get(
            "https://reactome.org/ContentService/search/query",
            params={"query": terms, "species": "Homo sapiens", "types": "Pathway", "cluster": "true"})
        if not data or not isinstance(data, dict):
            return ""
        results = data.get("results", [])
        if not results:
            return ""
        lines = []
        for group in results[:2]:
            for entry in group.get("entries", [])[:3]:
                st_id = entry.get("stId", "")
                name = entry.get("name", "")
                lines.append(f"[{st_id}] {name}")
        return "REACTOME PATHWAYS:\n" + "\n".join(lines) if lines else ""

    # -- STRING --

    async def _search_string(self, terms: str) -> str:
        first_term = terms.split()[0] if terms.split() else terms
        data = await self._http_get(
            "https://string-db.org/api/json/network",
            params={"identifiers": first_term, "species": "9606", "limit": "5"})
        if not data or not isinstance(data, list):
            return ""
        if not data:
            return ""
        lines = []
        seen = set()
        for interaction in data[:10]:
            a = interaction.get("preferredName_A", "")
            b = interaction.get("preferredName_B", "")
            score = interaction.get("score", 0)
            pair = tuple(sorted([a, b]))
            if pair not in seen:
                seen.add(pair)
                lines.append(f"{a} <-> {b} (score: {score:.3f})")
        return "STRING INTERACTIONS:\n" + "\n".join(lines[:5]) if lines else ""

    # -- KEGG --

    async def _search_kegg(self, terms: str) -> str:
        first_term = terms.split()[0] if terms.split() else terms
        text = await self._http_get(f"https://rest.kegg.jp/find/pathway/{first_term}")
        if not text or not isinstance(text, str):
            return ""
        lines = []
        for line in text.strip().split("\n")[:5]:
            parts = line.split("\t")
            if len(parts) >= 2:
                lines.append(f"[{parts[0]}] {parts[1]}")
        return "KEGG PATHWAYS:\n" + "\n".join(lines) if lines else ""

    # -- ClinicalTrials.gov --

    async def _search_clinicaltrials(self, terms: str) -> str:
        data = await self._http_get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"query.term": terms, "pageSize": "5", "format": "json"})
        if not data or not isinstance(data, dict):
            return ""
        studies = data.get("studies", [])
        if not studies:
            return ""
        lines = []
        for s in studies[:5]:
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            nct = ident.get("nctId", "")
            title = ident.get("briefTitle", "")
            status = status_mod.get("overallStatus", "")
            lines.append(f"[{nct}] {title} (Status: {status})")
        return "CLINICAL TRIALS:\n" + "\n".join(lines)

    # -- openFDA --

    async def _search_openfda(self, terms: str) -> str:
        data = await self._http_get(
            "https://api.fda.gov/drug/label.json",
            params={"search": f'openfda.generic_name:"{terms.split()[0]}"', "limit": "3"})
        if not data or not isinstance(data, dict):
            return ""
        results = data.get("results", [])
        if not results:
            return ""
        lines = []
        for r in results[:3]:
            brand = r.get("openfda", {}).get("brand_name", [""])[0] if r.get("openfda", {}).get("brand_name") else ""
            generic = r.get("openfda", {}).get("generic_name", [""])[0] if r.get("openfda", {}).get("generic_name") else ""
            purpose = r.get("purpose", [""])[0] if r.get("purpose") else ""
            lines.append(f"{brand} ({generic}): {purpose[:200]}")
        return "FDA DRUG DATA:\n" + "\n".join(lines) if lines else ""

    # -- RxNorm --

    async def _search_rxnorm(self, terms: str) -> str:
        data = await self._http_get(
            "https://rxnav.nlm.nih.gov/REST/drugs.json",
            params={"name": terms.split()[0]})
        if not data or not isinstance(data, dict):
            return ""
        groups = data.get("drugGroup", {}).get("conceptGroup", [])
        if not groups:
            return ""
        lines = []
        for g in groups:
            for prop in g.get("conceptProperties", [])[:3]:
                rxcui = prop.get("rxcui", "")
                name = prop.get("name", "")
                tty = prop.get("tty", "")
                lines.append(f"[RxCUI:{rxcui}] {name} ({tty})")
        return "RXNORM:\n" + "\n".join(lines[:5]) if lines else ""

    # -- GDC (Cancer Genomics) --

    async def _search_gdc(self, terms: str) -> str:
        data = await self._http_post(
            "https://api.gdc.cancer.gov/genes",
            json_data={"filters": {"op": "in", "content": {"field": "symbol",
                       "value": [terms.split()[0].upper()]}},
                       "size": 3, "fields": "symbol,name,gene_id,biotype"})
        if not data or not isinstance(data, dict):
            return ""
        hits = data.get("data", {}).get("hits", [])
        if not hits:
            return ""
        lines = []
        for h in hits[:3]:
            lines.append(f"[{h.get('gene_id', '')}] {h.get('symbol', '')} - {h.get('name', '')} ({h.get('biotype', '')})")
        return "GDC CANCER GENOMICS:\n" + "\n".join(lines)

    # -- cBioPortal --

    async def _search_cbioportal(self, terms: str) -> str:
        first_term = terms.split()[0].upper() if terms.split() else terms.upper()
        data = await self._http_get(
            f"https://www.cbioportal.org/api/genes/{first_term}",
            headers={"Accept": "application/json"})
        if not data or not isinstance(data, dict):
            return ""
        gene_id = data.get("entrezGeneId", "")
        hugo = data.get("hugoGeneSymbol", "")
        gtype = data.get("type", "")
        return f"CBIOPORTAL: {hugo} (Entrez:{gene_id}, type:{gtype})"


def _xml_tag(xml: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# AI Backend Clients
# ---------------------------------------------------------------------------

class OllamaClient:
    """Local Ollama inference with auto-detection."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("labdojo.ollama")
        self.available = False
        self.model = ""

    async def check_available(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.ollama_host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status != 200:
                        return False
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    if not models:
                        return False
                    preferred = ["llama3", "qwen", "mistral", "gemma", "phi"]
                    for pref in preferred:
                        for m in models:
                            if pref in m.lower():
                                self.model = m
                                self.available = True
                                return True
                    self.model = models[0]
                    self.available = True
                    return True
        except Exception:
            self.available = False
            return False

    async def chat(self, prompt: str, system: str = "",
                   temperature: float = 0.7, messages: list = None) -> str:
        if not self.available:
            await self.check_available()
        if not self.available:
            return ""

        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        else:
            msg_list = []
            if system:
                msg_list.append({"role": "system", "content": system})
            msg_list.extend(messages)
            messages = msg_list

        try:
            payload = {"model": self.model, "messages": messages,
                       "stream": False, "options": {"temperature": temperature}}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.ollama_host}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        return ""
                    data = await resp.json()
                    return data.get("message", {}).get("content", "")
        except Exception as exc:
            self.logger.warning(f"Ollama error: {exc}")
            return ""


class ServerlessClient:
    """Vast.ai serverless inference with cold-start retry."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("labdojo.serverless")

    async def chat(self, prompt: str, system: str = "",
                   temperature: float = 0.7, messages: list = None) -> str:
        if not self.config.vastai_api_key or not self.config.serverless_endpoint_id:
            return ""
        endpoint = self.config.serverless_endpoint_id
        url = f"https://run.vast.ai/route/{endpoint}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.vastai_api_key}",
                   "Content-Type": "application/json"}

        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        else:
            msg_list = []
            if system:
                msg_list.append({"role": "system", "content": system})
            msg_list.extend(messages)
            messages = msg_list

        payload = {"model": "default", "messages": messages,
                   "temperature": temperature, "max_tokens": 2048}

        for attempt in range(4):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, json=payload, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=90)
                    ) as resp:
                        data = await resp.json()
                        if "choices" in data and data["choices"]:
                            return data["choices"][0].get("message", {}).get("content", "")
                        status_msg = str(data.get("status", ""))
                        if "workers: 0" in status_msg or "loading" in status_msg:
                            self.logger.info(f"Cold start (attempt {attempt + 1}/4), waiting 15s")
                            await asyncio.sleep(15)
                            continue
                        return ""
            except Exception as exc:
                self.logger.warning(f"Serverless error: {exc}")
                if attempt < 3:
                    await asyncio.sleep(5)
        return ""


class OpenAIClient:
    """OpenAI-compatible API client."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("labdojo.openai")

    async def chat(self, prompt: str, system: str = "",
                   temperature: float = 0.7, messages: list = None) -> str:
        if not self.config.openai_api_key:
            return ""
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/")
        url = f"{base_url}/chat/completions" if "/v1" in base_url else f"{base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.openai_api_key}",
                   "Content-Type": "application/json"}

        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        else:
            msg_list = []
            if system:
                msg_list.append({"role": "system", "content": system})
            msg_list.extend(messages)
            messages = msg_list

        payload = {"model": self.config.openai_model or "gpt-4o-mini",
                   "messages": messages, "temperature": temperature, "max_tokens": 2048}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        self.logger.debug(f"OpenAI HTTP {resp.status}")
                        return ""
                    data = await resp.json()
                    if "choices" in data and data["choices"]:
                        return data["choices"][0].get("message", {}).get("content", "")
                    return ""
        except Exception as exc:
            self.logger.warning(f"OpenAI error: {exc}")
            return ""


class AnthropicClient:
    """Anthropic Claude API client."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("labdojo.anthropic")

    async def chat(self, prompt: str, system: str = "",
                   temperature: float = 0.7, messages: list = None) -> str:
        if not self.config.anthropic_api_key:
            return ""
        headers = {"x-api-key": self.config.anthropic_api_key,
                   "Content-Type": "application/json",
                   "anthropic-version": "2023-06-01"}

        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            msg_list = list(messages)
            messages = msg_list

        payload = {"model": self.config.anthropic_model,
                   "messages": messages, "max_tokens": 2048, "temperature": temperature}
        if system:
            payload["system"] = system
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    data = await resp.json()
                    content = data.get("content", [])
                    if content:
                        return content[0].get("text", "")
                    return ""
        except Exception as exc:
            self.logger.warning(f"Anthropic error: {exc}")
            return ""


class InferenceRouter:
    """Routes inference requests through available backends with fallback."""

    def __init__(self, config: Config):
        self.ollama = OllamaClient(config)
        self.serverless = ServerlessClient(config)
        self.openai = OpenAIClient(config)
        self.anthropic = AnthropicClient(config)
        self.logger = logging.getLogger("labdojo.router")

    async def chat(self, prompt: str, system: str = "",
                   temperature: float = 0.7, deterministic: bool = False,
                   messages: list = None) -> tuple[str, str]:
        """Returns (response_text, backend_name). Tries all backends in priority order."""
        if deterministic:
            temperature = 0.0

        backends = [
            ("ollama", self.ollama),
            ("serverless", self.serverless),
            ("openai", self.openai),
            ("anthropic", self.anthropic),
        ]

        for name, client in backends:
            try:
                result = await client.chat(
                    prompt, system=system, temperature=temperature,
                    messages=messages)
                if result and result.strip():
                    self.logger.info(f"Response from {name} ({len(result)} chars)")
                    return result.strip(), name
            except Exception as exc:
                self.logger.debug(f"{name} failed: {exc}")

        return "", "none"

    async def get_status(self) -> dict:
        await self.ollama.check_available()
        return {
            "ollama": {"available": self.ollama.available, "model": self.ollama.model},
            "serverless": {"available": bool(self.serverless.config.vastai_api_key
                                             and self.serverless.config.serverless_endpoint_id)},
            "openai": {"available": bool(self.openai.config.openai_api_key)},
            "anthropic": {"available": bool(self.anthropic.config.anthropic_api_key)},
        }


# ---------------------------------------------------------------------------
# System Prompt and Response Processing
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are Lab Dojo, a research-grade AI assistant for principal investigators and senior researchers in pathology and biomedical sciences. You are designed for use in any research laboratory at any university worldwide.

Core operating principles:
1. Treat every user as a domain expert. Never explain foundational concepts unless explicitly asked. If someone asks about O-GlcNAcylation of NF-kB, discuss the specific transferases, sites, and functional consequences, not what post-translational modifications are.
2. Cite by PMID when grounding data is available. Use inline references like [PMID:12345678]. Do not fabricate PMIDs.
3. When you lack sufficient data, state the gap directly: "No published data found on X" rather than hedging with generic disclaimers.
4. Distinguish clearly between established findings and your inference. Use phrasing like "Based on [PMID:X]..." vs "A plausible mechanism would be..."
5. Never suggest the user "consult an expert" or "form a hypothesis first." They are the expert.
6. Never add generic safety disclaimers, confidence labels, or pedagogical framing.
7. Provide mechanistic depth. Discuss specific residues, binding partners, signaling cascades, and experimental approaches.
8. When relevant, mention contradictions in the literature or open questions.
9. Keep responses focused. Do not pad with background the user already knows.
10. For conversational messages (greetings, thanks, etc.), respond naturally and briefly without launching into research mode."""

_VERBOSITY = {
    "concise": "\nBe direct. Data and conclusions only. No background, no hedging. 2-4 sentences max.",
    "detailed": "\nProvide mechanistic detail with citations. Include relevant methodology considerations. 2-3 paragraphs.",
    "comprehensive": "\nFull analysis: primary findings, contradictions, methodological caveats, open questions, and suggested experimental approaches. No length limit.",
}


def _clean_response(text: str) -> str:
    """Strip LLM artifacts that degrade response quality."""
    if not text:
        return text
    patterns = [
        r"\[(?:HIGH|MODERATE|LOW)\s*CONFIDENCE\]\s*",
        r"Sources to verify:.*?(?:\n|$)",
        r"(?:^|\n)\s*(?:It is important to note that|It's essential to note that|It's worth noting that|Please note that|Remember that|Keep in mind that|It should be noted that)",
        r"(?:^|\n)\s*(?:I recommend consulting|Please consult|You should consult|Seek professional|Contact professionals|Consult with your|I suggest speaking with)",
        r"(?:^|\n)\s*(?:As an AI|I'm an AI|I am an AI|As a language model)",
        r"(?:^|\n)\s*(?:We must first form a hypothesis|Let's start by forming|First, let's establish)",
        r"\*\*Disclaimer\*\*:.*?(?:\n\n|\n$|$)",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ---------------------------------------------------------------------------
# Export Utilities
# ---------------------------------------------------------------------------

def _format_bibtex(papers: list[dict]) -> str:
    entries = []
    for p in papers:
        pmid = p.get("pmid", "unknown")
        authors = p.get("authors", [])
        auth_str = " and ".join(authors[:5]) if authors else "Unknown"
        entries.append(
            f"@article{{pmid{pmid},\n"
            f"  title = {{{p.get('title', '')}}},\n"
            f"  author = {{{auth_str}}},\n"
            f"  journal = {{{p.get('journal', '')}}},\n"
            f"  year = {{{p.get('pub_date', '')}}},\n"
            f"  pmid = {{{pmid}}},\n"
            f"  doi = {{{p.get('doi_url', '').replace('https://doi.org/', '')}}}\n"
            f"}}"
        )
    return "\n\n".join(entries)


def _format_ris(papers: list[dict]) -> str:
    entries = []
    for p in papers:
        lines = ["TY  - JOUR", f"TI  - {p.get('title', '')}"]
        for a in p.get("authors", [])[:10]:
            lines.append(f"AU  - {a}")
        lines.extend([
            f"JO  - {p.get('journal', '')}",
            f"PY  - {p.get('pub_date', '')}",
            f"AN  - PMID:{p.get('pmid', '')}",
            f"DO  - {p.get('doi_url', '').replace('https://doi.org/', '')}",
            "ER  - "
        ])
        entries.append("\n".join(lines))
    return "\n\n".join(entries)


def _format_markdown(papers: list[dict]) -> str:
    lines = ["# Literature Export\n"]
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        lines.append(f"## {i}. {p.get('title', '')}\n")
        lines.append(f"**Authors:** {authors}  ")
        lines.append(f"**Journal:** {p.get('journal', '')} ({p.get('pub_date', '')})  ")
        lines.append(f"**PMID:** {p.get('pmid', '')}  ")
        if p.get("doi_url"):
            lines.append(f"**DOI:** {p['doi_url']}  ")
        if p.get("abstract"):
            lines.append(f"\n{p['abstract'][:500]}\n")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    config = Config.load()
    logger = setup_logging(config)
    kb = KnowledgeBase(config)
    apis = ScienceAPIs(config, kb)
    router = InferenceRouter(config)

    app = FastAPI(title="Lab Dojo", version=__version__,
                  description="Research workstation for pathology and biomedical sciences")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        await router.ollama.check_available()
        if router.ollama.available:
            logger.info(f"Ollama detected: {router.ollama.model}")
        else:
            logger.info("No local Ollama found. Configure a backend in Settings.")

    @app.on_event("shutdown")
    async def shutdown():
        await apis.close()

    # -- Status --

    @app.get("/status")
    async def get_status():
        ai_status = await router.get_status()
        return {
            "version": __version__,
            "ai_backends": ai_status,
            "apis": apis.get_api_status(),
            "stats": kb.get_usage(),
        }

    # -- Chat --

    class ChatRequest(BaseModel):
        message: str
        project_id: Optional[str] = None
        verbosity: str = "detailed"
        deterministic: bool = False

    @app.post("/chat")
    async def chat(req: ChatRequest):
        if not req.message.strip():
            raise HTTPException(400, "Message cannot be empty")

        project_id = req.project_id or "default"
        kb.add_conversation(project_id, "user", req.message)
        start = time.time()

        # Check if this is a casual message (greeting, thanks, etc.)
        intent, casual_response = classify_intent(req.message)
        if intent == "casual":
            kb.add_conversation(project_id, "assistant", casual_response)
            return {"response": casual_response, "source": "system",
                    "grounding": [], "latency": round(time.time() - start, 2)}

        # Check memory for similar questions
        recalled = kb.recall_similar(req.message)
        if recalled:
            answer = recalled["answer"]
            kb.add_conversation(project_id, "assistant", answer)
            return {"response": answer, "source": "memory",
                    "grounding": [], "latency": round(time.time() - start, 2)}

        # Fetch grounding data from science APIs (parallel)
        context, sources = await apis.fetch_grounding_data(req.message)

        # Build conversation history for context
        history = kb.get_conversations(project_id, limit=10)
        conv_messages = []
        for msg in history[:-1]:  # exclude the current message we just added
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                conv_messages.append({"role": role, "content": content[:500]})

        # Build prompt with grounding
        system = _SYSTEM_PROMPT + _VERBOSITY.get(req.verbosity, _VERBOSITY["detailed"])
        prompt = req.message
        if context:
            prompt = f"GROUNDING DATA (cite by PMID where available):\n{context}\n\nQUESTION: {req.message}"

        # Add current message to conversation
        conv_messages.append({"role": "user", "content": prompt})

        # Get AI response with conversation history
        response, backend = await router.chat(
            prompt, system=system,
            temperature=0.7, deterministic=req.deterministic,
            messages=conv_messages if len(conv_messages) > 1 else None)

        if not response and context:
            response = f"No AI backend available. Raw data from {', '.join(sources)}:\n\n{context}"
            backend = "api_data"
        elif not response:
            raise HTTPException(503, "No AI backends configured. Add Ollama, an API key, or serverless endpoint in Settings.")

        response = _clean_response(response)
        kb.add_conversation(project_id, "assistant", response)
        kb.learn_qa(req.message, response, backend)
        kb.record_usage(local=1 if backend == 'ollama' else 0,
                        serverless=1 if backend in ('serverless', 'openai', 'anthropic') else 0)

        return {
            "response": response,
            "source": backend,
            "grounding": sources,
            "latency": round(time.time() - start, 2),
        }

    # -- Hypothesis Generation --

    class HypothesisRequest(BaseModel):
        message: str
        project_id: Optional[str] = None

    @app.post("/hypothesis")
    async def generate_hypothesis(req: HypothesisRequest):
        if not req.message.strip():
            raise HTTPException(400, "Message cannot be empty")

        context, sources = await apis.fetch_grounding_data(req.message)
        system = (
            "Generate a mechanistically-grounded hypothesis based on the provided data. "
            "Include: (1) the hypothesis statement, (2) key predictions that would confirm or refute it, "
            "(3) suggested experimental approaches. Cite PMIDs where available. "
            "Do not explain what a hypothesis is. Do not suggest consulting the literature."
        )
        prompt = req.message
        if context:
            prompt = f"DATA:\n{context}\n\nTOPIC: {req.message}"

        response, backend = await router.chat(prompt, system=system)
        if not response:
            raise HTTPException(503, "No AI backend available")
        response = _clean_response(response)
        return {"hypothesis": response, "source": backend, "grounding": sources}

    # -- Papers --

    @app.get("/papers/search")
    async def search_papers(query: str, max_results: int = 10):
        results = await apis.search_pubmed(query, max_results=max_results)
        kb.record_usage(api=1)
        return {"papers": results, "count": len(results), "query": query}

    # -- Projects --

    class ProjectCreate(BaseModel):
        name: str
        description: str = ""
        key_terms: str = ""

    @app.post("/projects")
    async def create_project(req: ProjectCreate):
        pid = kb.create_project(req.name, req.description, req.key_terms)
        return {"id": pid, "name": req.name}

    @app.get("/projects")
    async def list_projects():
        return {"projects": kb.get_projects()}

    @app.get("/projects/{project_id}")
    async def get_project(project_id: str):
        proj = kb.get_project(project_id)
        if not proj:
            raise HTTPException(404, "Project not found")
        return proj

    @app.delete("/projects/{project_id}")
    async def delete_project(project_id: str):
        kb.delete_project(project_id)
        return {"deleted": project_id}

    # -- Decisions --

    class DecisionEntry(BaseModel):
        decision: str
        rationale: str = ""
        evidence: str = ""

    @app.post("/projects/{project_id}/decisions")
    async def add_decision(project_id: str, req: DecisionEntry):
        kb.add_decision(project_id, req.decision, req.rationale, req.evidence)
        return {"status": "recorded"}

    @app.get("/projects/{project_id}/decisions")
    async def get_decisions(project_id: str):
        return {"decisions": kb.get_decisions(project_id)}

    # -- Export --

    @app.get("/export/bibtex")
    async def export_bibtex(pmids: str):
        id_list = [p.strip() for p in pmids.split(",") if p.strip()]
        papers = await apis.search_pubmed(" OR ".join(f"{p}[uid]" for p in id_list), max_results=len(id_list))
        content = _format_bibtex(papers)
        return Response(content=content, media_type="text/plain",
                        headers={"Content-Disposition": "attachment; filename=export.bib"})

    @app.get("/export/ris")
    async def export_ris(pmids: str):
        id_list = [p.strip() for p in pmids.split(",") if p.strip()]
        papers = await apis.search_pubmed(" OR ".join(f"{p}[uid]" for p in id_list), max_results=len(id_list))
        content = _format_ris(papers)
        return Response(content=content, media_type="text/plain",
                        headers={"Content-Disposition": "attachment; filename=export.ris"})

    @app.get("/export/markdown")
    async def export_markdown(pmids: str):
        id_list = [p.strip() for p in pmids.split(",") if p.strip()]
        papers = await apis.search_pubmed(" OR ".join(f"{p}[uid]" for p in id_list), max_results=len(id_list))
        content = _format_markdown(papers)
        return Response(content=content, media_type="text/markdown",
                        headers={"Content-Disposition": "attachment; filename=export.md"})

    @app.get("/export/conversation")
    async def export_conversation():
        history = kb.get_conversations("default", limit=200)
        lines = []
        for msg in history:
            role = msg.get("role", "").upper()
            text = msg.get("content", "")
            ts = msg.get("created_at", "")
            lines.append(f"[{ts}] {role}: {text}")
        return Response(content="\n\n".join(lines), media_type="text/plain",
                        headers={"Content-Disposition": "attachment; filename=conversation.txt"})

    # -- Pipeline --

    class PipelineRequest(BaseModel):
        pipeline_type: str
        query: str
        params: dict = {}

    @app.post("/pipeline/run")
    async def run_pipeline(req: PipelineRequest):
        run_id = kb.create_pipeline_run(req.pipeline_type, {"query": req.query, **req.params})

        if req.pipeline_type == "literature_review":
            papers = await apis.search_pubmed(req.query, max_results=10)
            context = "\n".join(f"[PMID:{p['pmid']}] {p['title']} - {p.get('abstract', '')[:300]}" for p in papers)
            system = "Synthesize these papers into a structured literature review. Cite by PMID. Include: key findings, methodological approaches, contradictions, and gaps."
            prompt = f"PAPERS:\n{context}\n\nTOPIC: {req.query}"
            response, backend = await router.chat(prompt, system=system)
            if not response:
                response = f"Literature data collected ({len(papers)} papers). No AI backend for synthesis."
            result = {"papers": len(papers), "synthesis": _clean_response(response)}

        elif req.pipeline_type == "protein_analysis":
            term = req.query.split()[0] if req.query.split() else req.query
            uniprot, pdb, string_data, alphafold = await asyncio.gather(
                apis._search_uniprot(term),
                apis._search_pdb(term),
                apis._search_string(term),
                apis._search_alphafold(term),
            )
            context = "\n\n".join(filter(None, [uniprot, pdb, string_data, alphafold]))
            system = "Provide a structural and functional analysis of this protein based on the data. Include known domains, interactions, and structural features."
            prompt = f"DATA:\n{context}\n\nPROTEIN: {req.query}"
            response, backend = await router.chat(prompt, system=system)
            if not response:
                response = context or "No data found"
            result = {"analysis": _clean_response(response), "databases_queried": ["UniProt", "PDB", "STRING", "AlphaFold"]}

        elif req.pipeline_type == "drug_target":
            chembl, pubchem, fda, rxnorm, trials = await asyncio.gather(
                apis._search_chembl(req.query),
                apis._search_pubchem(req.query),
                apis._search_openfda(req.query),
                apis._search_rxnorm(req.query),
                apis._search_clinicaltrials(req.query),
            )
            context = "\n\n".join(filter(None, [chembl, pubchem, fda, rxnorm, trials]))
            system = "Analyze this compound/target from a drug development perspective. Include known pharmacology, clinical status, and safety data."
            prompt = f"DATA:\n{context}\n\nQUERY: {req.query}"
            response, backend = await router.chat(prompt, system=system)
            if not response:
                response = context or "No data found"
            result = {"analysis": _clean_response(response), "databases_queried": ["ChEMBL", "PubChem", "openFDA", "RxNorm", "ClinicalTrials"]}

        elif req.pipeline_type == "pathway_analysis":
            reactome, kegg, go = await asyncio.gather(
                apis._search_reactome(req.query),
                apis._search_kegg(req.query),
                apis._search_gene_ontology(req.query),
            )
            context = "\n\n".join(filter(None, [reactome, kegg, go]))
            system = "Analyze the pathway data. Describe the signaling cascade, key nodes, and regulatory mechanisms."
            prompt = f"DATA:\n{context}\n\nPATHWAY: {req.query}"
            response, backend = await router.chat(prompt, system=system)
            if not response:
                response = context or "No data found"
            result = {"analysis": _clean_response(response), "databases_queried": ["Reactome", "KEGG", "Gene Ontology"]}

        elif req.pipeline_type == "cancer_genomics":
            gdc, cbio, pubmed = await asyncio.gather(
                apis._search_gdc(req.query),
                apis._search_cbioportal(req.query),
                apis._search_pubmed(f"{req.query} cancer mutation"),
            )
            context = "\n\n".join(filter(None, [gdc, cbio, pubmed]))
            system = "Analyze the cancer genomics data. Discuss mutation frequency, clinical significance, and therapeutic implications."
            prompt = f"DATA:\n{context}\n\nGENE/TARGET: {req.query}"
            response, backend = await router.chat(prompt, system=system)
            if not response:
                response = context or "No data found"
            result = {"analysis": _clean_response(response), "databases_queried": ["GDC", "cBioPortal", "PubMed"]}

        else:
            raise HTTPException(400, f"Unknown pipeline type: {req.pipeline_type}. "
                                     f"Available: literature_review, protein_analysis, drug_target, pathway_analysis, cancer_genomics")

        kb.update_pipeline_run(run_id, "completed", result)
        return {"run_id": run_id, "status": "completed", "result": result}

    @app.get("/pipeline/history")
    async def pipeline_history():
        return {"runs": kb.get_pipeline_runs()}

    # -- Monitor --

    @app.post("/monitor/topics")
    async def add_monitor_topic(topic: dict):
        name = topic.get("topic", "") or topic.get("name", "")
        if not name:
            raise HTTPException(400, "Topic name required")
        kb.add_monitored_topic(name)
        return {"status": "added", "name": name}

    @app.get("/monitor/topics")
    async def list_monitor_topics():
        return {"topics": kb.get_monitored_topics()}

    @app.delete("/monitor/topics/{topic_id}")
    async def remove_monitor_topic(topic_id: int):
        kb.remove_monitored_topic(topic_id)
        return {"status": "removed"}

    @app.post("/monitor/check")
    async def check_monitored_topics():
        topics = kb.get_monitored_topics()
        results = {}
        for t in topics:
            papers = await apis.search_pubmed(t["topic"], max_results=3)
            results[t["topic"]] = {"new_papers": len(papers), "papers": papers}
        return {"results": results}

    # -- Learning --

    @app.get("/learning/stats")
    async def learning_stats():
        return kb.get_usage()

    @app.post("/learning/clear_bad")
    async def clear_bad_data():
        kb.clear_bad_data()
        return {"status": "cleared"}

    @app.post("/learning/reset")
    async def reset_learning():
        kb.reset_learning()
        return {"status": "reset"}

    @app.post("/conversation/clear")
    async def clear_conversations():
        kb.clear_conversations()
        return {"status": "cleared"}

    # -- Settings --

    @app.get("/settings")
    async def get_settings():
        d = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        for sk in _SENSITIVE_KEYS:
            if sk in d and d[sk]:
                d[sk] = "configured"
            elif sk in d:
                d[sk] = ""
        return d

    @app.post("/settings/update")
    async def update_settings(updates: dict):
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        config.save()
        if "ollama_host" in updates:
            await router.ollama.check_available()
        return {"status": "saved"}

    @app.get("/monitor/alerts")
    async def get_alerts():
        return {"alerts": kb.get_alerts()}

    # -- API Catalog --

    @app.get("/apis")
    async def list_apis():
        return {"apis": apis.get_api_status()}

    # -- Dashboard --

    @app.get("/")
    async def dashboard():
        return HTMLResponse(get_dashboard_html())

    return app


# ---------------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------------

def get_dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lab Dojo - Research Workstation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #1c2128;
    --bg-card: #21262d;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --accent: #58a6ff;
    --accent-hover: #79c0ff;
    --green: #3fb950;
    --red: #f85149;
    --orange: #d29922;
    --purple: #bc8cff;
    --sidebar-width: 240px;
    --radius: 8px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, sans-serif; background: var(--bg-primary); color: var(--text-primary); height: 100vh; overflow: hidden; display: flex; }

/* Sidebar */
.sidebar { width: var(--sidebar-width); background: var(--bg-secondary); border-right: 1px solid var(--border); display: flex; flex-direction: column; flex-shrink: 0; }
.sidebar-brand { padding: 20px; border-bottom: 1px solid var(--border); }
.sidebar-brand h1 { font-size: 18px; font-weight: 700; color: var(--accent); letter-spacing: -0.5px; }
.sidebar-brand span { font-size: 11px; color: var(--text-muted); display: block; margin-top: 2px; }
.sidebar-nav { flex: 1; padding: 12px 8px; overflow-y: auto; }
.nav-item { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: var(--radius); cursor: pointer; color: var(--text-secondary); font-size: 13px; font-weight: 500; transition: all 0.15s; margin-bottom: 2px; }
.nav-item:hover { background: var(--bg-tertiary); color: var(--text-primary); }
.nav-item.active { background: rgba(88,166,255,0.1); color: var(--accent); }
.nav-icon { width: 18px; text-align: center; font-size: 14px; }
.sidebar-footer { padding: 12px; border-top: 1px solid var(--border); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.status-dot.on { background: var(--green); }
.status-dot.off { background: var(--red); }
.sidebar-footer .status-line { font-size: 11px; color: var(--text-muted); padding: 3px 8px; }

/* Main content */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.page { display: none; flex-direction: column; height: 100%; overflow: hidden; }
.page.active { display: flex; }
.page-header { padding: 20px 24px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.page-header h2 { font-size: 20px; font-weight: 600; }
.page-header p { font-size: 13px; color: var(--text-secondary); margin-top: 4px; }
.page-body { flex: 1; overflow-y: auto; padding: 20px 24px; }

/* Chat */
.chat-container { display: flex; flex-direction: column; height: 100%; }
.chat-controls { padding: 12px 24px; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: center; flex-shrink: 0; flex-wrap: wrap; }
.chat-controls select, .chat-controls label { font-size: 12px; color: var(--text-secondary); }
.chat-controls select { background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text-primary); padding: 4px 8px; border-radius: 4px; }
.chat-controls .toggle { display: flex; align-items: center; gap: 4px; }
.chat-controls input[type="checkbox"] { accent-color: var(--accent); }
.chat-messages { flex: 1; overflow-y: auto; padding: 16px 24px; }
.message { margin-bottom: 16px; max-width: 85%; }
.message.user { margin-left: auto; }
.message.assistant { margin-right: auto; }
.message-bubble { padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.6; }
.message.user .message-bubble { background: var(--accent); color: #fff; border-bottom-right-radius: 4px; }
.message.assistant .message-bubble { background: var(--bg-card); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
.message-meta { font-size: 11px; color: var(--text-muted); margin-top: 4px; padding: 0 4px; }
.message.user .message-meta { text-align: right; }
.message-bubble h1,.message-bubble h2,.message-bubble h3 { margin: 12px 0 6px; font-size: 15px; }
.message-bubble p { margin: 6px 0; }
.message-bubble code { background: var(--bg-tertiary); padding: 2px 5px; border-radius: 3px; font-size: 13px; }
.message-bubble pre { background: var(--bg-tertiary); padding: 10px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }
.message-bubble a { color: var(--accent); }
.chat-input-area { padding: 16px 24px; border-top: 1px solid var(--border); flex-shrink: 0; }
.chat-input-row { display: flex; gap: 8px; }
.chat-input-row input { flex: 1; background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text-primary); padding: 12px 16px; border-radius: var(--radius); font-size: 14px; outline: none; }
.chat-input-row input:focus { border-color: var(--accent); }
.btn { padding: 10px 20px; border-radius: var(--radius); border: none; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.15s; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: var(--accent-hover); }
.btn-secondary { background: var(--bg-card); color: var(--text-primary); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--bg-tertiary); }
.btn-danger { background: var(--red); color: #fff; }
.btn-sm { padding: 6px 12px; font-size: 12px; }

/* Papers */
.search-bar { display: flex; gap: 8px; margin-bottom: 16px; }
.search-bar input { flex: 1; background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text-primary); padding: 10px 14px; border-radius: var(--radius); font-size: 14px; outline: none; }
.paper-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; margin-bottom: 10px; }
.paper-card h3 { font-size: 14px; font-weight: 600; margin-bottom: 6px; line-height: 1.4; }
.paper-card .meta { font-size: 12px; color: var(--text-secondary); }
.paper-card .abstract { font-size: 13px; color: var(--text-secondary); margin-top: 8px; line-height: 1.5; }
.paper-card a { color: var(--accent); text-decoration: none; font-size: 12px; }

/* Projects */
.project-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; margin-bottom: 10px; cursor: pointer; transition: border-color 0.15s; }
.project-card:hover { border-color: var(--accent); }
.project-card h3 { font-size: 15px; font-weight: 600; }
.project-card p { font-size: 13px; color: var(--text-secondary); margin-top: 4px; }

/* Pipeline */
.pipeline-types { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px; }
.pipeline-type { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; cursor: pointer; transition: all 0.15s; text-align: center; }
.pipeline-type:hover { border-color: var(--accent); background: rgba(88,166,255,0.05); }
.pipeline-type h4 { font-size: 14px; margin-bottom: 4px; }
.pipeline-type p { font-size: 12px; color: var(--text-secondary); }

/* Settings */
.settings-section { margin-bottom: 24px; }
.settings-section h3 { font-size: 15px; font-weight: 600; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
.setting-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.setting-row label { width: 160px; font-size: 13px; color: var(--text-secondary); flex-shrink: 0; }
.setting-row input { flex: 1; background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text-primary); padding: 8px 12px; border-radius: 6px; font-size: 13px; outline: none; }
.setting-row input:focus { border-color: var(--accent); }

/* API Grid */
.api-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }
.api-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 12px; }
.api-card h4 { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
.api-card p { font-size: 11px; color: var(--text-secondary); }
.api-card .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; }
.badge-free { background: rgba(63,185,80,0.15); color: var(--green); }

/* Loading */
.loading { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.typing-indicator { display: flex; gap: 4px; padding: 12px 16px; }
.typing-indicator span { width: 6px; height: 6px; background: var(--text-muted); border-radius: 50%; animation: bounce 1.4s infinite; }
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-6px); } }

/* Welcome */
.welcome-msg { text-align: center; padding: 60px 40px; color: var(--text-secondary); }
.welcome-msg h3 { font-size: 18px; color: var(--text-primary); margin-bottom: 12px; }
.welcome-msg p { font-size: 14px; line-height: 1.8; max-width: 500px; margin: 0 auto; }
.welcome-msg .examples { margin-top: 20px; text-align: left; display: inline-block; }
.welcome-msg .example { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 16px; margin: 6px 0; cursor: pointer; font-size: 13px; transition: border-color 0.15s; }
.welcome-msg .example:hover { border-color: var(--accent); color: var(--accent); }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
</head>
<body>

<div class="sidebar">
    <div class="sidebar-brand">
        <h1>Lab Dojo</h1>
        <span>Pathology Research Workstation</span>
    </div>
    <div class="sidebar-nav">
        <div class="nav-item active" onclick="showPage('chat')">
            <span class="nav-icon">&#128172;</span> Chat
        </div>
        <div class="nav-item" onclick="showPage('papers')">
            <span class="nav-icon">&#128196;</span> Papers
        </div>
        <div class="nav-item" onclick="showPage('pipelines')">
            <span class="nav-icon">&#9881;</span> Pipelines
        </div>
        <div class="nav-item" onclick="showPage('projects')">
            <span class="nav-icon">&#128193;</span> Projects
        </div>
        <div class="nav-item" onclick="showPage('apis')">
            <span class="nav-icon">&#127760;</span> APIs
        </div>
        <div class="nav-item" onclick="showPage('settings')">
            <span class="nav-icon">&#9881;</span> Settings
        </div>
    </div>
    <div class="sidebar-footer" id="sidebar-status">
        <div class="status-line"><span class="status-dot off" id="dot-ai"></span><span id="lbl-ai">Checking AI...</span></div>
        <div class="status-line"><span class="status-dot on"></span>20 APIs connected</div>
    </div>
</div>

<div class="main">
    <!-- Chat Page -->
    <div class="page active" id="page-chat">
        <div class="page-header">
            <h2>Chat</h2>
            <p>Ask research questions grounded in 20 biomedical databases</p>
        </div>
        <div class="chat-container">
            <div class="chat-controls">
                <label>Verbosity:</label>
                <select id="verbosity">
                    <option value="concise">Concise</option>
                    <option value="detailed" selected>Detailed</option>
                    <option value="comprehensive">Comprehensive</option>
                </select>
                <div class="toggle">
                    <input type="checkbox" id="deterministic">
                    <label for="deterministic">Deterministic</label>
                </div>
            </div>
            <div class="chat-messages" id="chat-messages">
                <div class="welcome-msg" id="welcome">
                    <h3>Welcome to Lab Dojo</h3>
                    <p>Your AI research assistant with access to 20 biomedical databases. Ask any research question to get started.</p>
                    <div class="examples">
                        <div class="example" onclick="fillChat('What is the role of BRCA1 in DNA damage repair?')">What is the role of BRCA1 in DNA damage repair?</div>
                        <div class="example" onclick="fillChat('Compare PD-1 and PD-L1 inhibitors in melanoma treatment')">Compare PD-1 and PD-L1 inhibitors in melanoma treatment</div>
                        <div class="example" onclick="fillChat('What are the latest findings on TP53 mutations in colorectal cancer?')">What are the latest findings on TP53 mutations in colorectal cancer?</div>
                    </div>
                </div>
            </div>
            <div class="chat-input-area">
                <div class="chat-input-row">
                    <input type="text" id="chat-input" placeholder="Ask a research question..." onkeydown="if(event.key==='Enter')sendChat()">
                    <button class="btn btn-primary" onclick="sendChat()">Send</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Papers Page -->
    <div class="page" id="page-papers">
        <div class="page-header">
            <h2>Papers</h2>
            <p>Search PubMed with citation verification</p>
        </div>
        <div class="page-body">
            <div class="search-bar">
                <input type="text" id="paper-query" placeholder="Search PubMed..." onkeydown="if(event.key==='Enter')searchPapers()">
                <button class="btn btn-primary" onclick="searchPapers()">Search</button>
                <button class="btn btn-secondary btn-sm" onclick="exportResults('bibtex')">BibTeX</button>
                <button class="btn btn-secondary btn-sm" onclick="exportResults('ris')">RIS</button>
                <button class="btn btn-secondary btn-sm" onclick="exportResults('markdown')">Markdown</button>
            </div>
            <div id="paper-results"></div>
        </div>
    </div>

    <!-- Pipelines Page -->
    <div class="page" id="page-pipelines">
        <div class="page-header">
            <h2>Pipelines</h2>
            <p>Multi-step automated research workflows</p>
        </div>
        <div class="page-body">
            <div class="pipeline-types">
                <div class="pipeline-type" onclick="showPipelineForm('literature_review')">
                    <h4>Literature Review</h4>
                    <p>Search, collect, and synthesize papers</p>
                </div>
                <div class="pipeline-type" onclick="showPipelineForm('protein_analysis')">
                    <h4>Protein Analysis</h4>
                    <p>UniProt + PDB + STRING + AlphaFold</p>
                </div>
                <div class="pipeline-type" onclick="showPipelineForm('drug_target')">
                    <h4>Drug/Target</h4>
                    <p>ChEMBL + PubChem + FDA + Trials</p>
                </div>
                <div class="pipeline-type" onclick="showPipelineForm('pathway_analysis')">
                    <h4>Pathway Analysis</h4>
                    <p>Reactome + KEGG + Gene Ontology</p>
                </div>
                <div class="pipeline-type" onclick="showPipelineForm('cancer_genomics')">
                    <h4>Cancer Genomics</h4>
                    <p>GDC + cBioPortal + Literature</p>
                </div>
            </div>
            <div id="pipeline-form" style="display:none;">
                <div class="search-bar">
                    <input type="text" id="pipeline-query" placeholder="Enter query...">
                    <button class="btn btn-primary" onclick="runPipeline()">Run Pipeline</button>
                </div>
            </div>
            <div id="pipeline-results"></div>
        </div>
    </div>

    <!-- Projects Page -->
    <div class="page" id="page-projects">
        <div class="page-header">
            <h2>Projects</h2>
            <p>Persistent research context and decision logs</p>
        </div>
        <div class="page-body">
            <div style="margin-bottom: 16px;">
                <button class="btn btn-primary btn-sm" onclick="showNewProject()">New Project</button>
            </div>
            <div id="new-project-form" style="display:none; margin-bottom: 16px;">
                <div class="setting-row"><label>Name:</label><input type="text" id="proj-name"></div>
                <div class="setting-row"><label>Description:</label><input type="text" id="proj-desc"></div>
                <div class="setting-row"><label>Key Terms:</label><input type="text" id="proj-terms" placeholder="comma-separated"></div>
                <button class="btn btn-primary btn-sm" onclick="createProject()" style="margin-top:8px;">Create</button>
            </div>
            <div id="project-list"></div>
        </div>
    </div>

    <!-- APIs Page -->
    <div class="page" id="page-apis">
        <div class="page-header">
            <h2>Connected APIs</h2>
            <p>20 biomedical databases, all free, no keys required</p>
        </div>
        <div class="page-body">
            <div class="api-grid" id="api-grid"></div>
        </div>
    </div>

    <!-- Settings Page -->
    <div class="page" id="page-settings">
        <div class="page-header">
            <h2>Settings</h2>
            <p>Configure AI backends and manage data</p>
        </div>
        <div class="page-body">
            <div class="settings-section">
                <h3>AI Backends</h3>
                <div class="setting-row"><label>Ollama Host:</label><input type="text" id="set-ollama" value="http://localhost:11434"></div>
                <div class="setting-row"><label>OpenAI API Key:</label><input type="password" id="set-openai" placeholder="sk-..."></div>
                <div class="setting-row"><label>Anthropic API Key:</label><input type="password" id="set-anthropic" placeholder="sk-ant-..."></div>
                <div class="setting-row"><label>Vast.ai API Key:</label><input type="password" id="set-vastai" placeholder="Optional"></div>
                <div class="setting-row"><label>Serverless Endpoint:</label><input type="text" id="set-endpoint" placeholder="Endpoint ID"></div>
                <button class="btn btn-primary btn-sm" onclick="saveSettings()">Save</button>
            </div>
            <div class="settings-section">
                <h3>Data Management</h3>
                <button class="btn btn-secondary btn-sm" onclick="clearConversation()">Clear Chat History</button>
                <button class="btn btn-secondary btn-sm" onclick="clearBadData()">Clean Bad Data</button>
                <button class="btn btn-danger btn-sm" onclick="resetLearning()">Reset All Learning</button>
                <button class="btn btn-secondary btn-sm" onclick="exportConversation()">Export Conversation</button>
            </div>
        </div>
    </div>
</div>

<script>
let currentPage = 'chat';
let currentPipelineType = '';
let lastPaperPmids = [];

function showPage(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    document.querySelectorAll('.nav-item')[['chat','papers','pipelines','projects','apis','settings'].indexOf(page)].classList.add('active');
    currentPage = page;
    if (page === 'apis') loadApis();
    if (page === 'projects') loadProjects();
    if (page === 'settings') loadSettings();
}

function fillChat(text) {
    document.getElementById('chat-input').value = text;
    document.getElementById('chat-input').focus();
}

function addMessage(role, text, meta) {
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.remove();
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'message ' + role;
    let html = '<div class="message-bubble">';
    if (role === 'assistant') {
        try { html += marked.parse(text); } catch(e) { html += text; }
    } else {
        html += text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
    html += '</div>';
    if (meta) html += '<div class="message-meta">' + meta + '</div>';
    div.innerHTML = html;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function showTyping() {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = 'typing';
    div.innerHTML = '<div class="message-bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div>';
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function hideTyping() {
    const el = document.getElementById('typing');
    if (el) el.remove();
}

async function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';
    addMessage('user', msg, new Date().toLocaleTimeString());
    showTyping();
    try {
        const resp = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: msg,
                verbosity: document.getElementById('verbosity').value,
                deterministic: document.getElementById('deterministic').checked
            })
        });
        hideTyping();
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({detail: 'Request failed'}));
            addMessage('assistant', 'Error: ' + (err.detail || 'Unknown error'), '');
            return;
        }
        const data = await resp.json();
        let meta = 'via ' + data.source;
        if (data.grounding && data.grounding.length) meta += ' | grounded in: ' + data.grounding.join(', ');
        if (data.latency) meta += ' | ' + data.latency + 's';
        addMessage('assistant', data.response, meta);
    } catch(e) {
        hideTyping();
        addMessage('assistant', 'Connection error. Is the server running?', '');
    }
}

async function searchPapers() {
    const query = document.getElementById('paper-query').value.trim();
    if (!query) return;
    const container = document.getElementById('paper-results');
    container.innerHTML = '<div class="loading"></div> Searching PubMed...';
    try {
        const resp = await fetch('/papers/search?query=' + encodeURIComponent(query) + '&max_results=10');
        const data = await resp.json();
        lastPaperPmids = data.papers.map(p => p.pmid);
        if (!data.papers.length) { container.innerHTML = '<p style="color:var(--text-muted)">No results found.</p>'; return; }
        container.innerHTML = data.papers.map(p => {
            const authors = (p.authors || []).slice(0, 3).join(', ') + (p.authors && p.authors.length > 3 ? ' et al.' : '');
            return '<div class="paper-card">' +
                '<h3>' + (p.title || 'Untitled') + '</h3>' +
                '<div class="meta">PMID: ' + p.pmid + ' | ' + authors + ' | ' + (p.journal || '') + ' (' + (p.pub_date || '') + ')</div>' +
                (p.doi_url ? '<a href="' + p.doi_url + '" target="_blank">DOI</a> | ' : '') +
                '<a href="https://pubmed.ncbi.nlm.nih.gov/' + p.pmid + '/" target="_blank">PubMed</a>' +
                (p.abstract ? '<div class="abstract">' + p.abstract.substring(0, 300) + '...</div>' : '') +
                '</div>';
        }).join('');
    } catch(e) {
        container.innerHTML = '<p style="color:var(--red)">Search failed.</p>';
    }
}

function exportResults(format) {
    if (!lastPaperPmids.length) { alert('Search for papers first.'); return; }
    window.open('/export/' + format + '?pmids=' + lastPaperPmids.join(','));
}

function showPipelineForm(type) {
    currentPipelineType = type;
    document.getElementById('pipeline-form').style.display = 'block';
    document.getElementById('pipeline-query').placeholder = 'Enter query for ' + type.replace('_', ' ') + '...';
    document.getElementById('pipeline-query').focus();
}

async function runPipeline() {
    const query = document.getElementById('pipeline-query').value.trim();
    if (!query || !currentPipelineType) return;
    const container = document.getElementById('pipeline-results');
    container.innerHTML = '<div class="loading"></div> Running ' + currentPipelineType.replace('_', ' ') + '... This may take 30-60 seconds.';
    try {
        const resp = await fetch('/pipeline/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ pipeline_type: currentPipelineType, query: query })
        });
        const data = await resp.json();
        if (!resp.ok) { container.innerHTML = '<p style="color:var(--red)">' + (data.detail || 'Pipeline failed') + '</p>'; return; }
        const result = data.result || {};
        let html = '<div class="paper-card"><h3>Pipeline: ' + currentPipelineType.replace('_', ' ') + '</h3>';
        if (result.databases_queried) html += '<div class="meta">Databases: ' + result.databases_queried.join(', ') + '</div>';
        const text = result.synthesis || result.analysis || JSON.stringify(result, null, 2);
        try { html += '<div style="margin-top:12px;">' + marked.parse(text) + '</div>'; } catch(e) { html += '<pre>' + text + '</pre>'; }
        html += '</div>';
        container.innerHTML = html;
    } catch(e) {
        container.innerHTML = '<p style="color:var(--red)">Pipeline error.</p>';
    }
}

async function loadApis() {
    try {
        const resp = await fetch('/apis');
        const data = await resp.json();
        const grid = document.getElementById('api-grid');
        grid.innerHTML = Object.entries(data.apis).map(([id, api]) =>
            '<div class="api-card">' +
            '<h4>' + api.name + '</h4>' +
            '<p>' + api.description + '</p>' +
            '<span class="badge badge-free">' + api.rate_limit + '</span>' +
            '</div>'
        ).join('');
    } catch(e) {}
}

async function loadProjects() {
    try {
        const resp = await fetch('/projects');
        const data = await resp.json();
        const container = document.getElementById('project-list');
        if (!data.projects.length) { container.innerHTML = '<p style="color:var(--text-muted)">No projects yet.</p>'; return; }
        container.innerHTML = data.projects.map(p =>
            '<div class="project-card">' +
            '<h3>' + p.name + '</h3>' +
            '<p>' + (p.description || 'No description') + '</p>' +
            '<div class="meta" style="font-size:11px;color:var(--text-muted);margin-top:6px;">Status: ' + p.status + ' | Created: ' + (p.created_at || '') + '</div>' +
            '<button class="btn btn-danger btn-sm" style="margin-top:8px;" onclick="deleteProject(\'' + p.id + '\')">Delete</button>' +
            '</div>'
        ).join('');
    } catch(e) {}
}

function showNewProject() {
    document.getElementById('new-project-form').style.display = document.getElementById('new-project-form').style.display === 'none' ? 'block' : 'none';
}

async function createProject() {
    const name = document.getElementById('proj-name').value.trim();
    if (!name) return;
    await fetch('/projects', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            name: name,
            description: document.getElementById('proj-desc').value,
            key_terms: document.getElementById('proj-terms').value
        })
    });
    document.getElementById('new-project-form').style.display = 'none';
    loadProjects();
}

async function deleteProject(id) {
    if (!confirm('Delete this project and all its data?')) return;
    await fetch('/projects/' + id, { method: 'DELETE' });
    loadProjects();
}

async function loadSettings() {
    try {
        const resp = await fetch('/settings');
        const data = await resp.json();
        document.getElementById('set-ollama').value = data.ollama_host || '';
        document.getElementById('set-openai').value = data.openai_api_key === 'configured' ? '' : (data.openai_api_key || '');
        document.getElementById('set-openai').placeholder = data.openai_api_key === 'configured' ? 'Configured (enter new to change)' : 'sk-...';
        document.getElementById('set-anthropic').value = data.anthropic_api_key === 'configured' ? '' : (data.anthropic_api_key || '');
        document.getElementById('set-anthropic').placeholder = data.anthropic_api_key === 'configured' ? 'Configured (enter new to change)' : 'sk-ant-...';
        document.getElementById('set-vastai').value = data.vastai_api_key === 'configured' ? '' : (data.vastai_api_key || '');
        document.getElementById('set-endpoint').value = data.serverless_endpoint_id || '';
    } catch(e) {}
}

async function saveSettings() {
    const updates = { ollama_host: document.getElementById('set-ollama').value };
    const openai = document.getElementById('set-openai').value.trim();
    const anthropic = document.getElementById('set-anthropic').value.trim();
    const vastai = document.getElementById('set-vastai').value.trim();
    const endpoint = document.getElementById('set-endpoint').value.trim();
    if (openai) updates.openai_api_key = openai;
    if (anthropic) updates.anthropic_api_key = anthropic;
    if (vastai) updates.vastai_api_key = vastai;
    if (endpoint) updates.serverless_endpoint_id = parseInt(endpoint) || 0;
    await fetch('/settings/update', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(updates)
    });
    alert('Settings saved.');
    checkStatus();
}

async function clearConversation() {
    if (!confirm('Clear all chat history?')) return;
    await fetch('/conversation/clear', { method: 'POST' });
    document.getElementById('chat-messages').innerHTML = '';
}

async function clearBadData() {
    await fetch('/learning/clear_bad', { method: 'POST' });
    alert('Bad data cleaned.');
}

async function resetLearning() {
    if (!confirm('This will delete ALL learned data. Continue?')) return;
    await fetch('/learning/reset', { method: 'POST' });
    alert('Learning data reset.');
}

async function exportConversation() {
    window.open('/export/conversation');
}

async function checkStatus() {
    try {
        const resp = await fetch('/status');
        const data = await resp.json();
        const ai = data.ai_backends || {};
        let available = false;
        let label = '';
        if (ai.ollama && ai.ollama.available) { available = true; label = 'Ollama: ' + ai.ollama.model; }
        else if (ai.openai && ai.openai.available) { available = true; label = 'OpenAI connected'; }
        else if (ai.anthropic && ai.anthropic.available) { available = true; label = 'Anthropic connected'; }
        else if (ai.serverless && ai.serverless.available) { available = true; label = 'Serverless connected'; }
        else { label = 'No AI backend'; }
        document.getElementById('dot-ai').className = 'status-dot ' + (available ? 'on' : 'off');
        document.getElementById('lbl-ai').textContent = label;
    } catch(e) {
        document.getElementById('lbl-ai').textContent = 'Offline';
    }
}

checkStatus();
setInterval(checkStatus, 30000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    config = Config.load()
    config.save()

    print(f"""
    Lab Dojo v{__version__} - Pathology Research Workstation
    Copyright (c) 2025-2026 JuiceVendor Labs Inc.

    Dashboard:  http://localhost:{config.port}
    Data dir:   {config.config_dir}
    """)

    app = create_app()
    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
