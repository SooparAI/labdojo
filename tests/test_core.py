"""Unit tests for Lab Dojo core components."""

import sys
import os
import tempfile
from pathlib import Path

# Add the repo root to the path so we can import the monolith
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


class TestConfig:
    """Tests for the Config dataclass."""

    def test_default_config(self):
        from labdojo import Config  # noqa: E402 - reimport intentional
        # Use a temp dir to avoid polluting the real config
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(config_dir=tmpdir)
            assert cfg.host == "0.0.0.0"
            assert cfg.port == 8080
            assert cfg.ollama_host == "http://localhost:11434"
            assert cfg.daily_budget == 5.0
            assert cfg.verbosity == "detailed"

    def test_config_save_load(self):
        from labdojo import Config
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(config_dir=tmpdir, port=9090, verbosity="concise")
            cfg.save()
            # Verify config file was created
            config_path = Path(tmpdir) / "config.json"
            assert config_path.exists()

    def test_env_var_override(self):
        from labdojo import Config
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["OPENAI_API_KEY"] = "sk-test-key-12345"
            try:
                cfg = Config(config_dir=tmpdir)
                assert cfg.openai_api_key == "sk-test-key-12345"
            finally:
                del os.environ["OPENAI_API_KEY"]


class TestKnowledgeBase:
    """Tests for the KnowledgeBase SQLite store."""

    def _make_kb(self, tmpdir):
        from labdojo import Config, KnowledgeBase
        cfg = Config(config_dir=tmpdir)
        return KnowledgeBase(cfg)

    def test_learn_and_recall(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = self._make_kb(tmpdir)
            kb.learn_qa("What is BRCA1?", "BRCA1 is a tumor suppressor gene.", source="pubmed")
            result = kb.recall_similar("What is BRCA1?")
            assert result is not None
            assert "tumor suppressor" in result["answer"]

    def test_recall_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = self._make_kb(tmpdir)
            kb.learn_qa("What is BRCA1?", "BRCA1 is a tumor suppressor gene.")
            result = kb.recall_similar("How do rockets work?")
            assert result is None

    def test_projects_crud(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = self._make_kb(tmpdir)
            pid = kb.create_project("Test Project", "A test", "cancer,brca1")
            assert pid is not None
            projects = kb.get_projects()
            assert len(projects) == 1
            assert projects[0]["name"] == "Test Project"
            kb.delete_project(pid)
            assert len(kb.get_projects()) == 0

    def test_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = self._make_kb(tmpdir)
            kb.add_conversation("default", "user", "Hello")
            kb.add_conversation("default", "assistant", "Hi there")
            convos = kb.get_conversations("default")
            assert len(convos) == 2
            kb.clear_conversations()
            assert len(kb.get_conversations("default")) == 0

    def test_usage_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = self._make_kb(tmpdir)
            kb.record_usage(local=1, api=3, cost=0.01)
            usage = kb.get_usage()
            assert usage["local_calls"] == 1
            assert usage["api_calls"] == 3

    def test_pipeline_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = self._make_kb(tmpdir)
            rid = kb.create_pipeline_run("literature_review", {"query": "BRCA1"})
            assert rid is not None
            kb.update_pipeline_run(rid, "completed", {"papers": 5})
            runs = kb.get_pipeline_runs()
            assert len(runs) == 1
            assert runs[0]["status"] == "completed"

    def test_clear_bad_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = self._make_kb(tmpdir)
            kb.learn_qa("q1", "short", confidence=0.1)  # bad: low quality + short
            kb.learn_qa("q2", "This is a proper answer with enough content.", confidence=0.8)
            kb.clear_bad_data()
            # The good answer should survive
            result = kb.recall_similar("q2")
            assert result is not None


class TestIntentClassification:
    """Tests for casual vs research intent classification."""

    def test_greeting(self):
        from labdojo import classify_intent
        intent, response = classify_intent("Hello")
        assert intent == "casual"
        assert "research assistant" in response.lower() or "Lab Dojo" in response

    def test_thanks(self):
        from labdojo import classify_intent
        intent, _ = classify_intent("Thanks!")
        assert intent == "casual"

    def test_research_query(self):
        from labdojo import classify_intent
        intent, response = classify_intent("What is the role of BRCA1 in DNA damage repair?")
        assert intent == "research"
        assert response == ""

    def test_short_message(self):
        from labdojo import classify_intent
        intent, _ = classify_intent("hi")
        assert intent == "casual"


class TestScienceAPIs:
    """Tests for the ScienceAPIs question classification."""

    def _make_apis(self, tmpdir):
        from labdojo import Config, KnowledgeBase, ScienceAPIs
        cfg = Config(config_dir=tmpdir)
        kb = KnowledgeBase(cfg)
        return ScienceAPIs(cfg, kb)

    def test_classify_protein_question(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            apis = self._make_apis(tmpdir)
            topics = apis.classify_question("What is the structure of BRCA1 protein?")
            assert "protein" in topics

    def test_classify_drug_question(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            apis = self._make_apis(tmpdir)
            topics = apis.classify_question("What are the side effects of imatinib drug?")
            assert "drug" in topics

    def test_classify_cancer_question(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            apis = self._make_apis(tmpdir)
            topics = apis.classify_question("TP53 mutations in colorectal cancer")
            assert "cancer" in topics

    def test_default_to_literature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            apis = self._make_apis(tmpdir)
            topics = apis.classify_question("Tell me about recent advances")
            assert "literature" in topics

    def test_api_routing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            apis = self._make_apis(tmpdir)
            api_ids = apis.get_apis_for_question("BRCA1 protein structure")
            assert "pubmed" in api_ids  # always included
            assert "uniprot" in api_ids or "pdb" in api_ids

    def test_api_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            apis = self._make_apis(tmpdir)
            status = apis.get_api_status()
            assert len(status) >= 18  # We have 19 APIs
            assert "pubmed" in status
