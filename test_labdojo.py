"""
Comprehensive test suite for Lab Dojo v0.1.1
Covers: Config, KnowledgeBase, ScienceAPIs, InferenceRouter, endpoints, edge cases.
Run: python3.11 -m pytest test_labdojo.py -v
"""

import asyncio
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import labdojo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config_dir(tmp_path):
    config_dir = tmp_path / "labdojo_test"
    config_dir.mkdir()
    return str(config_dir)


@pytest.fixture
def config(tmp_config_dir):
    cfg = labdojo.Config()
    cfg.config_dir = tmp_config_dir
    cfg.db_path = os.path.join(tmp_config_dir, "knowledge.db")
    cfg.openai_api_key = ""
    cfg.anthropic_api_key = ""
    cfg.vastai_api_key = ""
    return cfg


@pytest.fixture
def kb(config):
    return labdojo.KnowledgeBase(config)


@pytest.fixture
def science_apis(config, kb):
    return labdojo.ScienceAPIs(config, kb)


# ---------------------------------------------------------------------------
# 1. Config Tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_values(self):
        cfg = labdojo.Config()
        assert cfg.port == 8080
        assert cfg.host == "0.0.0.0"
        assert cfg.ollama_host == "http://localhost:11434"

    def test_save_and_load_preserves_keys(self, tmp_config_dir):
        """API keys must survive save/load cycle via _save_secrets."""
        cfg = labdojo.Config()
        cfg.config_dir = tmp_config_dir
        cfg.openai_api_key = "sk-test-key-12345"
        cfg.anthropic_api_key = "sk-ant-test-67890"
        cfg.vastai_api_key = "vast-test-key"
        cfg.save()

        # Load fresh and apply secrets
        cfg2 = labdojo.Config()
        cfg2.config_dir = tmp_config_dir
        cfg2.openai_api_key = ""
        cfg2.anthropic_api_key = ""
        cfg2._load_secrets()

        assert cfg2.openai_api_key == "sk-test-key-12345"
        assert cfg2.anthropic_api_key == "sk-ant-test-67890"

    def test_config_dir_creation(self, tmp_path):
        cfg = labdojo.Config()
        new_dir = str(tmp_path / "new_labdojo")
        cfg.config_dir = new_dir
        os.makedirs(new_dir, exist_ok=True)
        assert os.path.isdir(new_dir)

    def test_save_creates_config_file(self, tmp_config_dir):
        cfg = labdojo.Config()
        cfg.config_dir = tmp_config_dir
        cfg.save()
        config_path = os.path.join(tmp_config_dir, "config.json")
        assert os.path.exists(config_path)

    def test_secrets_file_permissions(self, tmp_config_dir):
        cfg = labdojo.Config()
        cfg.config_dir = tmp_config_dir
        cfg.openai_api_key = "sk-test"
        cfg._save_secrets()
        secrets_path = os.path.join(tmp_config_dir, "secrets.json")
        assert os.path.exists(secrets_path)
        mode = oct(os.stat(secrets_path).st_mode)[-3:]
        assert mode == "600"


# ---------------------------------------------------------------------------
# 2. Intent Classification Tests
# ---------------------------------------------------------------------------

class TestIntentClassification:
    @pytest.mark.parametrize("msg,expected_intent", [
        ("hello", "casual"),
        ("hi", "casual"),
        ("hi there!", "casual"),
        ("hey", "casual"),
        ("thanks", "casual"),
        ("thank you so much", "casual"),
        ("goodbye", "casual"),
        ("bye", "casual"),
        ("good morning", "casual"),
        ("good night", "casual"),
        ("how are you", "casual"),
        ("what's up", "casual"),
        ("yo", "casual"),
        ("sup", "casual"),
        ("ok", "casual"),
        ("great", "casual"),
        ("test", "casual"),
        ("ping", "casual"),
        ("who are you", "casual"),
        ("what can you do", "casual"),
        ("help", "casual"),
    ])
    def test_casual_messages(self, msg, expected_intent):
        intent, response = labdojo.classify_intent(msg)
        assert intent == expected_intent, f"'{msg}' classified as '{intent}', expected '{expected_intent}'"
        assert response  # Should have a non-empty response

    @pytest.mark.parametrize("msg", [
        "What is the role of BRCA1 in DNA damage repair?",
        "Compare PD-1 and PD-L1 inhibitors",
        "TP53 mutations in colorectal cancer",
        "How does O-GlcNAcylation affect NF-kB signaling?",
        "What are the latest clinical trials for pembrolizumab?",
        "Explain the PI3K/AKT pathway in breast cancer",
        "Find papers on CRISPR gene editing in pathology",
    ])
    def test_research_messages(self, msg):
        intent, response = labdojo.classify_intent(msg)
        assert intent == "research", f"'{msg}' classified as '{intent}', expected 'research'"

    def test_empty_message(self):
        intent, response = labdojo.classify_intent("")
        assert intent == "casual"

    def test_single_char(self):
        intent, _ = labdojo.classify_intent("a")
        assert intent == "casual"

    def test_single_word_research(self):
        intent, _ = labdojo.classify_intent("BRCA1")
        assert intent == "research"

    def test_unicode_input(self):
        intent, _ = labdojo.classify_intent("What about p53 in cancer research?")
        assert intent == "research"

    def test_casual_response_categories(self):
        _, resp = labdojo.classify_intent("hello")
        assert "Lab Dojo" in resp  # greeting

        _, resp = labdojo.classify_intent("thanks")
        assert "welcome" in resp.lower()  # thanks

        _, resp = labdojo.classify_intent("bye")
        assert "care" in resp.lower() or "saved" in resp.lower()  # farewell


# ---------------------------------------------------------------------------
# 3. KnowledgeBase Tests
# ---------------------------------------------------------------------------

class TestKnowledgeBase:
    def test_tables_created(self, kb):
        conn = sqlite3.connect(kb.db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        expected = {"conversations", "learned_qa", "projects", "decisions",
                    "verified_citations", "pipeline_runs", "monitored_topics",
                    "usage_stats"}
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_conversation_add_and_get(self, kb):
        kb.add_conversation("test_proj", "user", "Hello world")
        import time; time.sleep(0.01)
        kb.add_conversation("test_proj", "assistant", "Hi there")
        history = kb.get_conversations("test_proj", limit=10)
        assert len(history) == 2
        # Verify both messages are present regardless of order
        roles = [h["role"] for h in history]
        assert "user" in roles
        assert "assistant" in roles
        contents = [h["content"] for h in history]
        assert "Hello world" in contents
        assert "Hi there" in contents

    def test_conversation_limit(self, kb):
        for i in range(20):
            kb.add_conversation("proj", "user", f"Message {i}")
            import time; time.sleep(0.001)
        history = kb.get_conversations("proj", limit=5)
        assert len(history) == 5

    def test_learn_qa_and_recall(self, kb):
        kb.learn_qa("What is BRCA1?", "BRCA1 is a tumor suppressor gene.", "ollama")
        result = kb.recall_similar("What is BRCA1?")
        assert result is not None
        assert "BRCA1" in result["answer"]

    def test_recall_returns_dict(self, kb):
        kb.learn_qa("What is p53?", "p53 is a tumor suppressor.", "ollama")
        result = kb.recall_similar("What is p53?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "question" in result
        assert "source" in result

    def test_recall_no_match(self, kb):
        result = kb.recall_similar("completely unrelated query about cooking")
        assert result is None

    def test_recall_threshold(self, kb):
        kb.learn_qa("BRCA1 in breast cancer", "BRCA1 is important.", "ollama")
        # Very different query should not match at 0.85 threshold
        result = kb.recall_similar("What is the weather today?")
        assert result is None

    def test_project_crud(self, kb):
        pid = kb.create_project("Test Project", "A test", "BRCA1, TP53")
        assert pid

        projects = kb.get_projects()
        assert len(projects) >= 1
        assert any(p["name"] == "Test Project" for p in projects)

        proj = kb.get_project(pid)
        assert proj is not None
        assert proj["name"] == "Test Project"

        kb.delete_project(pid)
        proj = kb.get_project(pid)
        assert proj is None

    def test_decision_add_and_get(self, kb):
        pid = kb.create_project("Decision Test", "", "")
        kb.add_decision(pid, "Use CRISPR", "Most efficient", "PMID:12345")
        decisions = kb.get_decisions(pid)
        assert len(decisions) == 1
        assert decisions[0]["decision"] == "Use CRISPR"

    def test_citation_verification(self, kb):
        kb.verify_citation("12345", "Test Paper", "Author A", "Journal X", "2024", "10.1234/test", "Abstract text")
        conn = sqlite3.connect(kb.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM verified_citations WHERE pmid='12345'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1

    def test_citation_get(self, kb):
        kb.verify_citation("99999", "Paper X", "Auth B", "J Y", "2025", "10.9999/x")
        result = kb.get_verified_citation("99999")
        assert result is not None
        assert result["title"] == "Paper X"

    def test_pipeline_run_lifecycle(self, kb):
        run_id = kb.create_pipeline_run("literature_review", {"query": "BRCA1"})
        assert run_id

        kb.update_pipeline_run(run_id, "completed", {"papers": 5})
        runs = kb.get_pipeline_runs()
        assert len(runs) >= 1
        completed = [r for r in runs if r["id"] == run_id]
        assert completed[0]["status"] == "completed"

    def test_monitored_topics(self, kb):
        kb.add_monitored_topic("BRCA1 mutations")
        topics = kb.get_monitored_topics()
        assert len(topics) >= 1
        assert any(t["topic"] == "BRCA1 mutations" for t in topics)

        tid = topics[0]["id"]
        kb.remove_monitored_topic(tid)
        topics = kb.get_monitored_topics()
        assert not any(t["topic"] == "BRCA1 mutations" for t in topics)

    def test_usage_stats(self, kb):
        kb.record_usage(local=1, api=2)
        kb.record_usage(serverless=1)
        stats = kb.get_usage()
        assert stats["local_calls"] >= 1
        assert stats["api_calls"] >= 2
        assert stats["serverless_calls"] >= 1

    def test_usage_stats_accumulate(self, kb):
        kb.record_usage(local=5)
        kb.record_usage(local=3)
        stats = kb.get_usage()
        assert stats["local_calls"] >= 8

    def test_clear_conversations(self, kb):
        kb.add_conversation("proj", "user", "test")
        kb.clear_conversations()
        history = kb.get_conversations("proj", limit=100)
        assert len(history) == 0

    def test_clear_bad_data(self, kb):
        kb.clear_bad_data()  # Should not raise

    def test_reset_learning(self, kb):
        kb.learn_qa("test q", "test a", "ollama")
        kb.reset_learning()
        result = kb.recall_similar("test q")
        assert result is None


# ---------------------------------------------------------------------------
# 4. ScienceAPIs Tests
# ---------------------------------------------------------------------------

class TestScienceAPIs:
    def test_classify_question_literature(self, science_apis):
        topics = science_apis.classify_question("What are the latest papers on BRCA1?")
        assert isinstance(topics, list)
        assert len(topics) > 0

    def test_classify_question_protein(self, science_apis):
        topics = science_apis.classify_question("What is the structure of p53 protein?")
        assert isinstance(topics, list)
        assert len(topics) > 0

    def test_classify_question_drug(self, science_apis):
        topics = science_apis.classify_question("What are the side effects of imatinib drug?")
        assert isinstance(topics, list)

    def test_classify_question_fallback(self, science_apis):
        topics = science_apis.classify_question("xyzzy foobar baz")
        assert isinstance(topics, list)
        assert "literature" in topics  # Default fallback

    def test_get_apis_for_question(self, science_apis):
        apis = science_apis.get_apis_for_question("BRCA1 protein structure")
        assert isinstance(apis, list)
        assert len(apis) <= 8
        assert "pubmed" in apis  # Always included

    def test_api_status(self, science_apis):
        status = science_apis.get_api_status()
        assert isinstance(status, dict)
        assert len(status) >= 19

    def test_api_status_structure(self, science_apis):
        status = science_apis.get_api_status()
        for api_id, info in status.items():
            assert "name" in info
            assert "description" in info

    def test_extract_search_terms(self, science_apis):
        terms = science_apis._extract_search_terms(
            "What is the role of BRCA1 in DNA damage repair?")
        assert "BRCA1" in terms
        assert len(terms) > 0

    def test_extract_search_terms_short_query(self, science_apis):
        terms = science_apis._extract_search_terms("p53")
        assert "p53" in terms

    def test_cache_operations(self, science_apis):
        key = science_apis._cache_key("pubmed", "BRCA1")
        assert isinstance(key, str)

        science_apis._cache_set(key, "cached data")
        result = science_apis._cache_get(key)
        assert result == "cached data"

    def test_cache_expiry(self, science_apis):
        key = "test_expiry"
        science_apis._cache[key] = {"data": "old", "ts": time.time() - 99999}
        result = science_apis._cache_get(key)
        assert result is None

    def test_cache_eviction(self, science_apis):
        for i in range(510):
            science_apis._cache_set(f"key_{i}", f"data_{i}")
        assert len(science_apis._cache) <= 500

    def test_parse_pubmed_xml(self, science_apis):
        xml = """<PubmedArticle>
            <PMID>12345678</PMID>
            <ArticleTitle>Test Paper Title</ArticleTitle>
            <Title>Test Journal</Title>
            <Year>2024</Year>
            <AbstractText>This is a test abstract.</AbstractText>
            <LastName>Smith</LastName>
            <LastName>Jones</LastName>
            <ArticleId IdType="doi">10.1234/test</ArticleId>
        </PubmedArticle>"""
        result = science_apis._parse_pubmed_xml(xml)
        assert "PMID:12345678" in result
        assert "Test Paper Title" in result

    def test_parse_pubmed_xml_empty(self, science_apis):
        result = science_apis._parse_pubmed_xml("<root></root>")
        assert result == ""


# ---------------------------------------------------------------------------
# 5. Response Cleaning Tests
# ---------------------------------------------------------------------------

class TestResponseCleaning:
    def test_clean_confidence_labels(self):
        text = "[HIGH CONFIDENCE] BRCA1 is a tumor suppressor."
        result = labdojo._clean_response(text)
        assert "[HIGH CONFIDENCE]" not in result
        assert "BRCA1" in result

    def test_clean_ai_disclaimers(self):
        text = "As an AI, I cannot verify this.\nBRCA1 is important."
        result = labdojo._clean_response(text)
        assert "As an AI" not in result
        assert "BRCA1" in result

    def test_clean_consult_expert(self):
        text = "BRCA1 is a gene.\nI recommend consulting a specialist."
        result = labdojo._clean_response(text)
        assert "consulting" not in result

    def test_clean_disclaimer_block(self):
        text = "Data shows X.\n**Disclaimer**: This is not medical advice."
        result = labdojo._clean_response(text)
        assert "Disclaimer" not in result

    def test_clean_preserves_content(self):
        text = "BRCA1 (PMID:12345) interacts with RAD51 at Ser1423."
        result = labdojo._clean_response(text)
        assert result == text

    def test_clean_empty_string(self):
        assert labdojo._clean_response("") == ""

    def test_clean_none(self):
        assert labdojo._clean_response(None) is None

    def test_clean_multiple_newlines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = labdojo._clean_response(text)
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# 6. Export Format Tests
# ---------------------------------------------------------------------------

class TestExportFormats:
    @pytest.fixture
    def sample_papers(self):
        return [
            {
                "pmid": "12345",
                "title": "Test Paper One",
                "authors": ["Smith", "Jones", "Brown"],
                "journal": "Nature",
                "pub_date": "2024",
                "doi_url": "https://doi.org/10.1234/test1",
                "abstract": "Abstract one text."
            },
            {
                "pmid": "67890",
                "title": "Test Paper Two",
                "authors": ["Lee", "Kim"],
                "journal": "Science",
                "pub_date": "2023",
                "doi_url": "",
                "abstract": ""
            }
        ]

    def test_bibtex_format(self, sample_papers):
        result = labdojo._format_bibtex(sample_papers)
        assert "@article{pmid12345" in result
        assert "Smith and Jones and Brown" in result
        assert "Nature" in result

    def test_ris_format(self, sample_papers):
        result = labdojo._format_ris(sample_papers)
        assert "TY  - JOUR" in result
        assert "AU  - Smith" in result
        assert "PMID:12345" in result

    def test_markdown_format(self, sample_papers):
        result = labdojo._format_markdown(sample_papers)
        assert "# Literature Export" in result
        assert "Test Paper One" in result
        assert "PMID" in result

    def test_bibtex_empty_list(self):
        assert labdojo._format_bibtex([]) == ""

    def test_ris_empty_list(self):
        assert labdojo._format_ris([]) == ""

    def test_bibtex_many_authors(self):
        paper = [{
            "pmid": "111",
            "title": "Many Authors",
            "authors": ["A", "B", "C", "D", "E", "F", "G"],
            "journal": "J",
            "pub_date": "2024",
            "doi_url": "",
            "abstract": ""
        }]
        result = labdojo._format_bibtex(paper)
        assert "A and B and C and D and E" in result
        assert "F" not in result  # Only first 5


# ---------------------------------------------------------------------------
# 7. XML Helper Tests
# ---------------------------------------------------------------------------

class TestXmlHelper:
    def test_xml_tag_found(self):
        assert labdojo._xml_tag("<root><Name>Test</Name></root>", "Name") == "Test"

    def test_xml_tag_not_found(self):
        assert labdojo._xml_tag("<root></root>", "Missing") == ""

    def test_xml_tag_with_attributes(self):
        assert labdojo._xml_tag('<root><Tag attr="x">Value</Tag></root>', "Tag") == "Value"

    def test_xml_tag_multiline(self):
        xml = "<root><Abstract>\n  Multi\n  Line\n</Abstract></root>"
        result = labdojo._xml_tag(xml, "Abstract")
        assert "Multi" in result
        assert "Line" in result

    def test_xml_tag_empty_content(self):
        assert labdojo._xml_tag("<root><Tag></Tag></root>", "Tag") == ""


# ---------------------------------------------------------------------------
# 8. InferenceRouter Tests
# ---------------------------------------------------------------------------

class TestInferenceRouter:
    def test_router_creation(self, config):
        router = labdojo.InferenceRouter(config)
        assert router.ollama is not None
        assert router.serverless is not None
        assert router.openai is not None
        assert router.anthropic is not None

    @pytest.mark.asyncio
    async def test_router_fallback(self, config):
        router = labdojo.InferenceRouter(config)
        result, backend = await router.chat("test prompt")
        assert backend == "none"
        assert result == ""

    @pytest.mark.asyncio
    async def test_router_with_mock_ollama(self, config):
        router = labdojo.InferenceRouter(config)

        async def mock_chat(prompt, system="", temperature=0.7, messages=None):
            return "Mocked response about BRCA1"

        router.ollama.available = True
        router.ollama.chat = mock_chat
        result, backend = await router.chat("What is BRCA1?")
        assert result == "Mocked response about BRCA1"
        assert "ollama" in backend.lower() or "local" in backend.lower()

    @pytest.mark.asyncio
    async def test_router_deterministic_temp(self, config):
        router = labdojo.InferenceRouter(config)
        captured_temp = {}

        async def mock_chat(prompt, system="", temperature=0.7, messages=None):
            captured_temp["temp"] = temperature
            return "test response"

        router.ollama.available = True
        router.ollama.chat = mock_chat
        await router.chat("test", deterministic=True)
        assert captured_temp.get("temp") == 0.0


# ---------------------------------------------------------------------------
# 9. FastAPI Endpoint Tests
# ---------------------------------------------------------------------------

class TestEndpoints:
    @pytest.fixture
    def app(self, tmp_config_dir):
        with patch.object(labdojo.Config, 'load') as mock_load:
            cfg = labdojo.Config()
            cfg.config_dir = tmp_config_dir
            cfg.db_path = os.path.join(tmp_config_dir, "knowledge.db")
            cfg.openai_api_key = ""
            cfg.anthropic_api_key = ""
            cfg.vastai_api_key = ""
            mock_load.return_value = cfg
            app = labdojo.create_app()
        return app

    @pytest.mark.asyncio
    async def test_status_endpoint(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/status")
            assert resp.status_code == 200
            data = resp.json()
            assert "version" in data
            assert "ai_backends" in data

    @pytest.mark.asyncio
    async def test_chat_empty_message(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/chat", json={"message": ""})
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_casual(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/chat", json={"message": "hello"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["source"] == "system"
            assert data["response"]

    @pytest.mark.asyncio
    async def test_projects_crud(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/projects", json={"name": "Test", "description": "A test project"})
            assert resp.status_code == 200
            pid = resp.json()["id"]

            resp = await client.get("/projects")
            assert resp.status_code == 200
            assert len(resp.json()["projects"]) >= 1

            resp = await client.get(f"/projects/{pid}")
            assert resp.status_code == 200

            resp = await client.delete(f"/projects/{pid}")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_project_not_found(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/projects/nonexistent-id")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_apis_endpoint(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/apis")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["apis"]) >= 19

    @pytest.mark.asyncio
    async def test_settings_get(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/settings")
            assert resp.status_code == 200
            data = resp.json()
            assert "ollama_host" in data

    @pytest.mark.asyncio
    async def test_settings_masks_keys(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/settings")
            data = resp.json()
            # API keys should be masked or empty, not raw
            assert data.get("openai_api_key", "") in ("", "configured")

    @pytest.mark.asyncio
    async def test_monitor_topics_crud(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/monitor/topics", json={"topic": "BRCA1"})
            assert resp.status_code == 200

            resp = await client.get("/monitor/topics")
            assert resp.status_code == 200
            topics = resp.json()["topics"]
            assert len(topics) >= 1

            tid = topics[0]["id"]
            resp = await client.delete(f"/monitor/topics/{tid}")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_pipeline_invalid_type(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/pipeline/run", json={
                "pipeline_type": "invalid_type",
                "query": "test"
            })
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_learning_stats(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/learning/stats")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_conversation_clear(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/conversation/clear")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_dashboard_html(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "Lab Dojo" in resp.text

    @pytest.mark.asyncio
    async def test_export_conversation(self, app):
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/export/conversation")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 10. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_special_characters_in_search(self, science_apis):
        terms = science_apis._extract_search_terms("What about p53 & BRCA1/2?")
        assert len(terms) > 0

    def test_very_long_message(self, kb):
        long_msg = "A" * 10000
        kb.add_conversation("proj", "user", long_msg)
        history = kb.get_conversations("proj", limit=1)
        assert len(history) == 1
        # Content should be truncated to 5000
        assert len(history[0]["content"]) <= 5000

    def test_sql_injection_attempt(self, kb):
        kb.add_conversation("proj", "user", "'; DROP TABLE conversations; --")
        history = kb.get_conversations("proj", limit=10)
        assert len(history) >= 1

    def test_unicode_in_conversation(self, kb):
        kb.add_conversation("proj", "user", "What about p53 in 日本語 研究?")
        history = kb.get_conversations("proj", limit=1)
        assert "日本語" in history[0]["content"]

    def test_empty_project_name(self, kb):
        pid = kb.create_project("", "", "")
        assert pid

    def test_duplicate_monitored_topic(self, kb):
        kb.add_monitored_topic("BRCA1")
        kb.add_monitored_topic("BRCA1")
        topics = kb.get_monitored_topics()
        assert len(topics) >= 2

    def test_concurrent_db_access(self, kb):
        import threading
        errors = []

        def write_data(i):
            try:
                kb.add_conversation(f"proj_{i}", "user", f"Message {i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_data, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Concurrent access errors: {errors}"

    def test_learn_qa_with_long_content(self, kb):
        long_q = "Q" * 1000
        long_a = "A" * 10000
        kb.learn_qa(long_q, long_a, "test")
        # Should truncate and not crash


# ---------------------------------------------------------------------------
# 11. System Prompt Tests
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_system_prompt_exists(self):
        assert labdojo._SYSTEM_PROMPT
        assert len(labdojo._SYSTEM_PROMPT) > 100

    def test_system_prompt_no_specific_university(self):
        prompt = labdojo._SYSTEM_PROMPT.lower()
        assert "case western" not in prompt

    def test_system_prompt_for_any_lab(self):
        prompt = labdojo._SYSTEM_PROMPT.lower()
        assert "any university" in prompt or "any research" in prompt or "worldwide" in prompt

    def test_system_prompt_conversational_rule(self):
        prompt = labdojo._SYSTEM_PROMPT.lower()
        assert "conversational" in prompt or "greeting" in prompt

    def test_verbosity_levels(self):
        assert "concise" in labdojo._VERBOSITY
        assert "detailed" in labdojo._VERBOSITY
        assert "comprehensive" in labdojo._VERBOSITY

    def test_verbosity_content(self):
        for level, text in labdojo._VERBOSITY.items():
            assert len(text) > 10


# ---------------------------------------------------------------------------
# 12. Version and Metadata Tests
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_version_format(self):
        assert re.match(r"\d+\.\d+\.\d+", labdojo.__version__)

    def test_api_catalog_count(self):
        assert len(labdojo._API_CATALOG) >= 19

    def test_api_catalog_structure(self):
        for api_id, info in labdojo._API_CATALOG.items():
            assert "name" in info, f"API {api_id} missing 'name'"
            assert "desc" in info, f"API {api_id} missing 'desc'"
            assert "free" in info, f"API {api_id} missing 'free'"

    def test_topic_keywords_coverage(self):
        assert len(labdojo._TOPIC_KEYWORDS) >= 5

    def test_route_map_references_valid_apis(self):
        for topic, api_ids in labdojo._ROUTE_MAP.items():
            for api_id in api_ids:
                assert api_id in labdojo._API_CATALOG, \
                    f"Route map references unknown API: {api_id}"

    def test_all_apis_in_route_map(self):
        """Every API in the catalog should be reachable from at least one route."""
        all_routed = set()
        for api_ids in labdojo._ROUTE_MAP.values():
            all_routed.update(api_ids)
        for api_id in labdojo._API_CATALOG:
            assert api_id in all_routed, f"API {api_id} not reachable from any route"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
