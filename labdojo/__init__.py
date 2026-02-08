"""
Lab Dojo Pathology
AI-powered research workstation for pathology laboratories.

Copyright (c) 2025-2026 JuiceVendor Labs Inc.
Released under the MIT License.

Usage:
    from labdojo import Config, KnowledgeBase, ScienceAPIs, AIRouter
    from labdojo import classify_intent
"""

__version__ = "0.1.2"
__author__ = "JuiceVendor Labs Inc."
__license__ = "MIT"

import importlib.util
from pathlib import Path

# Re-export core classes from the monolith (labdojo.py)
_monolith = Path(__file__).resolve().parent.parent / "labdojo.py"

if _monolith.exists():
    _spec = importlib.util.spec_from_file_location("_labdojo_mono", str(_monolith))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    Config = _mod.Config
    KnowledgeBase = _mod.KnowledgeBase
    ScienceAPIs = _mod.ScienceAPIs
    OllamaClient = _mod.OllamaClient
    ServerlessClient = _mod.ServerlessClient
    OpenAIClient = _mod.OpenAIClient
    AnthropicClient = _mod.AnthropicClient
    InferenceRouter = _mod.InferenceRouter
    AIRouter = InferenceRouter  # alias
    setup_logging = _mod.setup_logging
    classify_intent = _mod.classify_intent
    create_app = _mod.create_app
else:
    raise ImportError(
        "labdojo.py monolith not found. Place labdojo.py in the project root."
    )

__all__ = [
    "Config",
    "KnowledgeBase",
    "ScienceAPIs",
    "OllamaClient",
    "ServerlessClient",
    "OpenAIClient",
    "AnthropicClient",
    "InferenceRouter",
    "AIRouter",
    "setup_logging",
    "classify_intent",
    "create_app",
]
