"""AI client and router re-exports."""

import importlib.util
from pathlib import Path

_monolith = Path(__file__).resolve().parent.parent.parent / "labdojo.py"

if _monolith.exists():
    spec = importlib.util.spec_from_file_location("_labdojo_mono", str(_monolith))
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)

    OllamaClient = _mod.OllamaClient
    ServerlessClient = _mod.ServerlessClient
    OpenAIClient = _mod.OpenAIClient
    AnthropicClient = _mod.AnthropicClient
    AIRouter = _mod.AIRouter
else:
    raise ImportError("labdojo.py monolith not found.")

__all__ = [
    "OllamaClient",
    "ServerlessClient",
    "OpenAIClient",
    "AnthropicClient",
    "AIRouter",
]
