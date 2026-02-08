"""Core configuration, knowledge base, and utility re-exports."""

# Re-export from the monolith so existing code keeps working
import sys
import importlib.util
from pathlib import Path

# Import from the monolith labdojo.py at the repo root
_monolith = Path(__file__).resolve().parent.parent.parent / "labdojo.py"

if _monolith.exists():
    spec = importlib.util.spec_from_file_location("_labdojo_mono", str(_monolith))
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)

    Config = _mod.Config
    KnowledgeBase = _mod.KnowledgeBase
    setup_logging = _mod.setup_logging
    classify_intent = _mod.classify_intent
else:
    raise ImportError(
        "labdojo.py monolith not found. The package currently re-exports from "
        "the single-file distribution. Place labdojo.py in the project root."
    )

__all__ = ["Config", "KnowledgeBase", "setup_logging", "classify_intent"]
