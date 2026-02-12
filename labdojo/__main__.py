"""Entry point for `python -m labdojo` and `labdojo` command."""

import sys
import importlib.util
from pathlib import Path

def main():
    """Main entry point for the labdojo command."""
    # Delegate to the monolith's main() function
    _monolith = Path(__file__).resolve().parent.parent / "labdojo.py"
    
    if _monolith.exists():
        spec = importlib.util.spec_from_file_location("_labdojo_mono", str(_monolith))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    else:
        print("Error: labdojo.py not found. Please run from the project root.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
