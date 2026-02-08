"""UI assets for the Lab Dojo dashboard.

The React UI is built from labdojo-ui/ and served by FastAPI
when the dist/ directory exists. This package provides helpers
for locating the built assets.
"""

from pathlib import Path

UI_DIST_DIR = Path(__file__).resolve().parent.parent.parent / "labdojo-ui" / "dist"


def has_ui() -> bool:
    """Return True if the React UI has been built."""
    return UI_DIST_DIR.exists() and (UI_DIST_DIR / "index.html").exists()


def get_index_html() -> str:
    """Return the contents of the React UI index.html."""
    if not has_ui():
        raise FileNotFoundError("React UI not built. Run: cd labdojo-ui && pnpm build")
    return (UI_DIST_DIR / "index.html").read_text()


__all__ = ["UI_DIST_DIR", "has_ui", "get_index_html"]
