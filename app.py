"""AI-TextAnalyze desktop entrypoint (pywebview)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

def _frontend_dir() -> Path:
    """開発時はリポジトリの frontend/、PyInstaller 凍結時は _MEIPASS 配下。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "frontend"
    return Path(__file__).resolve().parent / "frontend"


FRONTEND_DIR = _frontend_dir()
INDEX_HTML = FRONTEND_DIR / "index.html"


def main() -> None:
    """Launch WebView with Python bridge API."""
    if not INDEX_HTML.is_file():
        print(f"Frontend not found: {INDEX_HTML}", file=sys.stderr)
        sys.exit(1)

    import webview

    from backend.api import Api
    from backend.runtime_config import apply_cpu_thread_env

    threads = apply_cpu_thread_env()
    logging.getLogger(__name__).info("CPU inference threads capped at %s", threads)

    api = Api()
    window = webview.create_window(
        title="AI-TextAnalyze",
        url=str(INDEX_HTML),
        js_api=api,
        width=1440,
        height=900,
        min_size=(1024, 640),
    )
    api.set_window(window)
    webview.start(debug=False)


if __name__ == "__main__":
    main()
