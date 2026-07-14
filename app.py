"""AI-TextAnalyze desktop entrypoint (pywebview)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _app_root() -> Path:
    """開発時はリポジトリ根、凍結時は exe ディレクトリ。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _frontend_dir() -> Path:
    """開発時はリポジトリの frontend/、PyInstaller 凍結時は _MEIPASS 配下。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "frontend"
    return Path(__file__).resolve().parent / "frontend"


def _window_icon() -> str | None:
    """ウィンドウ／タスクバー用アイコン（あれば）。"""
    candidates = [
        _app_root() / "assets" / "AI-TextAnalyze.ico",
        Path(__file__).resolve().parent / "assets" / "AI-TextAnalyze.ico",
    ]
    if getattr(sys, "frozen", False):
        candidates.insert(0, Path(getattr(sys, "_MEIPASS")) / "assets" / "AI-TextAnalyze.ico")
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


FRONTEND_DIR = _frontend_dir()
INDEX_HTML = FRONTEND_DIR / "index.html"


def main() -> None:
    """Launch WebView quickly; heavy model load runs after UI is up."""
    if not INDEX_HTML.is_file():
        print(f"Frontend not found: {INDEX_HTML}", file=sys.stderr)
        sys.exit(1)

    # webview だけ先に入れてウィンドウを作る（torch/CT2 は warmup 時まで遅延）
    import webview

    from backend.runtime_config import apply_cpu_thread_env

    threads = apply_cpu_thread_env()
    logging.getLogger(__name__).info("CPU inference threads capped at %s", threads)

    from backend.api import Api

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
    icon = _window_icon()
    if icon:
        webview.start(debug=False, icon=icon)
    else:
        webview.start(debug=False)


if __name__ == "__main__":
    main()
