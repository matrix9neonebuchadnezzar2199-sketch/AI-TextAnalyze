"""AI-TextAnalyze desktop entrypoint (pywebview)."""

from __future__ import annotations

import atexit
import logging
import signal
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
HELP_HTML = FRONTEND_DIR / "help.html"


def main() -> None:
    """Launch WebView quickly; heavy model load runs after UI is up."""
    if not INDEX_HTML.is_file():
        print(f"Frontend not found: {INDEX_HTML}", file=sys.stderr)
        sys.exit(1)

    # webview だけ先に入れてウィンドウを作る（torch/CT2 は warmup 時まで遅延）
    import webview

    from backend.api import Api
    from backend.runtime_config import apply_cpu_thread_env

    threads = apply_cpu_thread_env()
    logger.info("CPU inference threads capped at %s", threads)

    api = Api(help_html=HELP_HTML)
    cleaned = {"done": False}

    def shutdown_once() -> None:
        """ウィンドウ閉鎖・シグナル・プロセス終了でモデルを解放する（多重呼び出し可）。"""
        if cleaned["done"]:
            return
        cleaned["done"] = True
        try:
            api.shutdown()
        except Exception:
            logger.exception("shutdown_once failed")

    atexit.register(shutdown_once)

    window = webview.create_window(
        title="AI-TextAnalyze",
        url=str(INDEX_HTML),
        js_api=api,
        width=1440,
        height=900,
        min_size=(1024, 640),
    )
    api.set_window(window)

    # 閉じる直前／直後に解放（closing はキャンセル可能なので False を返さない）
    window.events.closing += shutdown_once
    window.events.closed += shutdown_once

    def _on_signal(signum: int, _frame: object) -> None:
        logger.info("signal %s received — shutting down", signum)
        shutdown_once()
        try:
            window.destroy()
        except Exception:
            logger.debug("window.destroy failed after signal", exc_info=True)
        raise SystemExit(0)

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            # 非メインスレッド等では設定できない環境がある
            pass

    icon = _window_icon()
    try:
        if icon:
            webview.start(debug=False, icon=icon)
        else:
            webview.start(debug=False)
    finally:
        # start() 復帰時も必ず解放してプロセスを終了側へ戻す
        shutdown_once()


if __name__ == "__main__":
    main()
