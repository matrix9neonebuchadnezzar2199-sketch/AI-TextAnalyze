"""pywebview bridge API exposed to the frontend."""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any

from backend.lang_detect import detect_language, lang_display_name
from backend.model_manager import ModelManager
from backend.pdf_reader import read_text_file

logger = logging.getLogger(__name__)


class Api:
    """JavaScript-callable API (``pywebview.api.*``)."""

    def __init__(self, model_manager: ModelManager | None = None) -> None:
        self._manager = model_manager or ModelManager()
        self._window: Any = None

    def set_window(self, window: Any) -> None:
        self._window = window

    def _ok(self, **payload: Any) -> dict[str, Any]:
        return {"ok": True, **payload}

    def _err(self, message: str) -> dict[str, Any]:
        logger.warning("API error: %s", message)
        return {"ok": False, "error": message}

    def _evaluate(self, js: str) -> None:
        if self._window is not None:
            try:
                self._window.evaluate_js(js)
            except Exception:
                logger.debug("evaluate_js failed", exc_info=True)

    def _set_status(self, message: str) -> None:
        payload = json.dumps(message, ensure_ascii=False)
        self._evaluate(f"status({payload})")

    def _show_progress(self, message: str, current: int = 0, total: int = 0) -> None:
        payload = json.dumps(message, ensure_ascii=False)
        self._evaluate(f"showProgress({payload}, {int(current)}, {int(total)})")

    def _hide_progress(self) -> None:
        self._evaluate("hideProgress()")

    def get_model_status(self) -> dict[str, Any]:
        try:
            status = self._manager.status_dict()
            status["mt_loaded"] = self._manager.mt_is_loaded()
            status["ner_loaded"] = self._manager.ner_is_loaded()
            return self._ok(**status)
        except Exception as exc:
            return self._err(str(exc))

    def warmup_mt(self) -> dict[str, Any]:
        """Preload translation model once per session (optional startup)."""
        if self._manager.mt_model() is None:
            return self._err("翻訳モデルが見つかりません")
        if self._manager.mt_is_loaded():
            return self._ok(warmed=True, cached=True)

        try:
            self._show_progress("翻訳モデル起動中…", 0, 0)
            self._set_status("翻訳モデルを起動中…")
            self._manager.load_mt()
            self._hide_progress()
            self._set_status("翻訳モデル準備完了")
            return self._ok(warmed=True, cached=False)
        except Exception as exc:
            logger.exception("warmup_mt failed")
            self._hide_progress()
            return self._err(str(exc))

    def pick_and_read_file(self) -> dict[str, Any]:
        if self._window is None:
            return self._err("Window not ready")

        try:
            import webview

            result = self._window.create_file_dialog(
                webview.OPEN_DIALOG,
                file_types=(
                    "Text and PDF (*.txt;*.pdf)",
                    "All files (*.*)",
                ),
            )
            if not result:
                return self._ok(cancelled=True, text="")
            path = result[0] if isinstance(result, (list, tuple)) else result
            text = read_text_file(path)
            lang = detect_language(text)
            return self._ok(cancelled=False, text=text, path=str(path), language=lang)
        except Exception as exc:
            logger.exception("pick_and_read_file failed")
            return self._err(str(exc))

    def detect_language(self, text: str) -> dict[str, Any]:
        try:
            code = detect_language(text or "")
            return self._ok(language=code, display=lang_display_name(code))
        except Exception as exc:
            return self._err(str(exc))

    def extract_keywords(self, text: str) -> dict[str, Any]:
        if not (text or "").strip():
            return self._err("本文が空です")
        if self._manager.ner_model() is None:
            return self._err("NER モデルが見つかりません。model/ に GLiNER ONNX を配置してください。")

        try:
            if not self._manager.ner_is_loaded():
                self._show_progress("NER モデルロード中…", 0, 0)
                self._set_status("NER モデルをロード中…")

            engine = self._manager.load_ner()

            def on_progress(current: int, total: int, _detail: str) -> None:
                self._show_progress("キーワード抽出中…", current, total)
                self._set_status(f"キーワード抽出中… {current}/{total}")

            keywords = engine.extract(text, on_progress=on_progress)
            # NER はセッション中保持（翻訳時のみ排他で入れ替え）
            self._hide_progress()
            self._set_status(f"抽出完了（{len(keywords)} 件）")
            return self._ok(keywords=keywords)
        except Exception as exc:
            logger.exception("extract_keywords failed")
            self._hide_progress()
            self._set_status("抽出エラー")
            return self._err(str(exc))

    def translate(self, text: str, src: str, tgt: str) -> dict[str, Any]:
        if not (text or "").strip():
            return self._err("本文が空です")
        if self._manager.mt_model() is None:
            return self._err("翻訳モデルが見つかりません。model/ に NLLB CT2 を配置してください。")

        try:
            resolved_src = src
            if src == "auto":
                resolved_src = detect_language(text)

            if not self._manager.mt_is_loaded():
                self._show_progress("翻訳モデルロード中…", 0, 0)
                self._set_status("翻訳モデルをロード中…")
            else:
                self._show_progress("翻訳中…", 0, 0)
                self._set_status("翻訳中…")

            engine = self._manager.load_mt()

            def on_progress(current: int, total: int, _piece: str) -> None:
                self._show_progress("翻訳中…", current, total)
                self._set_status(f"翻訳中… {current}/{total} 文")

            result = engine.translate(
                text,
                resolved_src,
                tgt,
                on_progress=on_progress,
            )
            self._hide_progress()
            self._set_status("翻訳完了")
            return self._ok(text=result, src=resolved_src, tgt=tgt)
        except Exception as exc:
            logger.exception("translate failed: %s", traceback.format_exc())
            self._manager.unload_mt()
            self._hide_progress()
            self._set_status("翻訳エラー")
            return self._err(str(exc))
