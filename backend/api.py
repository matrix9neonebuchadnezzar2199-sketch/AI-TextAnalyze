"""pywebview bridge API exposed to the frontend."""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any

from backend.lang_detect import detect_language, lang_display_name
from backend.model_manager import ModelManager
from backend.ner_engine import aggregate_keywords, chunk_text
from backend.pdf_reader import read_text_file

logger = logging.getLogger(__name__)


class Api:
    """JavaScript-callable API (``pywebview.api.*``)."""

    def __init__(self, model_manager: ModelManager | None = None) -> None:
        self._manager = model_manager or ModelManager()
        self._window: Any = None
        self._shutdown_done = False

    def set_window(self, window: Any) -> None:
        self._window = window

    def shutdown(self) -> dict[str, Any]:
        """Release loaded models. Safe to call multiple times on exit."""
        if self._shutdown_done:
            return self._ok(shutdown=True, already=True)
        try:
            self._manager.unload_all()
            self._shutdown_done = True
            logger.info("API shutdown: models unloaded")
            return self._ok(shutdown=True)
        except Exception as exc:
            logger.exception("shutdown failed")
            return self._err(str(exc))

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

    def _show_progress(
        self,
        message: str,
        current: int = 0,
        total: int = 0,
        detail: str = "",
    ) -> None:
        payload = json.dumps(message, ensure_ascii=False)
        detail_payload = json.dumps(detail, ensure_ascii=False)
        self._evaluate(
            f"showProgress({payload}, {int(current)}, {int(total)}, {detail_payload})"
        )

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

    def warmup_mt(self, model_id: str | None = None) -> dict[str, Any]:
        """Preload translation model once per session (optional startup)."""
        if model_id:
            try:
                self._manager.set_selected_mt(model_id)
            except ValueError as exc:
                return self._err(str(exc))

        if self._manager.mt_model() is None:
            spec_id = self._manager.selected_mt_id
            return self._err(
                "翻訳モデルが見つかりません。"
                f" python scripts/download-models.py --mt {spec_id}"
            )
        if self._manager.mt_matches_selection():
            return self._ok(warmed=True, cached=True, selected_mt_id=self._manager.selected_mt_id)

        try:
            self._show_progress("翻訳モデル起動中…", 0, 0, "モデルを読み込んでいます")
            self._set_status("翻訳モデルを起動中…")
            self._manager.load_mt()
            self._hide_progress()
            self._set_status("翻訳モデル準備完了")
            meta = self._manager.mt_model()
            return self._ok(
                warmed=True,
                cached=False,
                selected_mt_id=self._manager.selected_mt_id,
                mt_name=meta.name if meta else None,
            )
        except Exception as exc:
            logger.exception("warmup_mt failed")
            self._hide_progress()
            return self._err(str(exc))

    def select_mt_model(self, model_id: str) -> dict[str, Any]:
        """Switch MT variant and load it (shows progress overlay from JS)."""
        try:
            self._manager.set_selected_mt(model_id)
        except ValueError as exc:
            return self._err(str(exc))

        meta = self._manager.mt_model()
        if meta is None:
            from backend.mt_catalog import get_mt_spec

            spec = get_mt_spec(model_id)
            return self._err(
                f"{spec.label} が models/{spec.folder}/ にありません。"
                f" python scripts/download-models.py --mt {model_id} で取得してください。"
            )

        if self._manager.mt_matches_selection():
            return self._ok(
                selected_mt_id=model_id,
                mt_name=meta.name,
                cached=True,
            )

        try:
            self._show_progress("翻訳モデルロード中…", 0, 0, "モデルを切り替えています")
            self._set_status("翻訳モデルをロード中…")
            self._manager.load_mt()
            self._hide_progress()
            self._set_status("翻訳モデル準備完了")
            return self._ok(selected_mt_id=model_id, mt_name=meta.name, cached=False)
        except Exception as exc:
            logger.exception("select_mt_model failed")
            self._manager.unload_mt()
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

    def _ner_chunk_total(self, source_text: str, target_text: str) -> tuple[int, int]:
        """Return chunk counts for source and target texts."""
        from backend.mt_engine import collapse_pdf_spacing

        src_chunks = len(chunk_text(collapse_pdf_spacing(source_text))) if source_text.strip() else 0
        tgt_chunks = len(chunk_text(collapse_pdf_spacing(target_text))) if target_text.strip() else 0
        return src_chunks, tgt_chunks

    def extract_keywords(self, source_text: str, target_text: str = "") -> dict[str, Any]:
        source = (source_text or "").strip()
        target = (target_text or "").strip()
        if not source and not target:
            return self._err("本文が空です")
        if self._manager.ner_model() is None:
            return self._err("NER モデルが見つかりません。models/ に GLiNER ONNX を配置してください。")

        try:
            if not self._manager.ner_is_loaded():
                self._show_progress("NER モデルロード中…", 0, 0, "モデルを読み込んでいます")
                self._set_status("NER モデルをロード中…")

            engine = self._manager.load_ner()
            src_chunks, tgt_chunks = self._ner_chunk_total(source, target)
            total_chunks = src_chunks + tgt_chunks
            if total_chunks == 0:
                return self._err("抽出対象テキストがありません")

            raw: list[dict[str, Any]] = []

            def on_progress(current: int, total: int, detail: str) -> None:
                self._show_progress("キーワード抽出中…", current, total, detail)
                self._set_status(f"キーワード抽出中… {current}/{total}")

            if source:
                raw.extend(
                    engine.extract(
                        source,
                        on_progress=on_progress,
                        progress_offset=0,
                        progress_total=total_chunks,
                        progress_label="本文",
                    )
                )
            if target:
                raw.extend(
                    engine.extract(
                        target,
                        on_progress=on_progress,
                        progress_offset=src_chunks,
                        progress_total=total_chunks,
                        progress_label="翻訳",
                    )
                )

            keywords = aggregate_keywords(raw)
            self._hide_progress()
            scope = "本文+翻訳" if source and target else ("翻訳" if target and not source else "本文")
            self._set_status(f"抽出完了（{len(keywords)} 件・{scope}）")
            return self._ok(keywords=keywords, scope=scope)
        except Exception as exc:
            logger.exception("extract_keywords failed")
            self._hide_progress()
            self._set_status("抽出エラー")
            return self._err(str(exc))

    def translate(self, text: str, src: str, tgt: str) -> dict[str, Any]:
        if not (text or "").strip():
            return self._err("本文が空です")
        if self._manager.mt_model() is None:
            return self._err("翻訳モデルが見つかりません。models/ に NLLB CT2 を配置してください。")

        try:
            resolved_src = src
            if src == "auto":
                resolved_src = detect_language(text)

            if not self._manager.mt_is_loaded():
                self._show_progress("翻訳モデルロード中…", 0, 0, "モデルを読み込んでいます")
                self._set_status("翻訳モデルをロード中…")
            else:
                self._show_progress("翻訳中…", 0, 1, "準備中")

            engine = self._manager.load_mt()

            def on_progress(current: int, total: int, detail: str) -> None:
                label = detail[:80] if detail else ""
                self._show_progress(
                    "翻訳中…",
                    current,
                    total,
                    f"文 {current}/{total}" + (f" — {label}" if label else ""),
                )
                self._set_status(f"翻訳中… {current}/{total} 文")

            result = engine.translate(
                text,
                resolved_src,
                tgt,
                on_progress=on_progress,
            )
            self._hide_progress()
            self._set_status("翻訳完了")
            return self._ok(
                text=result.text,
                units=result.units,
                src=resolved_src,
                tgt=tgt,
            )
        except Exception as exc:
            logger.exception("translate failed: %s", traceback.format_exc())
            self._manager.unload_mt()
            self._hide_progress()
            self._set_status("翻訳エラー")
            return self._err(str(exc))
