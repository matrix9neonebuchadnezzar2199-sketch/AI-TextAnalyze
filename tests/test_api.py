"""Tests for API bridge."""

from pathlib import Path
from unittest.mock import MagicMock

from backend.api import Api
from backend.model_manager import ModelManager


def test_get_model_status_empty(tmp_path: Path) -> None:
    api = Api(ModelManager(tmp_path))
    res = api.get_model_status()
    assert res["ok"] is True
    assert res["ner_available"] is False


def test_extract_keywords_empty_text(tmp_path: Path) -> None:
    api = Api(ModelManager(tmp_path))
    res = api.extract_keywords("  ", "")
    assert res["ok"] is False
    assert "空" in res["error"]


def test_extract_keywords_target_only_rejected(tmp_path: Path) -> None:
    api = Api(ModelManager(tmp_path))
    res = api.extract_keywords("", "  ")
    assert res["ok"] is False


def test_detect_language_api() -> None:
    api = Api()
    res = api.detect_language("Hello world")
    assert res["ok"] is True
    assert res["language"] == "en"


def test_translate_no_model(tmp_path: Path) -> None:
    api = Api(ModelManager(tmp_path))
    res = api.translate("test", "en", "ja")
    assert res["ok"] is False


def test_pick_and_read_file_no_window() -> None:
    api = Api()
    res = api.pick_and_read_file()
    assert res["ok"] is False


def test_set_window() -> None:
    api = Api()
    win = MagicMock()
    api.set_window(win)
    api._set_status("test")
    win.evaluate_js.assert_called_once()


def test_shutdown_unloads_once(tmp_path: Path) -> None:
    mgr = ModelManager(tmp_path)
    mgr.unload_all = MagicMock()  # type: ignore[method-assign]
    api = Api(mgr)
    assert api.shutdown()["ok"] is True
    assert api.shutdown().get("already") is True
    mgr.unload_all.assert_called_once()
