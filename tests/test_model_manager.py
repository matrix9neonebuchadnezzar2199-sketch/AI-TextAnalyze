"""Tests for model directory scanning and exclusive loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.model_manager import (
    ActiveEngine,
    ModelManager,
    _has_ct2_model,
    _has_gliner_onnx,
    scan_models,
)


def test_has_gliner_onnx(tmp_path: Path) -> None:
    ner_dir = tmp_path / "ner-test"
    ner_dir.mkdir()
    (ner_dir / "model.onnx").write_bytes(b"fake")
    (ner_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert _has_gliner_onnx(ner_dir) is True


def test_has_ct2_model(tmp_path: Path) -> None:
    mt_dir = tmp_path / "mt-test"
    mt_dir.mkdir()
    (mt_dir / "model.bin").write_bytes(b"fake")
    (mt_dir / "config.json").write_text("{}", encoding="utf-8")
    (mt_dir / "shared_vocabulary.txt").write_text("vocab", encoding="utf-8")
    assert _has_ct2_model(mt_dir) is True


def test_scan_models_detects_both(tmp_path: Path) -> None:
    ner = tmp_path / "ner-gliner"
    ner.mkdir()
    (ner / "model.onnx").write_bytes(b"x")
    (ner / "tokenizer.json").write_text("{}", encoding="utf-8")

    mt = tmp_path / "mt-nllb"
    mt.mkdir()
    (mt / "model.bin").write_bytes(b"x")
    (mt / "config.json").write_text("{}", encoding="utf-8")
    (mt / "shared_vocabulary.txt").write_text("v", encoding="utf-8")

    found = scan_models(tmp_path)
    kinds = {m.kind for m in found}
    assert kinds == {"ner", "mt"}


def test_exclusive_load_unloads_other(tmp_path: Path) -> None:
    ner = tmp_path / "ner"
    ner.mkdir()
    (ner / "model.onnx").write_bytes(b"x")
    (ner / "tokenizer.json").write_text("{}", encoding="utf-8")
    (ner / "gliner_config.json").write_text('{"max_length":128}', encoding="utf-8")

    mt = tmp_path / "mt"
    mt.mkdir()
    (mt / "model.bin").write_bytes(b"x")
    (mt / "config.json").write_text("{}", encoding="utf-8")
    (mt / "shared_vocabulary.txt").write_text("v", encoding="utf-8")

    mgr = ModelManager(tmp_path)

    mock_ner = MagicMock()
    mock_mt = MagicMock()

    with patch("backend.model_manager.NerEngine", return_value=mock_ner):
        with patch("backend.model_manager.MtEngine", return_value=mock_mt):
            mgr.load_ner()
            assert mgr.active_engine == ActiveEngine.NER
            mgr.load_mt()
            assert mgr.active_engine == ActiveEngine.MT
            mock_ner.close.assert_called()


def test_status_dict_no_models(tmp_path: Path) -> None:
    mgr = ModelManager(tmp_path)
    status = mgr.status_dict()
    assert status["ner_available"] is False
    assert status["mt_available"] is False
    assert len(status["mt_models"]) == 2
    assert status["selected_mt_id"] == "nllb-600m"


def test_mt_model_selection(tmp_path: Path) -> None:
    mt600 = tmp_path / "mt-nllb-600m-ct2-int8"
    mt600.mkdir()
    (mt600 / "model.bin").write_bytes(b"x")
    (mt600 / "config.json").write_text("{}", encoding="utf-8")
    (mt600 / "shared_vocabulary.txt").write_text("v", encoding="utf-8")

    mt13 = tmp_path / "mt-nllb-1.3b-ct2-int8"
    mt13.mkdir()
    (mt13 / "model.bin").write_bytes(b"x")
    (mt13 / "config.json").write_text("{}", encoding="utf-8")
    (mt13 / "shared_vocabulary.txt").write_text("v", encoding="utf-8")

    mgr = ModelManager(tmp_path)
    assert mgr.mt_model() is not None
    assert mgr.mt_model().name == "mt-nllb-600m-ct2-int8"

    mgr.set_selected_mt("nllb-1.3b")
    assert mgr.selected_mt_id == "nllb-1.3b"
    assert mgr.mt_model().name == "mt-nllb-1.3b-ct2-int8"
