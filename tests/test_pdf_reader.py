"""Tests for file reading."""

from pathlib import Path

import pytest

from backend.pdf_reader import read_text_file


def test_read_txt_utf8(tmp_path: Path) -> None:
    f = tmp_path / "sample.txt"
    f.write_text("テスト本文", encoding="utf-8")
    assert read_text_file(f) == "テスト本文"


def test_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "bad.doc"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        read_text_file(f)
