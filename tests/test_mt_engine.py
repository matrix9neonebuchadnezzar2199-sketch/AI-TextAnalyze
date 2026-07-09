"""Tests for MT sentence splitting."""

from backend.mt_engine import iter_translation_units, split_paragraphs, split_sentences


def test_split_japanese_sentences() -> None:
    text = "一行目です。二行目です！"
    parts = split_sentences(text)
    assert len(parts) == 2


def test_split_paragraphs() -> None:
    text = "Para one.\n\nPara two."
    parts = split_paragraphs(text)
    assert len(parts) == 2


def test_iter_translation_units() -> None:
    text = "A. B.\n\nC."
    units = iter_translation_units(text)
    assert len(units) == 3
