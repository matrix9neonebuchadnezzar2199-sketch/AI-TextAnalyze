"""Tests for MT sentence splitting."""

from backend.mt_engine import split_sentences


def test_split_japanese_sentences() -> None:
    text = "一行目です。二行目です！"
    parts = split_sentences(text)
    assert len(parts) == 2


def test_split_newlines() -> None:
    text = "Line one\nLine two"
    parts = split_sentences(text)
    assert len(parts) == 2


def test_empty_returns_empty() -> None:
    assert split_sentences("   ") == []
