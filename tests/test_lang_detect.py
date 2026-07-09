"""Tests for language detection."""

from backend.lang_detect import detect_language, lang_display_name


def test_detect_japanese() -> None:
    assert detect_language("東京は日本の首都です。") == "ja"


def test_detect_english() -> None:
    assert detect_language("Hello world from London.") == "en"


def test_detect_korean() -> None:
    assert detect_language("안녕하세요 서울입니다") == "ko"


def test_detect_empty_defaults_ja() -> None:
    assert detect_language("") == "ja"


def test_display_name() -> None:
    assert "ja" in lang_display_name("ja")
