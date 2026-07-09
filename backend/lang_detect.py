"""Offline language detection for supported locales."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 仕様の5言語
SUPPORTED_LANGS = frozenset({"ja", "en", "zh", "ru", "ko"})

# langid → 仕様コードへの正規化
_LANGID_MAP = {
    "ja": "ja",
    "en": "en",
    "zh": "zh",
    "ru": "ru",
    "ko": "ko",
}


def detect_language(text: str) -> str:
    """Detect language code from text (offline).

    Uses ``langid`` when available; falls back to script heuristics.

    Args:
        text: Input text (non-empty recommended).

    Returns:
        One of ``ja``, ``en``, ``zh``, ``ru``, ``ko``. Defaults to ``en``.
    """
    sample = text.strip()
    if not sample:
        return "ja"

    try:
        import langid

        code, _score = langid.classify(sample[:4000])
        mapped = _LANGID_MAP.get(code)
        if mapped:
            return mapped
    except Exception:
        logger.debug("langid classify failed; using heuristic", exc_info=True)

    return _heuristic_detect(sample)


def _heuristic_detect(text: str) -> str:
    """Script-based fallback without external models."""
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u0400-\u04ff]", text):
        return "ru"
    if re.search(r"[\u4e00-\u9fff]", text) and not re.search(r"[\u3040-\u30ff]", text):
        return "zh"
    return "en"


def lang_display_name(code: str) -> str:
    """Human-readable label for UI."""
    names = {
        "ja": "日本語 (ja)",
        "en": "English (en)",
        "zh": "中文 (zh)",
        "ru": "Русский (ru)",
        "ko": "한국어 (ko)",
    }
    return names.get(code, code)
