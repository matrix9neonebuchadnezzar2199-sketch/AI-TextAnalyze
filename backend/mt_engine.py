"""NLLB CTranslate2 machine translation (CPU only)."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

NLLB_LANG_CODES: dict[str, str] = {
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "ko": "kor_Hang",
}

# 文ごとに短い休止を入れ UI/WebView の応答性を確保
YIELD_SECONDS = 0.01

ProgressCallback = Callable[[int, int, str], None]
MAX_SENTENCE_CHARS = 500


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines while keeping paragraph structure."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    return parts if parts else [normalized]


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences (Latin + CJK punctuation)."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    parts = re.split(r"(?<=[。．！？!?\.])\s*", paragraph)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [paragraph]


def iter_translation_units(text: str) -> list[tuple[int, str]]:
    """Flatten paragraphs into translation units with paragraph index."""
    units: list[tuple[int, str]] = []
    paragraphs = split_paragraphs(text)
    for para_idx, paragraph in enumerate(paragraphs):
        for sentence in split_sentences(paragraph):
            if len(sentence) <= MAX_SENTENCE_CHARS:
                units.append((para_idx, sentence))
                continue
            # 極端に長い文はさらに分割
            for i in range(0, len(sentence), MAX_SENTENCE_CHARS):
                units.append((para_idx, sentence[i : i + MAX_SENTENCE_CHARS]))
    return units if units else [(0, text.strip())]


class MtEngine:
    """NLLB CT2 engine — load → translate → close."""

    def __init__(self, model_dir: Path, *, intra_threads: int = 2) -> None:
        self._model_dir = Path(model_dir)
        self._translator: Any = None
        self._tokenizer: Any = None
        self._load_model(intra_threads)

    def _load_model(self, intra_threads: int) -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        self._translator = ctranslate2.Translator(
            str(self._model_dir),
            device="cpu",
            inter_threads=1,
            intra_threads=intra_threads,
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_dir),
                fix_mistral_regex=True,
            )
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
        logger.info("MT CT2 loaded from %s", self._model_dir)

    def close(self) -> None:
        self._translator = None
        self._tokenizer = None

    def translate(
        self,
        text: str,
        src: str,
        tgt: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        """Translate text sentence by sentence, preserving paragraphs."""
        if self._translator is None or self._tokenizer is None:
            raise RuntimeError("MT engine is not loaded")

        src_code = NLLB_LANG_CODES.get(src)
        tgt_code = NLLB_LANG_CODES.get(tgt)
        if tgt_code is None:
            raise ValueError(f"Unsupported target language: {tgt}")
        if src != "auto" and src_code is None:
            raise ValueError(f"Unsupported source language: {src}")

        units = iter_translation_units(text)
        total = len(units)
        paragraph_count = max((idx for idx, _ in units), default=0) + 1
        translated_by_para: dict[int, list[str]] = {i: [] for i in range(paragraph_count)}

        for step, (para_idx, sentence) in enumerate(units, start=1):
            piece = self._translate_sentence(sentence, src_code, tgt_code)
            translated_by_para[para_idx].append(piece)
            if on_progress:
                on_progress(step, total, piece)
            if YIELD_SECONDS > 0:
                time.sleep(YIELD_SECONDS)

        paragraphs = [
            "".join(translated_by_para[i]).strip()
            for i in range(paragraph_count)
            if translated_by_para[i]
        ]
        return "\n\n".join(paragraphs)

    def _translate_sentence(
        self,
        sentence: str,
        src_code: str | None,
        tgt_code: str,
    ) -> str:
        """Translate a single sentence with NLLB tokenizer + CT2."""
        assert self._translator is not None
        assert self._tokenizer is not None

        tokenizer = self._tokenizer
        if src_code:
            tokenizer.src_lang = src_code

        source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence))
        results = self._translator.translate_batch(
            [source_tokens],
            target_prefix=[[tgt_code]],
            beam_size=1,
            max_batch_size=1,
            max_decoding_length=512,
        )
        if not results or not results[0].hypotheses:
            return ""

        tokens = list(results[0].hypotheses[0])
        if tokens and tokens[0] == tgt_code:
            tokens = tokens[1:]
        decoded = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(tokens),
            skip_special_tokens=True,
        ).strip()
        # 日本語向け: 文末に句点が無ければ付与
        if tgt_code == "jpn_Jpan" and decoded and decoded[-1] not in "。．！？":
            decoded += "。"
        return decoded
