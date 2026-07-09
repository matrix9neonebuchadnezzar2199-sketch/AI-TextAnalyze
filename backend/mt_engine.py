"""NLLB CTranslate2 machine translation (CPU only)."""

from __future__ import annotations

import logging
import re
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

ProgressCallback = Callable[[int, int, str], None]

# CT2 はバッチ推論の方が逐次より大幅に速い
TRANSLATE_BATCH_SIZE = 16
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
        """Translate text in batches, preserving paragraphs."""
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

        done = 0
        for start in range(0, total, TRANSLATE_BATCH_SIZE):
            batch = units[start : start + TRANSLATE_BATCH_SIZE]
            sentences = [sentence for _, sentence in batch]
            pieces = self._translate_batch(sentences, src_code, tgt_code)
            for (para_idx, _), piece in zip(batch, pieces):
                translated_by_para[para_idx].append(piece)
            done += len(batch)
            if on_progress:
                on_progress(done, total, pieces[-1] if pieces else "")

        paragraphs = [
            "".join(translated_by_para[i]).strip()
            for i in range(paragraph_count)
            if translated_by_para[i]
        ]
        return "\n\n".join(paragraphs)

    def _translate_batch(
        self,
        sentences: list[str],
        src_code: str | None,
        tgt_code: str,
    ) -> list[str]:
        """Translate multiple sentences in one CT2 batch call."""
        assert self._translator is not None
        assert self._tokenizer is not None

        if not sentences:
            return []

        tokenizer = self._tokenizer
        if src_code:
            tokenizer.src_lang = src_code

        sources = [
            tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence))
            for sentence in sentences
        ]
        prefixes = [[tgt_code]] * len(sources)
        results = self._translator.translate_batch(
            sources,
            target_prefix=prefixes,
            beam_size=1,
            max_batch_size=max(TRANSLATE_BATCH_SIZE, len(sources)),
            max_decoding_length=512,
        )

        decoded: list[str] = []
        for result in results:
            if not result.hypotheses:
                decoded.append("")
                continue
            tokens = list(result.hypotheses[0])
            if tokens and tokens[0] == tgt_code:
                tokens = tokens[1:]
            text = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(tokens),
                skip_special_tokens=True,
            ).strip()
            if tgt_code == "jpn_Jpan" and text and text[-1] not in "。．！？":
                text += "。"
            decoded.append(text)
        return decoded
