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


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for incremental translation."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    chunks = re.split(r"(?<=[。．！？!?\.])\s*|\n+", normalized)
    sentences = [c.strip() for c in chunks if c.strip()]
    return sentences if sentences else [normalized]


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
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
        logger.info("MT CT2 loaded from %s", self._model_dir)

    def close(self) -> None:
        """Release translator and tokenizer."""
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
        """Translate text sentence by sentence."""
        if self._translator is None or self._tokenizer is None:
            raise RuntimeError("MT engine is not loaded")

        src_code = NLLB_LANG_CODES.get(src)
        tgt_code = NLLB_LANG_CODES.get(tgt)
        if tgt_code is None:
            raise ValueError(f"Unsupported target language: {tgt}")
        if src != "auto" and src_code is None:
            raise ValueError(f"Unsupported source language: {src}")

        sentences = split_sentences(text)
        total = len(sentences)
        translated: list[str] = []

        for idx, sentence in enumerate(sentences, start=1):
            piece = self._translate_sentence(sentence, src_code, tgt_code)
            translated.append(piece)
            if on_progress:
                on_progress(idx, total, piece)

        return "\n".join(translated)

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
        )
        if not results or not results[0].hypotheses:
            return ""

        tokens = list(results[0].hypotheses[0])
        if tokens and tokens[0] == tgt_code:
            tokens = tokens[1:]
        return tokenizer.decode(
            tokenizer.convert_tokens_to_ids(tokens),
            skip_special_tokens=True,
        ).strip()
