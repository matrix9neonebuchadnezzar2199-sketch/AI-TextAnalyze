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

# CT2 同梱 tokenizer.json は NLLB 用ではなく誤トークン化するため、HF 正規 tokenizer を別途配置する
NLLB_HF_TOKENIZER_REPO = "facebook/nllb-200-distilled-600M"
HF_TOKENIZER_SUBDIR = "hf-tokenizer"

ProgressCallback = Callable[[int, int, str], None]

# CT2 はバッチ推論の方が逐次より大幅に速い
TRANSLATE_BATCH_SIZE = 16
MAX_PARAGRAPH_CHARS = 1000
MAX_SENTENCE_CHARS = 500
# 段落丸ごと翻訳は長文で出力が途中打ち切りになるため、常に文単位を優先
MAX_WHOLE_PARAGRAPH_CHARS = 0

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_ORPHAN_MARKER_RE = re.compile(r"^\(\d+\)$")
_TOEFL_MARKER_RE = re.compile(r"^\[\s*\]$|^\[X\]$", re.IGNORECASE)
_TOEFL_PARA_ONLY_RE = re.compile(r"^\(\d{1,2}\)$")
_PARA_NUM_START_RE = re.compile(r"^\(\d+\)")
_LEGAL_CITE_INLINE_RE = re.compile(
    r"(?:Sections?\s+)?1260H(?:\([^)]*\))+|10\s+U\.S\.C\.\s*§\s*[^,\n;]+|Public Law \d+-\d+",
    re.IGNORECASE,
)
_ENTITY_SUFFIX_RE = re.compile(
    r"\b(Inc\.|Ltd\.|Limited|Corporation|Corp\.|LLC|PLC|Co\.|Group)\b",
    re.IGNORECASE,
)
_LITERAL_SPLIT_RE = re.compile(
    r"(https?://[^\s\]\)<>]+|www\.[^\s\]\)<>]+|\[\s*\]|\[X\]|\[END\]|-{5,}|"
    r"10\s+U\.S\.C\.\s*§\s*[^,\n;]+|Public Law \d+-\d+|"
    r"(?:Sections?\s+)?1260H(?:\([^)]*\))+)",
    re.IGNORECASE,
)
_COUAGR_RE = re.compile(r"\b[Cc]ougars?\b")
_MOOSE_RE = re.compile(r"\bmoose\b", re.IGNORECASE)


def collapse_soft_linebreaks(text: str) -> str:
    """Join PDF line-wraps inside URLs, paths, and hyphenated words."""
    normalized = text.replace("\r\n", "\n")
    normalized = re.sub(
        r"(?<=[a-zA-Z0-9_./?=&%#:@-])\n(?=[a-zA-Z0-9_./?=&%#:@-])",
        "",
        normalized,
    )
    normalized = re.sub(r"(\w)-\n(\w)", r"\1\2", normalized)
    return normalized


def is_regulatory_list_text(text: str) -> bool:
    """Heuristic: NDAA 1260H entity lists and similar regulatory rosters."""
    return "1260H" in text or "U.S.C." in text


def is_entity_name_line(line: str) -> bool:
    """Latin-only entity / subsidiary lines should not be machine-translated."""
    stripped = line.strip()
    if not stripped or stripped.startswith("•") or stripped.startswith("*"):
        return False
    lower = stripped.lower()
    if any(
        v in lower
        for v in (
            " is ",
            " are ",
            " has ",
            " was ",
            " have ",
            " shall ",
            " because ",
            " directly ",
            " indirectly ",
        )
    ):
        return False
    if _ENTITY_SUFFIX_RE.search(stripped):
        return True
    if re.search(r"\b(formerly|also known as)\b", lower):
        return True
    if re.search(r"\b[a-z]{3,}\b", stripped):
        allowed_lower = {"of", "and", "for", "the", "van", "de", "di", "formerly"}
        lowers = re.findall(r"\b[a-z]{3,}\b", stripped)
        if any(w not in allowed_lower for w in lowers):
            return False
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ,.&()'\-/]*", stripped) and len(stripped) < 160:
        return True
    return False


def format_toefl_paragraph_markers(text: str) -> str:
    """Ensure (1)/(2) markers sit on their own line before translation."""
    return re.sub(r"\((\d{1,2})\)(?=[^\n])", r"(\1)\n", text)


def join_literal_segments(segments: list[tuple[bool, str]]) -> str:
    """Join translated segments with readable spacing around TOEFL markers."""
    out: list[str] = []
    for idx, (is_literal, content) in enumerate(segments):
        if idx > 0:
            prev_literal, prev_content = segments[idx - 1]
            if is_literal and content.strip() in ("[]", "[X]"):
                if out and not out[-1].endswith("\n"):
                    out.append("\n")
            elif is_literal and _TOEFL_PARA_ONLY_RE.fullmatch(content.strip()):
                if out and not out[-1].endswith("\n\n"):
                    out.append("\n\n")
            elif is_literal and re.fullmatch(r"-{5,}", content.strip()):
                if out and not out[-1].endswith("\n"):
                    out.append("\n")
            elif not is_literal and prev_literal and re.fullmatch(
                r"-{5,}", prev_content.strip()
            ):
                if out and not out[-1].endswith("\n\n"):
                    out.append("\n\n")
            elif is_literal and _URL_RE.match(content.strip()):
                if out and not out[-1].endswith("\n"):
                    out.append("\n")
            elif not is_literal and _PARA_NUM_START_RE.match(content.lstrip()):
                if prev_literal and prev_content.strip() in ("[]", "[X]"):
                    if out and not out[-1].endswith("\n\n"):
                        out.append("\n\n" if not out[-1].endswith("\n") else "\n")
            elif not is_literal and prev_literal and _TOEFL_PARA_ONLY_RE.fullmatch(
                prev_content.strip()
            ):
                if out and not out[-1].endswith("\n"):
                    out.append("\n")
        out.append(content)
        if is_literal and content.strip() in ("[]", "[X]") and not content.endswith("\n"):
            out.append("\n")
        if is_literal and content.strip().lower() == "[end]" and not content.endswith("\n"):
            out.append("\n")
        if is_literal and re.fullmatch(r"-{5,}", content.strip()) and not content.endswith("\n"):
            out.append("\n\n")
    return "".join(out)


def normalize_source_for_mt(text: str, src: str, tgt: str) -> str:
    """Apply lightweight source fixes for known NLLB failure modes."""
    if tgt != "ja":
        return text
    if src not in ("en", "auto"):
        return text

    def _cougar_repl(match: re.Match[str]) -> str:
        word = match.group(0)
        plural = word.lower().endswith("s")
        if word[0].isupper():
            return "Mountain lions" if plural else "Mountain lion"
        return "mountain lions" if plural else "mountain lion"

    text = _COUAGR_RE.sub(_cougar_repl, text)
    text = _MOOSE_RE.sub("North American moose", text)
    if is_regulatory_list_text(text):
        return text
    text = format_toefl_paragraph_markers(text)
    # TOEFL 問題文の "paragraph N" は「第N項」に寄せる（"図書" 誤訳回避）
    text = re.sub(r"\bparagraph\s+(\d{1,2})\b", r"item \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bin\s+paragraph\s+(\d{1,2})\b", r"in item \1", text, flags=re.IGNORECASE)
    # "1. Why..." → "Question 1: Why..."（"1 図書" 誤訳回避）
    text = re.sub(r"(?m)^(\d{1,2})\.\s+", r"Question \1: ", text)
    text = re.sub(r"\bThe word\b", "The term", text)
    text = re.sub(r"\bThe phrase\b", "The expression", text)
    return text


def split_literal_segments(text: str) -> list[tuple[bool, str]]:
    """Split text into literal (URL/TOEFL marker) and translatable segments."""
    text = collapse_soft_linebreaks(text)
    segments: list[tuple[bool, str]] = []
    pos = 0
    for match in _LITERAL_SPLIT_RE.finditer(text):
        if match.start() > pos:
            segments.append((False, text[pos : match.start()]))
        segments.append((True, match.group(0)))
        pos = match.end()
    if pos < len(text):
        segments.append((False, text[pos:]))
    if not segments:
        segments.append((False, text))
    return segments


def merge_orphan_markers(paragraphs: list[str]) -> list[str]:
    """Attach standalone (1)/(2) markers to the following paragraph."""
    merged: list[str] = []
    idx = 0
    while idx < len(paragraphs):
        current = paragraphs[idx].strip()
        if _ORPHAN_MARKER_RE.fullmatch(current) and idx + 1 < len(paragraphs):
            merged.append(f"{current}\n{paragraphs[idx + 1].strip()}")
            idx += 2
            continue
        merged.append(current)
        idx += 1
    return merged


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines while keeping paragraph structure."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    paragraphs = parts if parts else [normalized]
    return merge_orphan_markers(paragraphs)


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences (Latin + CJK punctuation)."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []

    # 法令引用 1260H(g)(2)(B) 内のピリオドで誤分割しない
    shielded = paragraph
    cite_hits: list[str] = []

    def _shield(match: re.Match[str]) -> str:
        cite_hits.append(match.group(0))
        return f"⟦C{len(cite_hits) - 1}⟧"

    shielded = _LEGAL_CITE_INLINE_RE.sub(_shield, shielded)
    parts = re.split(r"(?<=[。．！？!?\.])\s*", shielded)
    sentences: list[str] = []
    for part in parts:
        p = part.strip()
        if not p:
            continue
        for idx, cite in enumerate(cite_hits):
            p = p.replace(f"⟦C{idx}⟧", cite)
        sentences.append(p)
    return sentences if sentences else [paragraph]


def split_quiz_options(unit: str) -> list[str]:
    """Split long multiple-choice blocks so one bad option does not loop the batch."""
    if len(unit) < 280 or not re.search(r"\([A-D]\)", unit):
        return [unit]
    parts = re.split(r"(?<=\))\s*(?=\([A-D]\))", unit)
    return [part.strip() for part in parts if part.strip()]


def chunk_paragraph(paragraph: str) -> list[str]:
    """Split into lines/sentences; long blocks are further chunked by character length."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    if (
        MAX_WHOLE_PARAGRAPH_CHARS > 0
        and len(paragraph) <= MAX_WHOLE_PARAGRAPH_CHARS
    ):
        return [paragraph]

    lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
    if len(lines) > 1:
        units: list[str] = []
        for line in lines:
            if is_entity_name_line(line):
                units.append(line)
                continue
            if line.startswith("•"):
                units.extend(_chunk_single_block(line))
                continue
            units.extend(_chunk_single_block(line))
        return units if units else [paragraph]

    return _chunk_single_block(paragraph)


def _chunk_single_block(block: str) -> list[str]:
    units: list[str] = []
    for sentence in split_sentences(block):
        if len(sentence) <= MAX_SENTENCE_CHARS:
            units.extend(split_quiz_options(sentence))
            continue
        for i in range(0, len(sentence), MAX_SENTENCE_CHARS):
            piece = sentence[i : i + MAX_SENTENCE_CHARS]
            units.extend(split_quiz_options(piece))
    return units if units else [block]


def is_pass_through_unit(unit: str) -> bool:
    """Units that should not be sent to the MT model."""
    stripped = unit.strip()
    if not stripped:
        return True
    if _URL_RE.fullmatch(stripped):
        return True
    if _TOEFL_MARKER_RE.fullmatch(stripped):
        return True
    if _TOEFL_PARA_ONLY_RE.fullmatch(stripped):
        return True
    if stripped.lower() == "[end]":
        return True
    if re.fullmatch(r"-{5,}", stripped):
        return True
    if is_entity_name_line(stripped):
        return True
    return False


def join_translated_units(units: list[str]) -> str:
    """Join translated units, preserving line breaks for roster-style lines."""
    if not units:
        return ""
    parts: list[str] = []
    for unit in units:
        parts.append(unit)
        stripped = unit.strip()
        if not stripped:
            continue
        if (
            is_entity_name_line(stripped)
            or stripped.startswith("•")
            or _LEGAL_CITE_INLINE_RE.fullmatch(stripped)
        ) and not unit.endswith("\n"):
            parts.append("\n")
    return "".join(parts)


def iter_translation_units(text: str) -> list[tuple[int, str]]:
    """Flatten paragraphs into translation units with paragraph index."""
    units: list[tuple[int, str]] = []
    paragraphs = split_paragraphs(text)
    for para_idx, paragraph in enumerate(paragraphs):
        for chunk in chunk_paragraph(paragraph):
            units.append((para_idx, chunk))
    return units if units else [(0, text.strip())]


class MtEngine:
    """NLLB CT2 engine — load → translate → close."""

    def __init__(self, model_dir: Path, *, intra_threads: int = 2) -> None:
        self._model_dir = Path(model_dir)
        self._translator: Any = None
        self._tokenizer: Any = None
        self._load_model(intra_threads)

    def _resolve_tokenizer_dir(self) -> Path:
        """Return local HF tokenizer path (required for correct NLLB tokenization)."""
        local = self._model_dir / HF_TOKENIZER_SUBDIR
        if local.is_dir() and (local / "tokenizer_config.json").is_file():
            return local
        raise FileNotFoundError(
            f"NLLB HF tokenizer not found at {local}. "
            "Run: python scripts/download-models.py"
        )

    def _load_hf_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer_dir = self._resolve_tokenizer_dir()
        try:
            return AutoTokenizer.from_pretrained(
                str(tokenizer_dir),
                fix_mistral_regex=True,
            )
        except TypeError:
            return AutoTokenizer.from_pretrained(str(tokenizer_dir))

    def _load_model(self, intra_threads: int) -> None:
        import ctranslate2

        self._translator = ctranslate2.Translator(
            str(self._model_dir),
            device="cpu",
            inter_threads=1,
            intra_threads=intra_threads,
        )
        self._tokenizer = self._load_hf_tokenizer()
        logger.info(
            "MT CT2 loaded from %s (tokenizer: %s)",
            self._model_dir,
            self._resolve_tokenizer_dir(),
        )

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

        segments = split_literal_segments(text)
        built: list[tuple[bool, str]] = []
        for is_literal, content in segments:
            if is_literal:
                built.append((True, content))
                continue
            if not content.strip():
                built.append((False, content))
                continue
            built.append(
                (
                    False,
                    self._translate_body(
                        normalize_source_for_mt(content, src, tgt),
                        src_code,
                        tgt_code,
                        on_progress=on_progress,
                    ),
                )
            )
        return join_literal_segments(built)

    def _translate_body(
        self,
        text: str,
        src_code: str | None,
        tgt_code: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        """Translate a text block that contains no protected literals."""
        units = iter_translation_units(text)
        total = len(units)
        paragraph_count = max((idx for idx, _ in units), default=0) + 1
        translated_by_para: dict[int, list[str]] = {i: [] for i in range(paragraph_count)}

        done = 0
        batch_indices: list[int] = []
        batch_sentences: list[str] = []

        def _flush_batch() -> None:
            nonlocal done, batch_indices, batch_sentences
            if not batch_sentences:
                return
            pieces = self._translate_batch(batch_sentences, src_code, tgt_code)
            for unit_idx, piece in zip(batch_indices, pieces):
                para_idx, original = units[unit_idx]
                translated_by_para[para_idx].append(piece if piece else original)
            done += len(batch_sentences)
            if on_progress:
                on_progress(done, total, pieces[-1] if pieces else "")
            batch_indices = []
            batch_sentences = []

        for unit_idx, (para_idx, sentence) in enumerate(units):
            if is_pass_through_unit(sentence):
                translated_by_para[para_idx].append(sentence.strip())
                done += 1
                if on_progress:
                    on_progress(done, total, sentence.strip())
                continue

            batch_indices.append(unit_idx)
            batch_sentences.append(sentence)
            if len(batch_sentences) >= TRANSLATE_BATCH_SIZE:
                _flush_batch()

        _flush_batch()

        paragraphs = [
            join_translated_units(translated_by_para[i]).strip()
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
            beam_size=2,
            max_batch_size=max(TRANSLATE_BATCH_SIZE, len(sources)),
            max_decoding_length=2048,
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
