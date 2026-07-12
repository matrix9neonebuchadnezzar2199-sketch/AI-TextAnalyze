"""GLiNER ONNX NER inference (CPU only)."""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

ENTITY_LABELS = ["Person", "Country", "City", "Organization"]
# CJK 文書向けの補助ラベル（GLiNER が英語ラベルだけだと取りこぼしやすい）
ENTITY_LABELS_CJK = [
    "Person",
    "人名",
    "Country",
    "国家",
    "国家/地区",
    "City",
    "城市",
    "地名",
    "Organization",
    "组织",
    "机构",
]

LABEL_TO_TYPE = {
    "Person": "per",
    "人名": "per",
    "Country": "country",
    "国家": "country",
    "国家/地区": "country",
    "City": "city",
    "城市": "city",
    "地名": "city",
    "Organization": "org",
    "组织": "org",
    "机构": "org",
    "person": "per",
    "country": "country",
    "city": "city",
    "organization": "org",
    "org": "org",
}

# GLiNER max_len=384 相当。int8 モデルはスコアが低めなので閾値も下げる
DEFAULT_THRESHOLD = 0.2
CHUNK_MAX_CHARS = 1200
CHUNK_YIELD_SECONDS = 0.01

ProgressCallback = Callable[[int, int, str], None]


def aggregate_keywords(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate terms and count frequency."""
    counter: Counter[tuple[str, str]] = Counter()
    display: dict[tuple[str, str], str] = {}
    for ent in entities:
        term = str(ent.get("term", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not term or not etype:
            continue
        key = (term.casefold(), etype)
        counter[key] += int(ent.get("freq", 1))
        display.setdefault(key, term)

    rows = [
        {"term": display[key], "type": key[1], "freq": freq}
        for key, freq in counter.items()
    ]
    rows.sort(key=lambda r: (-r["freq"], r["term"]))
    return rows


def chunk_text(text: str, *, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Split long documents into GLiNER-safe chunks.

    GLiNER truncates around 384 tokens; chunking avoids silent loss on long input.
    """
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    paragraphs = re.split(r"\n\s*\n", normalized)
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            flush()
            sentences = re.split(r"(?<=[.!?。．！？])\s+", paragraph)
            part = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) > max_chars:
                    flush()
                    for i in range(0, len(sentence), max_chars):
                        chunks.append(sentence[i : i + max_chars])
                    part = ""
                    continue
                candidate = f"{part} {sentence}".strip() if part else sentence
                if len(candidate) <= max_chars:
                    part = candidate
                else:
                    if part:
                        chunks.append(part)
                    part = sentence
            if part:
                chunks.append(part)
            continue

        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            flush()
            buffer = paragraph

    flush()
    return chunks if chunks else [normalized[:max_chars]]


class NerEngine:
    """GLiNER ONNX engine via gliner library (ONNX Runtime backend)."""

    def __init__(self, model_dir: Path, *, intra_op_threads: int = 2, threshold: float = DEFAULT_THRESHOLD) -> None:
        self._model_dir = Path(model_dir)
        self._threshold = threshold
        self._model: Any = None
        self._load_model()

    def _resolve_onnx_file(self) -> str:
        preferred = self._model_dir / "model.onnx"
        if preferred.is_file():
            return preferred.name
        onnx_files = sorted(self._model_dir.glob("*.onnx"))
        if not onnx_files:
            nested = sorted((self._model_dir / "onnx").glob("model_int8.onnx"))
            if nested:
                return str(nested[0].relative_to(self._model_dir)).replace("\\", "/")
            raise FileNotFoundError(f"No .onnx file in {self._model_dir}")
        return onnx_files[0].name

    def _load_model(self) -> None:
        from gliner import GLiNER

        onnx_file = self._resolve_onnx_file()
        self._model = GLiNER.from_pretrained(
            str(self._model_dir),
            load_onnx_model=True,
            onnx_model_file=onnx_file,
            map_location="cpu",
        )
        logger.info("NER GLiNER ONNX loaded: %s (%s)", self._model_dir.name, onnx_file)

    def close(self) -> None:
        self._model = None

    def extract(
        self,
        text: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        """Extract entities from text (chunked for long documents)."""
        if not text.strip():
            return []
        if self._model is None:
            raise RuntimeError("NER engine is not loaded")

        # PDF 由来の CJK/ハングル空白を潰してから NER（エンティティ切断を防ぐ）
        from backend.mt_engine import collapse_pdf_spacing

        chunks = chunk_text(collapse_pdf_spacing(text))
        # CJK は int8 スコアが低めになりやすい
        has_cjk = bool(
            re.search(r"[\u4e00-\u9fff\uac00-\ud7a3]", text)
        )
        labels = ENTITY_LABELS_CJK if has_cjk else ENTITY_LABELS
        threshold = min(self._threshold, 0.12) if has_cjk else self._threshold
        total = len(chunks)
        raw: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks, start=1):
            if on_progress:
                on_progress(idx, total, f"チャンク {idx}/{total}")
            predictions = self._model.predict_entities(
                chunk,
                labels,
                threshold=threshold,
                flat_ner=True,
            )
            for pred in predictions:
                term = str(pred.get("text", "")).strip()
                label = str(pred.get("label", "")).strip()
                etype = LABEL_TO_TYPE.get(label)
                if term and etype:
                    raw.append({"term": term, "type": etype, "freq": 1})
            if CHUNK_YIELD_SECONDS > 0 and idx < total:
                time.sleep(CHUNK_YIELD_SECONDS)

        return aggregate_keywords(raw)
