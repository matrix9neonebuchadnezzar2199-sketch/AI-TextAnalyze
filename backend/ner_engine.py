"""GLiNER ONNX NER inference (CPU only)."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# GLiNER 抽出ラベル（仕様固定）
ENTITY_LABELS = ["Person", "Country", "City", "Organization"]

LABEL_TO_TYPE = {
    "Person": "per",
    "Country": "country",
    "City": "city",
    "Organization": "org",
    "person": "per",
    "country": "country",
    "city": "city",
    "organization": "org",
    "org": "org",
}


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


class NerEngine:
    """GLiNER ONNX engine via gliner library (ONNX Runtime backend)."""

    def __init__(self, model_dir: Path, *, intra_op_threads: int = 2, threshold: float = 0.35) -> None:
        self._model_dir = Path(model_dir)
        self._threshold = threshold
        self._intra_op_threads = intra_op_threads
        self._model: Any = None
        self._load_model()

    def _resolve_onnx_file(self) -> str:
        """Prefer int8 ONNX at model root."""
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
        """Release model reference."""
        self._model = None

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from text."""
        if not text.strip():
            return []
        if self._model is None:
            raise RuntimeError("NER engine is not loaded")

        predictions = self._model.predict_entities(
            text,
            ENTITY_LABELS,
            threshold=self._threshold,
            flat_ner=True,
        )
        raw: list[dict[str, Any]] = []
        for pred in predictions:
            term = str(pred.get("text", "")).strip()
            label = str(pred.get("label", "")).strip()
            etype = LABEL_TO_TYPE.get(label)
            if term and etype:
                raw.append({"term": term, "type": etype, "freq": 1})
        return aggregate_keywords(raw)
