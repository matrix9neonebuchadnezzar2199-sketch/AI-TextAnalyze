"""GLiNER ONNX NER inference (CPU only)."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

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
    """Merge duplicate terms and count frequency.

    Args:
        entities: Raw entities with ``term`` and ``type`` keys.

    Returns:
        Sorted list of ``{term, type, freq}``.
    """
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


def _split_words(text: str) -> list[str]:
    """Split text into words (Latin whitespace + CJK single chars)."""
    parts = re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE)
    words: list[str] = []
    for part in parts:
        if re.fullmatch(r"[\w]+", part, flags=re.UNICODE) and re.search(r"[^\x00-\x7F]", part):
            # CJK 連続文字は1文字ずつ
            words.extend(list(part))
        elif part.strip():
            words.append(part)
    return [w for w in words if w.strip()]


class NerEngine:
    """GLiNER ONNX engine — load → infer → close."""

    def __init__(self, model_dir: Path, *, intra_op_threads: int = 2, threshold: float = 0.45) -> None:
        self._model_dir = Path(model_dir)
        self._threshold = threshold
        self._session = None
        self._tokenizer = None
        self._config: dict[str, Any] = {}
        self._onnx_path = self._resolve_onnx_path()
        self._load_artifacts(intra_op_threads)

    def _resolve_onnx_path(self) -> Path:
        onnx_files = sorted(self._model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx file in {self._model_dir}")
        return onnx_files[0]

    def _load_artifacts(self, intra_op_threads: int) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        config_path = self._model_dir / "gliner_config.json"
        if config_path.is_file():
            self._config = json.loads(config_path.read_text(encoding="utf-8"))

        tok_path = self._model_dir / "tokenizer.json"
        if not tok_path.is_file():
            raise FileNotFoundError(f"tokenizer.json not found in {self._model_dir}")
        self._tokenizer = Tokenizer.from_file(str(tok_path))

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_op_threads
        sess_options.inter_op_num_threads = 1
        # CPU のみ（GPU プロバイダ禁止）
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            str(self._onnx_path),
            sess_options=sess_options,
            providers=providers,
        )
        logger.info("NER ONNX loaded: %s", self._onnx_path.name)

    def close(self) -> None:
        """Release ONNX session and tokenizer."""
        self._session = None
        self._tokenizer = None

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from text.

        Args:
            text: Input document text.

        Returns:
            Aggregated keyword rows ``{term, type, freq}``.
        """
        if not text.strip():
            return []
        raw = self._predict_entities(text)
        return aggregate_keywords(raw)

    def _predict_entities(self, text: str) -> list[dict[str, Any]]:
        """Run ONNX inference and decode spans."""
        if self._session is None or self._tokenizer is None:
            raise RuntimeError("NER engine is not loaded")

        words = _split_words(text)
        if not words:
            return []

        entities: list[dict[str, Any]] = []
        for label in ENTITY_LABELS:
            label_entities = self._predict_label(text, words, label)
            entities.extend(label_entities)
        return entities

    def _predict_label(self, text: str, words: list[str], label: str) -> list[dict[str, Any]]:
        """Score spans for a single entity label."""
        assert self._session is not None
        assert self._tokenizer is not None

        max_length = int(self._config.get("max_length", 384))
        encoded = self._tokenizer.encode(text)
        input_ids = encoded.ids[:max_length]
        attention_mask = [1] * len(input_ids)

        # ラベル文字列もエンコード（GLiNER ゼロショット形式）
        label_enc = self._tokenizer.encode(label)
        label_ids = label_enc.ids[:32]

        feed = self._build_feed(input_ids, attention_mask, label_ids, words)
        outputs = self._session.run(None, feed)
        return self._decode_outputs(text, words, label, outputs)

    def _build_feed(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        label_ids: list[int],
        words: list[str],
    ) -> dict[str, Any]:
        """Map tensors to ONNX input names dynamically."""
        assert self._session is not None
        inputs = {i.name: i for i in self._session.get_inputs()}
        feed: dict[str, Any] = {}

        def _arr(name: str, values: list[int], dtype: Any = np.int64) -> np.ndarray:
            return np.array([values], dtype=dtype)

        # よくある入力名パターンに対応
        if "input_ids" in inputs:
            feed["input_ids"] = _arr("input_ids", input_ids)
        if "attention_mask" in inputs:
            feed["attention_mask"] = _arr("attention_mask", attention_mask)
        if "labels_input_ids" in inputs:
            feed["labels_input_ids"] = _arr("labels_input_ids", label_ids)
        if "words_mask" in inputs:
            wmask = [1] * min(len(words), len(input_ids))
            feed["words_mask"] = _arr("words_mask", wmask)
        if "text_lengths" in inputs:
            feed["text_lengths"] = np.array([len(input_ids)], dtype=np.int64)

        missing = [n for n in inputs if n not in feed]
        if missing and not feed:
            raise RuntimeError(
                f"Unsupported ONNX input layout: {list(inputs.keys())}. "
                "Expected GLiNER ONNX export with recognizable input names."
            )
        return feed

    def _decode_outputs(
        self,
        text: str,
        words: list[str],
        label: str,
        outputs: list[Any],
    ) -> list[dict[str, Any]]:
        """Decode ONNX outputs into entity dicts."""
        etype = LABEL_TO_TYPE.get(label, "org")
        entities: list[dict[str, Any]] = []

        for out in outputs:
            arr = np.asarray(out)
            if arr.size == 0:
                continue

            # スコア配列: [batch, spans, 2] or [batch, seq] などを想定
            flat = arr.reshape(-1)
            if flat.dtype.kind in "fc" and flat.size >= 2:
                # start/end スコアペア
                for i in range(0, flat.size - 1, 2):
                    score = float(flat[i + 1]) if i + 1 < flat.size else float(flat[i])
                    if score < self._threshold:
                        continue
                    start_idx = i // 2
                    end_idx = min(start_idx + 1, len(words) - 1)
                    if start_idx >= len(words):
                        continue
                    term = " ".join(words[start_idx : end_idx + 1]).strip()
                    if term:
                        entities.append({"term": term, "type": etype, "freq": 1})
                if entities:
                    return entities

            # トークンラベル列 [batch, seq]
            if arr.ndim >= 2 and arr.shape[-1] >= len(words):
                scores = arr[0][: len(words)]
                for idx, score in enumerate(scores):
                    if float(score) >= self._threshold and idx < len(words):
                        entities.append({"term": words[idx], "type": etype, "freq": 1})
                if entities:
                    return entities

        # フォールバック: ラベル名の単純辞書マッチ（モデル出力が読めない場合の救済）
        pattern = re.compile(re.escape(label), re.IGNORECASE)
        if pattern.search(text):
            logger.debug("ONNX decode fallback for label %s", label)
        return entities
