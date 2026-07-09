"""Model directory scanning and exclusive NER/MT loading."""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from backend.mt_engine import MtEngine
from backend.ner_engine import NerEngine

logger = logging.getLogger(__name__)

ModelKind = Literal["ner", "mt"]


class ActiveEngine(str, Enum):
    """Currently loaded engine kind (at most one)."""

    NONE = "none"
    NER = "ner"
    MT = "mt"


@dataclass(frozen=True)
class DetectedModel:
    """A model folder discovered under ``model/``."""

    kind: ModelKind
    name: str
    path: Path


def default_model_dir() -> Path:
    """Resolve the model directory relative to project root."""
    return Path(__file__).resolve().parent.parent / "model"


def _has_ct2_model(folder: Path) -> bool:
    """CTranslate2: model.bin + config.json + shared_vocabulary*."""
    if not (folder / "model.bin").is_file():
        return False
    if not (folder / "config.json").is_file():
        return False
    vocab = list(folder.glob("shared_vocabulary*"))
    return len(vocab) > 0


def _has_gliner_onnx(folder: Path) -> bool:
    """GLiNER ONNX: *.onnx + tokenizer.json."""
    onnx_files = list(folder.glob("*.onnx"))
    return len(onnx_files) > 0 and (folder / "tokenizer.json").is_file()


def scan_models(model_dir: Path | None = None) -> list[DetectedModel]:
    """Scan ``model/`` subfolders and classify by file layout.

    Args:
        model_dir: Override model root (for tests).

    Returns:
        Detected model descriptors.
    """
    root = model_dir or default_model_dir()
    detected: list[DetectedModel] = []
    if not root.is_dir():
        return detected

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if _has_gliner_onnx(child):
            detected.append(DetectedModel(kind="ner", name=child.name, path=child))
        elif _has_ct2_model(child):
            detected.append(DetectedModel(kind="mt", name=child.name, path=child))

    return detected


class ModelManager:
    """Exclusive loader: at most one of NER or MT in memory."""

    def __init__(self, model_dir: Path | None = None, *, cpu_threads: int | None = None) -> None:
        self._model_dir = model_dir or default_model_dir()
        self._cpu_threads = cpu_threads or max(1, (os.cpu_count() or 2) - 1)
        self._detected = scan_models(self._model_dir)
        self._active: ActiveEngine = ActiveEngine.NONE
        self._ner: NerEngine | None = None
        self._mt: MtEngine | None = None

    @property
    def detected(self) -> list[DetectedModel]:
        return list(self._detected)

    def ner_model(self) -> DetectedModel | None:
        return next((m for m in self._detected if m.kind == "ner"), None)

    def mt_model(self) -> DetectedModel | None:
        return next((m for m in self._detected if m.kind == "mt"), None)

    @property
    def active_engine(self) -> ActiveEngine:
        return self._active

    def status_dict(self) -> dict[str, Any]:
        """Summary for UI status bar."""
        ner = self.ner_model()
        mt = self.mt_model()
        return {
            "ner_name": ner.name if ner else None,
            "ner_available": ner is not None,
            "mt_name": mt.name if mt else None,
            "mt_available": mt is not None,
            "active_engine": self._active.value,
        }

    def unload_ner(self) -> None:
        """Release NER engine and session."""
        if self._ner is not None:
            self._ner.close()
            self._ner = None
        if self._active == ActiveEngine.NER:
            self._active = ActiveEngine.NONE
        gc.collect()

    def unload_mt(self) -> None:
        """Release MT engine and session."""
        if self._mt is not None:
            self._mt.close()
            self._mt = None
        if self._active == ActiveEngine.MT:
            self._active = ActiveEngine.NONE
        gc.collect()

    def unload_all(self) -> None:
        """Release any loaded engine."""
        self.unload_ner()
        self.unload_mt()

    def load_ner(self) -> NerEngine:
        """Load NER exclusively (unloads MT first).

        Returns:
            Loaded NerEngine instance.

        Raises:
            FileNotFoundError: No NER model detected.
            RuntimeError: Another engine is active and could not be released.
        """
        self.unload_mt()
        if self._ner is not None:
            self._active = ActiveEngine.NER
            return self._ner

        meta = self.ner_model()
        if meta is None:
            raise FileNotFoundError(
                f"NER model not found under {self._model_dir}. "
                "Place GLiNER ONNX files (model.onnx, tokenizer.json) in model/."
            )

        logger.info("Loading NER model: %s", meta.name)
        self._ner = NerEngine(meta.path, intra_op_threads=self._cpu_threads)
        self._active = ActiveEngine.NER
        return self._ner

    def load_mt(self) -> MtEngine:
        """Load MT exclusively (unloads NER first).

        Returns:
            Loaded MtEngine instance.

        Raises:
            FileNotFoundError: No MT model detected.
        """
        self.unload_ner()
        if self._mt is not None:
            self._active = ActiveEngine.MT
            return self._mt

        meta = self.mt_model()
        if meta is None:
            raise FileNotFoundError(
                f"MT model not found under {self._model_dir}. "
                "Place CTranslate2 NLLB files in model/."
            )

        logger.info("Loading MT model: %s", meta.name)
        self._mt = MtEngine(meta.path, intra_threads=self._cpu_threads)
        self._active = ActiveEngine.MT
        return self._mt

    def with_ner(self, fn: Any) -> Any:
        """Run callable with NER loaded, then unload."""
        engine = self.load_ner()
        try:
            return fn(engine)
        finally:
            self.unload_ner()

    def with_mt(self, fn: Any) -> Any:
        """Run callable with MT loaded, then unload."""
        engine = self.load_mt()
        try:
            return fn(engine)
        finally:
            self.unload_mt()
