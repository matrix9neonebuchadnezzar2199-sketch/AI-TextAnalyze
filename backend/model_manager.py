"""Model directory scanning and exclusive NER/MT loading."""

from __future__ import annotations

import gc
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from backend.mt_catalog import DEFAULT_MT_MODEL_ID, MT_CATALOG, get_mt_spec
from backend.mt_engine import MtEngine
from backend.ner_engine import NerEngine
from backend.runtime_config import cpu_thread_count

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


def app_root() -> Path:
    """プロジェクト／配布ルート（開発時はリポジトリ根、凍結時は exe ディレクトリ）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def default_model_dir() -> Path:
    """Resolve the model directory next to the app root."""
    return app_root() / "model"


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
        self._cpu_threads = cpu_threads if cpu_threads is not None else cpu_thread_count()
        self._detected = scan_models(self._model_dir)
        self._active: ActiveEngine = ActiveEngine.NONE
        self._ner: NerEngine | None = None
        self._mt: MtEngine | None = None
        self._selected_mt_id = DEFAULT_MT_MODEL_ID
        self._loaded_mt_id: str | None = None

    @property
    def selected_mt_id(self) -> str:
        return self._selected_mt_id

    def mt_catalog_status(self) -> list[dict[str, Any]]:
        """Catalog entries with on-disk availability."""
        detected_names = {m.name for m in self._detected if m.kind == "mt"}
        return [
            {
                "id": spec.id,
                "label": spec.label,
                "folder": spec.folder,
                "available": spec.folder in detected_names,
            }
            for spec in MT_CATALOG
        ]

    def set_selected_mt(self, model_id: str) -> None:
        """Select MT variant; unload current MT if selection changes."""
        get_mt_spec(model_id)
        if model_id == self._selected_mt_id:
            return
        self.unload_mt()
        self._selected_mt_id = model_id

    @property
    def detected(self) -> list[DetectedModel]:
        return list(self._detected)

    def ner_model(self) -> DetectedModel | None:
        return next((m for m in self._detected if m.kind == "ner"), None)

    def mt_model(self) -> DetectedModel | None:
        spec = get_mt_spec(self._selected_mt_id)
        selected = next(
            (m for m in self._detected if m.kind == "mt" and m.name == spec.folder),
            None,
        )
        if selected is not None:
            return selected
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
            "mt_models": self.mt_catalog_status(),
            "selected_mt_id": self._selected_mt_id,
            "active_engine": self._active.value,
        }

    def mt_is_loaded(self) -> bool:
        return self._mt is not None

    def ner_is_loaded(self) -> bool:
        return self._ner is not None

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
        self._loaded_mt_id = None
        if self._active == ActiveEngine.MT:
            self._active = ActiveEngine.NONE
        gc.collect()

    def mt_matches_selection(self) -> bool:
        """True when loaded MT matches the currently selected catalog id."""
        return self._mt is not None and self._loaded_mt_id == self._selected_mt_id

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

    def load_mt(self, model_id: str | None = None) -> MtEngine:
        """Load MT exclusively (unloads NER first).

        Args:
            model_id: Optional catalog id to select before loading.

        Returns:
            Loaded MtEngine instance.

        Raises:
            FileNotFoundError: No MT model detected.
        """
        if model_id is not None:
            self.set_selected_mt(model_id)

        self.unload_ner()
        if self._mt is not None and self.mt_matches_selection():
            self._active = ActiveEngine.MT
            return self._mt

        meta = self.mt_model()
        if meta is None:
            spec = get_mt_spec(self._selected_mt_id)
            raise FileNotFoundError(
                f"MT model '{spec.label}' not found under {self._model_dir / spec.folder}. "
                f"Run: python scripts/download-models.py --mt {spec.id}"
            )

        self.unload_mt()
        logger.info("Loading MT model: %s", meta.name)
        self._mt = MtEngine(meta.path, intra_threads=self._cpu_threads)
        self._loaded_mt_id = self._selected_mt_id
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
