#!/usr/bin/env python3
"""Download NER/MT models into model/ (one-time setup)."""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
NER_DIR = ROOT / "model" / "ner-gliner-multi-medium-int8"
MT_DIR = ROOT / "model" / "mt-nllb-600m-ct2-int8"

NER_REPO = "onnx-community/gliner_multi-v2.1"
MT_REPO = "Tushe/nllb-200-600M-ct2-int8"


def download_mt() -> None:
    MT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MT: {MT_REPO}")
    snapshot_download(MT_REPO, local_dir=str(MT_DIR))


def download_ner() -> None:
    NER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NER: {NER_REPO}")
    snapshot_download(
        NER_REPO,
        local_dir=str(NER_DIR),
        allow_patterns=[
            "*.json",
            "*.model",
            "onnx/model_int8.onnx",
            "spm.model",
            "tokenizer*",
            "gliner_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )
    src = NER_DIR / "onnx" / "model_int8.onnx"
    dst = NER_DIR / "model.onnx"
    if src.is_file():
        shutil.copy2(src, dst)
        print(f"NER ONNX: {dst}")


def main() -> None:
    download_mt()
    download_ner()
    print("All models downloaded.")


if __name__ == "__main__":
    main()
