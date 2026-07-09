#!/usr/bin/env python3
"""Download NER/MT models into model/ (one-time setup)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download

from backend.mt_catalog import MT_CATALOG, get_mt_spec

NER_DIR = ROOT / "model" / "ner-gliner-multi-medium-int8"
NER_REPO = "onnx-community/gliner_multi-v2.1"
HF_TOKENIZER_SUBDIR = "hf-tokenizer"


def download_mt_variant(model_id: str) -> None:
    """Download one NLLB CT2 variant and its matching HF tokenizer."""
    spec = get_mt_spec(model_id)
    mt_dir = ROOT / "model" / spec.folder
    tokenizer_dir = mt_dir / HF_TOKENIZER_SUBDIR

    mt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MT ({spec.label}): {spec.download_repo}")
    snapshot_download(spec.download_repo, local_dir=str(mt_dir))

    print(f"Downloading NLLB HF tokenizer: {spec.hf_tokenizer_repo}")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        spec.hf_tokenizer_repo,
        local_dir=str(tokenizer_dir),
        allow_patterns=[
            "tokenizer.json",
            "tokenizer_config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "config.json",
        ],
    )


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
    parser = argparse.ArgumentParser(description="Download AI-TextAnalyze models")
    parser.add_argument(
        "--mt",
        choices=[spec.id for spec in MT_CATALOG],
        help="Download a single MT variant (default: all catalog entries)",
    )
    parser.add_argument("--ner-only", action="store_true", help="Download NER only")
    parser.add_argument("--mt-only", action="store_true", help="Download MT only")
    args = parser.parse_args()

    if args.ner_only:
        download_ner()
    elif args.mt_only:
        if args.mt:
            download_mt_variant(args.mt)
        else:
            for spec in MT_CATALOG:
                download_mt_variant(spec.id)
    elif args.mt:
        download_mt_variant(args.mt)
    else:
        for spec in MT_CATALOG:
            download_mt_variant(spec.id)
        download_ner()

    print("Done.")


if __name__ == "__main__":
    main()
