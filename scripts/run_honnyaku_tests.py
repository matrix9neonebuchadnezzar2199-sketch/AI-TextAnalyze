#!/usr/bin/env python3
"""Batch translate / NER for honnyakutesuto fixtures."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.model_manager import ModelManager  # noqa: E402

EXTRACTED = ROOT / "honnyakutesuto" / "_extracted"
OUT = ROOT / "honnyakutesuto" / "_results"

FIXTURES = [
    ("ru_tass.txt", "ru", "ja"),
    ("zh_qianlong.txt", "zh", "ja"),
    ("ko_nhk.txt", "ko", "ja"),
    ("en_toefl.txt", "en", "ja"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt", default="nllb-1.3b", choices=["nllb-600m", "nllb-1.3b"])
    parser.add_argument("--only", choices=["ru", "zh", "ko", "en"], default=None)
    parser.add_argument("--ner", action="store_true")
    parser.add_argument("--max-chars", type=int, default=0, help="Truncate source for quick runs")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    mgr = ModelManager()
    mgr.set_selected_mt(args.mt)

    fixtures = FIXTURES
    if args.only:
        fixtures = [f for f in fixtures if f[1] == args.only]

    for name, src, tgt in fixtures:
        path = EXTRACTED / name
        text = path.read_text(encoding="utf-8")
        if args.max_chars > 0:
            text = text[: args.max_chars]
        print(f"\n=== MT {args.mt} {src}->{tgt} {name} ({len(text)} chars) ===", flush=True)
        t0 = time.time()
        engine = mgr.load_mt()
        result = engine.translate(text, src, tgt)
        elapsed = time.time() - t0
        dest = OUT / f"{path.stem}__{args.mt}__{src}-{tgt}.txt"
        dest.write_text(result, encoding="utf-8")
        print(f"saved {dest.name} in {elapsed:.1f}s  out_chars={len(result)}", flush=True)
        print("--- preview ---", flush=True)
        print(result[:900], flush=True)

    if args.ner:
        print("\n=== NER ===", flush=True)
        ner = mgr.load_ner()
        for name, src, _tgt in fixtures:
            text = (EXTRACTED / name).read_text(encoding="utf-8")
            if args.max_chars > 0:
                text = text[: args.max_chars]
            kws = ner.extract(text)
            dest = OUT / f"{Path(name).stem}__ner.txt"
            lines = [f"{k['type']}\t{k['term']}\t{k['freq']}" for k in kws]
            dest.write_text("\n".join(lines), encoding="utf-8")
            print(f"{name}: {len(kws)} keywords -> {dest.name}", flush=True)
            for row in lines[:15]:
                print(" ", row, flush=True)

    mgr.unload_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
