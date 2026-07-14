#!/usr/bin/env python3
"""Translate 1260H sample/full for iterative accuracy evaluation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.model_manager import ModelManager  # noqa: E402

OUT = ROOT / "honnyakutesuto" / "_results" / "1260h"
SRC = ROOT / "honnyakutesuto" / "_extracted" / "en_1260h.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt", default="nllb-1.3b", choices=["nllb-600m", "nllb-1.3b"])
    parser.add_argument("--tag", default="r0")
    parser.add_argument("--max-chars", type=int, default=2800)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text(encoding="utf-8")
    if not args.full and args.max_chars > 0:
        text = text[: args.max_chars]

    src_path = OUT / f"sample_{args.tag}_src.txt"
    src_path.write_text(text, encoding="utf-8")
    print(f"chars={len(text)} mt={args.mt} tag={args.tag}", flush=True)

    mgr = ModelManager()
    mgr.set_selected_mt(args.mt)
    engine = mgr.load_mt()

    def on_progress(current: int, total: int, detail: str) -> None:
        if current == 1 or current == total or current % 10 == 0:
            print(f"  progress {current}/{total}", flush=True)

    t0 = time.time()
    result = engine.translate(text, "en", "ja", on_progress=on_progress)
    elapsed = time.time() - t0
    engine.close()

    ja_path = OUT / f"sample_{args.tag}_ja.txt"
    ja_path.write_text(result.text, encoding="utf-8")
    units_lines: list[str] = []
    for unit in result.units:
        units_lines.append(f"[{unit['id']}]\nSRC: {unit['src']}\nTGT: {unit['tgt']}")
    (OUT / f"sample_{args.tag}_units.txt").write_text(
        "\n---\n".join(units_lines), encoding="utf-8"
    )
    print(f"elapsed={elapsed:.1f}s units={len(result.units)} out={len(result.text)}", flush=True)
    print("=== OUTPUT ===", flush=True)
    print(result.text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
