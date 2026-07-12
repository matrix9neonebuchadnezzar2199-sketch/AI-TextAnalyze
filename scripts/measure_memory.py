#!/usr/bin/env python3
"""Measure memory at each AI-TextAnalyze lifecycle phase (Api backend, no GUI)."""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.api import Api  # noqa: E402

SAMPLE_EN = (
    "The Deputy Secretary of Defense has determined that the following entities "
    "qualify for designation as Chinese military companies in accordance with "
    "section 1260H. Qihoo 360 is affiliated with MIIT (Section 1260H(g)(2)(B)(i)(I))."
)
SAMPLE_NER = (
    "Barack Obama visited Tokyo. Microsoft and Google announced a partnership "
    "in Germany. The United Nations met in New York."
)


def mem_snapshot() -> dict[str, float | str]:
    """Return current process memory (Windows-friendly)."""
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        full = proc.memory_full_info()
        return {
            "rss_mb": round(info.rss / 1024 / 1024, 1),
            "working_set_mb": round(info.wss / 1024 / 1024, 1) if hasattr(info, "wss") else round(info.rss / 1024 / 1024, 1),
            "private_mb": round(full.private / 1024 / 1024, 1),
        }
    except ImportError:
        return {"rss_mb": -1.0, "working_set_mb": -1.0, "private_mb": -1.0}


def record(phase: str, rows: list[dict]) -> None:
    gc.collect()
    time.sleep(0.5)
    snap = mem_snapshot()
    row = {"phase": phase, **snap}
    rows.append(row)
    priv = snap.get("private_mb", "?")
    ws = snap.get("working_set_mb", "?")
    print(f"[{phase}] Working Set {ws} MB | Private {priv} MB", flush=True)


def main() -> int:
    rows: list[dict] = []
    api = Api()

    record("01_baseline_imports", rows)

    warm = api.warmup_mt("nllb-600m")
    if not warm.get("ok"):
        print("warmup_mt(600m) failed:", warm.get("error"), file=sys.stderr)
        return 1
    record("02_mt_600m_loaded", rows)

    tr = api.translate(SAMPLE_EN, "en", "ja")
    if not tr.get("ok"):
        print("translate(600m) failed:", tr.get("error"), file=sys.stderr)
        return 1
    record("03_mt_600m_after_translate", rows)

    sw = api.select_mt_model("nllb-1.3b")
    if not sw.get("ok"):
        print("select_mt_model(1.3b) failed:", sw.get("error"), file=sys.stderr)
        return 1
    record("04_mt_1.3b_loaded", rows)

    tr2 = api.translate(SAMPLE_EN, "en", "ja")
    if not tr2.get("ok"):
        print("translate(1.3b) failed:", tr2.get("error"), file=sys.stderr)
        return 1
    record("05_mt_1.3b_after_translate", rows)

    ex = api.extract_keywords(SAMPLE_NER)
    if not ex.get("ok"):
        print("extract_keywords failed:", ex.get("error"), file=sys.stderr)
        return 1
    record("06_ner_after_extract", rows)

    print("\n=== SUMMARY ===")
    for row in rows:
        print(
            f"{row['phase']:28}  WS {row['working_set_mb']:>7} MB  "
            f"Private {row['private_mb']:>7} MB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
