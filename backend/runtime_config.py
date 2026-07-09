"""Runtime tuning for CPU load and memory."""

from __future__ import annotations

import os


def cpu_thread_count() -> int:
    """Inference thread budget (default 2 — avoids saturating all cores)."""
    raw = os.environ.get("AI_TEXTANALYZE_CPU_THREADS", "2")
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 2


def apply_cpu_thread_env() -> int:
    """Set BLAS/ONNX thread env before model load."""
    threads = cpu_thread_count()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ORT_NUM_THREADS",
    ):
        os.environ[key] = str(threads)
    return threads
