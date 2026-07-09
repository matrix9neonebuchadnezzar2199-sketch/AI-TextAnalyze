#!/usr/bin/env python3
"""Run All_Status user story tests and generate CSV reports."""

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.user_story_catalog import STORIES  # noqa: E402

MATRIX_PATH = ROOT / "docs" / "user-stories" / "user-stories-matrix.csv"
RESULTS_PATH = ROOT / "docs" / "user-stories" / "test-results.csv"
SUMMARY_PATH = ROOT / "docs" / "user-stories" / "summary.md"


def run_pytest() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    detail = proc.stdout + proc.stderr
    return ok, detail.strip()


def write_matrix() -> None:
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MATRIX_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "story_id",
                "epic",
                "feature",
                "route",
                "persona",
                "user_story",
                "precondition",
                "steps",
                "expected_behavior",
                "apis",
                "impl_status",
                "test_method",
                "test_result",
            ]
        )
        for s in STORIES:
            writer.writerow(
                [
                    s.story_id,
                    s.epic,
                    s.feature,
                    s.route,
                    s.persona,
                    s.user_story,
                    s.precondition,
                    s.steps,
                    s.expected_behavior,
                    s.apis,
                    s.impl_status,
                    s.test_method,
                    "",
                ]
            )


def main() -> int:
    write_matrix()
    pytest_ok, pytest_detail = run_pytest()

    rows: list[dict[str, str]] = []
    fail_count = 0
    for s in STORIES:
        if s.test_method == "unit":
            result = "pass" if pytest_ok else "fail"
        elif s.test_method == "manual":
            result = "skip"
        else:
            result = "skip"
        if result == "fail":
            fail_count += 1
        rows.append({"story_id": s.story_id, "test_method": s.test_method, "result": result})

    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["story_id", "test_method", "result"])
        writer.writeheader()
        writer.writerows(rows)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pass_n = sum(1 for r in rows if r["result"] == "pass")
    skip_n = sum(1 for r in rows if r["result"] == "skip")
    SUMMARY_PATH.write_text(
        f"# Test Summary\n\n"
        f"- Generated: {now}\n"
        f"- Total: {len(rows)}\n"
        f"- Pass: {pass_n}\n"
        f"- Fail: {fail_count}\n"
        f"- Skip: {skip_n}\n\n"
        f"## pytest\n\n```\n{pytest_detail}\n```\n",
        encoding="utf-8",
    )

    print(f"Matrix: {MATRIX_PATH}")
    print(f"Results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"fail={fail_count}")
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
