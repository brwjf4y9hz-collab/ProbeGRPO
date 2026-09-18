#!/usr/bin/env python3
"""Print a compact Markdown table from ProbeGRPO experiment JSONL records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from probegrpo.metrics import load_jsonl, summarize_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    summaries = summarize_records(load_jsonl(args.path))

    print("| Method | Task | Runs | Reward | Success | Probe/main cost | GPU hours |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for (method, task), values in summaries.items():
        print(
            f"| {method} | {task} | {int(values['runs'])} | "
            f"{values['reward_mean']:.3f} ± {values['reward_std']:.3f} | "
            f"{values['success_mean']:.3f} ± {values['success_std']:.3f} | "
            f"{values['probe_cost_ratio_mean']:.3f} | {values['gpu_hours']:.2f} |"
        )


if __name__ == "__main__":
    main()
