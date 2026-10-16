#!/usr/bin/env python3
"""Describe historical runtime logs, without inventing matched before/after speedups."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

from plot_training_curves import read_log_metrics


def audit(run):
    manifest = json.loads((run / "run_manifest.json").read_text())
    expected = int(manifest["total_training_steps"])
    metrics = read_log_metrics(run)
    rows = [metrics[step] for step in range(1, expected + 1) if step in metrics]
    required = ("actor/perf/max_memory_allocated_gb", "actor/perf/max_memory_reserved_gb", "timing_s/step")
    if len(rows) != expected or any(key not in row for row in rows for key in required):
        raise ValueError("incomplete update/memory/time coverage; no inferred values")
    if any(row[key] < 0 for row in rows for key in required) or any(row["timing_s/step"] <= 0 for row in rows):
        raise ValueError("invalid memory/time metric")
    log = sorted((run / "logs").glob("train-*-steps.log"))[-1]
    times = [row["timing_s/step"] for row in rows]
    return {"method": manifest["method"], "seed": manifest["seed"], "updates": expected,
            "logged_allocated_peak_gb": max(row[required[0]] for row in rows),
            "logged_reserved_peak_gb": max(row[required[1]] for row in rows),
            "step_seconds_mean": fmean(times), "step_seconds_min": min(times),
            "step_seconds_max": max(times), "sum_step_seconds": sum(times),
            "manifest_sha256": hashlib.sha256((run / "run_manifest.json").read_bytes()).hexdigest(),
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            "source_log": str(log.relative_to(run)),
            "configuration_caveat": "pre-padding-fix GRPO" if manifest["seed"] == 17 and manifest["method"] == "grpo" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("use a new output directory")
    paths = sorted(args.history_root.rglob("run_manifest.json"))
    if not paths:
        raise ValueError("no historical manifests found")
    rows = [audit(path.parent) for path in paths]
    payload = {"status": "descriptive_historical_logs", "runs": rows,
               "scope": ["trainer-reported memory; not whole colocated GPU peak or model-only allocation",
                         "retain upstream *_gb metric names; do not reinterpret a failed 11.97 GiB request as a peak",
                         "sum timing_s/step is not billed GPU hours or full launcher wall time",
                         "not a matched engineering optimization ablation; no speedup percentage claimed"]}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# 历史运行诊断（不是优化前后消融）", "", "单位保留上游日志 *_gb 命名；内存不是整个共置 GPU 的总峰值。", "",
             "| 方法 | seed | allocated peak (gb) | reserved peak (gb) | 平均 step 秒 | 累计 step 秒 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['seed']} | {row['logged_allocated_peak_gb']:.2f} | "
                     f"{row['logged_reserved_peak_gb']:.2f} | {row['step_seconds_mean']:.2f} | {row['sum_step_seconds']:.2f} |")
    lines += ["", "seed17 GRPO 是 padding 修复前配置；不能拿它作匹配性能基线。",
              "原始日志与 manifest 的 SHA-256 见 summary.json。没有推导优化百分比或吞吐提升。"]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"audited {len(rows)} historical runs; no new model execution")


if __name__ == "__main__":
    main()
