#!/usr/bin/env python3
"""Aggregate matched public-Sokoban runs across random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

METHOD_ORDER = ("grpo", "random_b2", "surprisal_b2", "linear_ucb_b2")
METHOD_LABELS = {
    "grpo": "GRPO",
    "random_b2": "Random-B2",
    "surprisal_b2": "Surprisal-B2",
    "linear_ucb_b2": "LinearUCB-B2",
}
CSV_FIELDS = (
    "seed",
    "method",
    "success_count",
    "eval_count",
    "final_val_success",
    "probe_extra_tokens",
    "rollout_token_overhead",
    "valid_probes",
    "probe_attempts",
    "high_impact_per_1k_probe_tokens",
    "gpu_hours",
)
FLOAT_FIELDS = (
    "final_val_success",
    "rollout_token_overhead",
    "high_impact_per_1k_probe_tokens",
    "gpu_hours",
)
INT_FIELDS = (
    "seed",
    "success_count",
    "eval_count",
    "probe_extra_tokens",
    "valid_probes",
    "probe_attempts",
)


def _optional_float(value: object) -> float | None:
    if value in (None, "", "-"):
        return None
    return float(value)


def read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for field in INT_FIELDS:
                row[field] = int(row[field])
            for field in FLOAT_FIELDS:
                row[field] = _optional_float(row[field])
            rows.append(row)
    return rows


def _seed_from_path(path: Path) -> int:
    match = re.search(r"seed[-_]?([0-9]+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot infer seed from path: {path}")
    return int(match.group(1))


def read_summary_roots(paths: list[Path], eval_count: int) -> list[dict]:
    rows: list[dict] = []
    for root in paths:
        seed = _seed_from_path(root)
        summary_path = root / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        for raw in json.loads(summary_path.read_text()):
            success = float(raw["final_val_success"])
            rows.append(
                {
                    "seed": seed,
                    "method": str(raw["method"]),
                    "success_count": round(success * eval_count),
                    "eval_count": eval_count,
                    "final_val_success": success,
                    "probe_extra_tokens": int(raw["probe_extra_tokens"]),
                    "rollout_token_overhead": _optional_float(
                        raw["rollout_token_overhead"]
                    ),
                    "valid_probes": int(raw["valid_probes"]),
                    "probe_attempts": int(raw["probe_attempts"]),
                    "high_impact_per_1k_probe_tokens": _optional_float(
                        raw["high_impact_per_1k_probe_tokens"]
                    ),
                    "gpu_hours": _optional_float(raw.get("gpu_hours")),
                }
            )
    return rows


def validate_rows(rows: list[dict]) -> list[int]:
    if not rows:
        raise ValueError("no result rows")
    by_seed: dict[int, set[str]] = defaultdict(set)
    seen: set[tuple[int, str]] = set()
    for row in rows:
        key = (int(row["seed"]), str(row["method"]))
        if key in seen:
            raise ValueError(f"duplicate seed/method row: {key}")
        seen.add(key)
        by_seed[key[0]].add(key[1])
        if int(row["eval_count"]) <= 0:
            raise ValueError("eval_count must be positive")
    expected = set(METHOD_ORDER)
    for seed, methods in by_seed.items():
        if methods != expected:
            raise ValueError(f"seed {seed} methods {sorted(methods)} != {sorted(expected)}")
    return sorted(by_seed)


def _sample_sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def aggregate(rows: list[dict]) -> list[dict]:
    seeds = validate_rows(rows)
    baseline = {
        int(row["seed"]): float(row["final_val_success"])
        for row in rows
        if row["method"] == "grpo"
    }
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    result = []
    for method in METHOD_ORDER:
        method_rows = sorted(by_method[method], key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in method_rows] != seeds:
            raise ValueError(f"method {method} does not cover seeds {seeds}")
        success = [float(row["final_val_success"]) for row in method_rows]
        uplifts = [
            float(row["final_val_success"]) - baseline[int(row["seed"])]
            for row in method_rows
        ]
        high_impact = [
            float(row["high_impact_per_1k_probe_tokens"])
            for row in method_rows
            if row["high_impact_per_1k_probe_tokens"] is not None
        ]
        gpu_hours = [
            float(row["gpu_hours"])
            for row in method_rows
            if row["gpu_hours"] is not None
        ]
        result.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "seeds": seeds,
                "success_by_seed": success,
                "success_mean": fmean(success),
                "success_sample_sd": _sample_sd(success),
                "paired_uplift_pp_mean": 100 * fmean(uplifts),
                "paired_uplift_pp_sample_sd": 100 * _sample_sd(uplifts),
                "probe_tokens_mean": fmean(
                    int(row["probe_extra_tokens"]) for row in method_rows
                ),
                "rollout_overhead_mean": fmean(
                    float(row["rollout_token_overhead"]) for row in method_rows
                ),
                "valid_probes": sum(int(row["valid_probes"]) for row in method_rows),
                "probe_attempts": sum(int(row["probe_attempts"]) for row in method_rows),
                "high_impact_per_1k_mean": fmean(high_impact) if high_impact else None,
                "gpu_hours_mean": fmean(gpu_hours) if gpu_hours else None,
            }
        )
    return result


def markdown(aggregates: list[dict]) -> str:
    seeds = aggregates[0]["seeds"]
    seed_headers = " | ".join(f"seed {seed}" for seed in seeds)
    lines = [
        f"| method | {seed_headers} | mean +/- sample SD | paired uplift vs GRPO | "
        "rollout overhead | valid probes | high-impact / 1k | GPU h |",
        "|---|" + "---:|" * len(seeds) + "---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        seed_values = " | ".join(f"{100 * value:.1f}%" for value in row["success_by_seed"])
        high_impact = row["high_impact_per_1k_mean"]
        high_impact_text = "-" if high_impact is None else f"{high_impact:.2f}"
        lines.append(
            f"| {row['label']} | {seed_values} | "
            f"{100 * row['success_mean']:.1f}% +/- "
            f"{100 * row['success_sample_sd']:.1f}% | "
            f"{row['paired_uplift_pp_mean']:+.1f} +/- "
            f"{row['paired_uplift_pp_sample_sd']:.1f} pp | "
            f"{100 * row['rollout_overhead_mean']:.1f}% | "
            f"{row['valid_probes']}/{row['probe_attempts']} | {high_impact_text} | "
            f"{row['gpu_hours_mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (int(item["seed"]), item["method"])):
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="*", type=Path, help="seed result directories")
    parser.add_argument("--csv", type=Path, help="use an existing compact per-seed CSV")
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.csv) == bool(args.roots):
        parser.error("provide either --csv or one or more seed result directories")

    rows = read_csv(args.csv) if args.csv else read_summary_roots(args.roots, args.eval_count)
    aggregates = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.csv:
        write_csv(args.output_dir / "per_seed.csv", rows)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregates, indent=2) + "\n"
    )
    report = markdown(aggregates)
    (args.output_dir / "summary.md").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
