#!/usr/bin/env python3
"""Summarize public-Sokoban runs from logs and episode sidecars."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def log_metrics(run_dir: Path) -> dict[str, float]:
    logs = sorted((run_dir / "logs").glob("train-*-steps.log"))
    if not logs:
        return {}
    values: dict[str, float] = {}
    step_seconds = []
    names = (
        "training/global_step",
        "critic/score/mean",
        "critic/advantages/mean",
        "actor/grad_norm",
        "probe/attempted",
        "probe/valid",
        "probe/failed",
        "probe/extra_tokens",
        "timing_s/step",
    )
    for line in logs[-1].read_text(errors="replace").splitlines():
        if "training/global_step:" not in line:
            continue
        for name in names:
            match = re.search(rf"(?:^|\s|-) {re.escape(name)}:({NUMBER})", line)
            if match:
                values[name] = float(match.group(1))
        if "timing_s/step" in values:
            match = re.search(rf"timing_s/step:({NUMBER})", line)
            if match:
                step_seconds.append(float(match.group(1)))
    values["gpu_hours"] = sum(step_seconds) / 3600
    return values


def sidecar_metrics(run_dir: Path) -> dict[str, float | int | None]:
    by_split: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for step_dir in sorted((run_dir / "rollouts" / "episodes").glob("step-*")):
        try:
            step = int(step_dir.name.removeprefix("step-"))
        except ValueError:
            continue
        for path in step_dir.glob("*.json"):
            episode = json.loads(path.read_text())
            by_split[str(episode.get("dataset_split", "train"))][step].append(episode)

    result: dict[str, float | int | None] = {
        "final_val_step": None,
        "final_val_success": None,
        "final_val_success_easy": None,
        "final_val_success_medium": None,
        "final_val_success_hard": None,
        "final_train_success": None,
        "main_generated_tokens": 0,
        "probe_extra_tokens": 0,
        "probe_attempts": 0,
        "valid_probes": 0,
        "failed_probes": 0,
        "replay_mismatches": 0,
        "high_impact_anchors": 0,
        "invalid_action_rate": None,
    }
    training_episodes = [
        episode for steps in by_split.get("train", {}).values() for episode in steps
    ]
    turns = [turn for episode in training_episodes for turn in episode.get("turns", [])]
    result["main_generated_tokens"] = sum(
        len(turn.get("generated_token_ids", [])) for turn in turns
    )
    if turns:
        result["invalid_action_rate"] = sum(
            not turn.get("action_valid", False) for turn in turns
        ) / len(turns)
    for episode in training_episodes:
        probe = episode.get("debug_probe")
        if not probe:
            continue
        result["probe_attempts"] += 1
        result["probe_extra_tokens"] += int(probe.get("additional_rollout_tokens", 0))
        if probe.get("skipped_reason"):
            result["failed_probes"] += 1
            continue
        result["valid_probes"] += 1
        result["replay_mismatches"] += int(not probe.get("state_match", False))
        result["high_impact_anchors"] += int(abs(float(probe.get("delta", 0.0))) > 0)

    train_steps = by_split.get("train", {})
    if train_steps:
        latest = train_steps[max(train_steps)]
        result["final_train_success"] = fmean(float(row["final_reward"]) for row in latest)
    validation_steps = by_split.get("test", {})
    if validation_steps:
        latest_step = max(validation_steps)
        latest = validation_steps[latest_step]
        result["final_val_step"] = latest_step
        result["final_val_success"] = fmean(float(row["final_reward"]) for row in latest)
        difficulty = {
            "easy": [row for row in latest if int(row["oracle_shortest_steps"]) <= 2],
            "medium": [row for row in latest if 3 <= int(row["oracle_shortest_steps"]) <= 5],
            "hard": [row for row in latest if int(row["oracle_shortest_steps"]) >= 6],
        }
        for name, episodes in difficulty.items():
            if episodes:
                result[f"final_val_success_{name}"] = fmean(
                    float(row["final_reward"]) for row in episodes
                )
    return result


def summarize(root: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        row = {"method": run_dir.name, **sidecar_metrics(run_dir), **log_metrics(run_dir)}
        main_tokens = int(row["main_generated_tokens"])
        probe_tokens = int(row["probe_extra_tokens"])
        row["rollout_token_overhead"] = probe_tokens / main_tokens if main_tokens else None
        row["high_impact_per_1k_probe_tokens"] = (
            1000 * int(row["high_impact_anchors"]) / probe_tokens if probe_tokens else None
        )
        rows.append(row)
    return rows


def markdown(rows: list[dict]) -> str:
    header = (
        "| method | final val success | probe tokens | overhead | valid/attempted | "
        "high-impact / 1k | GPU h |\n|---|---:|---:|---:|---:|---:|---:|"
    )

    def number(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"

    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['method']} | {number(row['final_val_success'])} | "
            f"{row['probe_extra_tokens']} | {number(row['rollout_token_overhead'])} | "
            f"{row['valid_probes']}/{row['probe_attempts']} | "
            f"{number(row['high_impact_per_1k_probe_tokens'])} | "
            f"{number(row.get('gpu_hours'))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = summarize(args.root)
    if not rows:
        raise ValueError(f"no run directories found under {args.root}")
    (args.root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = markdown(rows)
    (args.root / "summary.md").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
