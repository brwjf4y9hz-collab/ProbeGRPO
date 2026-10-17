#!/usr/bin/env python3
"""Export per-step verl metrics and Sokoban train/validation curves as CSV and SVG."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from xml.sax.saxutils import escape

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC = re.compile(
    rf"(?<![\w./@-])([A-Za-z_][\w./@-]*):\s*"
    rf"(?:np\.(?:float\d+|int\d+)\()?({NUMBER})"
)

PLOTS = (
    ("critic/score/mean", "Training reward"),
    ("critic/advantages/mean", "Mean policy advantage"),
    ("actor/pg_loss", "Actor policy loss"),
    ("actor/grad_norm", "Gradient norm"),
    ("train_success", "Training success (%)"),
    ("val_success", "Validation success (%)"),
    ("probe/valid", "Valid probes per update"),
    ("probe/extra_tokens", "Probe tokens per update"),
)
COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#000000", "#0072B2",
)


def read_log_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    logs = sorted((run_dir / "logs").glob("train-*-steps.log"))
    if not logs:
        raise FileNotFoundError(f"no train-*-steps.log found under {run_dir / 'logs'}")
    result: dict[int, dict[str, float]] = {}
    for line in logs[-1].read_text(errors="replace").splitlines():
        if "training/global_step:" not in line and "val-core/" not in line:
            continue
        metrics = {key: float(value) for key, value in METRIC.findall(line)}
        metrics = {key: value for key, value in metrics.items() if math.isfinite(value)}
        step = metrics.get("training/global_step", metrics.get("step"))
        if step is not None:
            result.setdefault(int(step), {}).update(metrics)
            val_key = "val-core/probegrpo_ragen_public_sokoban/reward/mean@1"
            if val_key in metrics:
                result[int(step)]["val_success"] = metrics[val_key]
    return result


def read_episode_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    episodes_root = run_dir / "rollouts" / "episodes"
    for step_dir in episodes_root.glob("step-*"):
        try:
            step = int(step_dir.name.removeprefix("step-"))
        except ValueError:
            continue
        for path in step_dir.glob("*.json"):
            episode = json.loads(path.read_text())
            split = str(episode.get("dataset_split", "train"))
            grouped[(step, split)].append(episode)

    result: dict[int, dict[str, float]] = defaultdict(dict)
    for (step, split), episodes in grouped.items():
        if not episodes:
            continue
        success = fmean(float(row["final_reward"]) for row in episodes)
        if split in {"test", "val", "validation"}:
            result[step]["val_success"] = success
            result[step]["val_episodes"] = len(episodes)
        elif split == "train":
            result[step]["train_success"] = success
            result[step]["train_episodes"] = len(episodes)
            turns = [turn for episode in episodes for turn in episode.get("turns", [])]
            result[step]["main_generated_tokens"] = sum(
                len(turn.get("generated_token_ids", [])) for turn in turns
            )
            if turns:
                result[step]["invalid_action_rate"] = sum(
                    not turn.get("action_valid", False) for turn in turns
                ) / len(turns)
            result[step]["probe/extra_tokens"] = float(
                sum(
                    int((row.get("debug_probe") or {}).get("additional_rollout_tokens", 0))
                    for row in episodes
                )
            )
            valid_probes = [
                episode["debug_probe"] for episode in episodes
                if episode.get("debug_probe") and not episode["debug_probe"].get("skipped_reason")
            ]
            if valid_probes:
                result[step]["probe/abs_delta_mean"] = fmean(
                    abs(float(probe.get("delta", 0))) for probe in valid_probes
                )
                result[step]["probe/high_impact"] = sum(
                    abs(float(probe.get("delta", 0))) > 0 for probe in valid_probes
                )
    return result


def collect_metrics(run_dir: Path) -> list[dict[str, float]]:
    by_step = read_log_metrics(run_dir)
    for step, metrics in read_episode_metrics(run_dir).items():
        by_step.setdefault(step, {}).update(metrics)
    token_total, seconds_total = 0.0, 0.0
    for step in sorted(by_step):
        values = by_step[step]
        token_total += values.get("main_generated_tokens", 0) + values.get("probe/extra_tokens", 0)
        seconds_total += values.get("timing_s/step", 0)
        values["cumulative_rollout_tokens"] = token_total
        values["cumulative_update_seconds"] = seconds_total
    return [{"step": step, **by_step[step]} for step in sorted(by_step)]


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    columns = ["step"] + sorted({key for row in rows for key in row if key != "step"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _text(x: float, y: float, value: str, **attrs: object) -> str:
    options = " ".join(
        f'{name.replace("_", "-")}="{escape(str(setting))}"'
        for name, setting in attrs.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {options}>{escape(value)}</text>'


def render_svg(rows: list[dict[str, float]], output: Path, title: str) -> None:
    width, height = 1050, 820
    margin_x, margin_y = 72, 62
    gap_x, gap_y = 60, 58
    panel_rows = (len(PLOTS) + 1) // 2
    panel_w = (width - 2 * margin_x - gap_x) / 2
    panel_h = (height - 2 * margin_y - gap_y * (panel_rows - 1)) / panel_rows
    steps = [row["step"] for row in rows]
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#222">',
        _text(width / 2, 30, title, text_anchor="middle", font_size="18", font_weight="bold"),
    ]
    for index, (metric, label) in enumerate(PLOTS):
        col, row_index = index % 2, index // 2
        x = margin_x + col * (panel_w + gap_x)
        y = margin_y + row_index * (panel_h + gap_y)
        aliases = {
            "critic/score/mean": ("critic/score/mean", "critic/rewards/mean"),
            "actor/pg_loss": ("actor/pg_loss", "actor/total_loss", "actor/loss"),
        }.get(metric, (metric,))
        metric_key = next((key for key in aliases if any(key in item for item in rows)), metric)
        scale = 100.0 if metric in {"train_success", "val_success"} else 1.0
        values = [(item["step"], item[metric_key] * scale) for item in rows if metric_key in item]
        parts.extend(
            [
                _text(x, y - 12, label, font_size="14", font_weight="bold"),
                (
                    f'<line x1="{x}" y1="{y + panel_h}" '
                    f'x2="{x + panel_w}" y2="{y + panel_h}" stroke="#555"/>'
                ),
                f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + panel_h}" stroke="#555"/>',
            ]
        )
        if not values:
            parts.append(
                _text(
                    x + panel_w / 2,
                    y + panel_h / 2,
                    "not recorded",
                    text_anchor="middle",
                    font_size="13",
                    fill="#777",
                )
            )
            continue
        low, high = min(value for _, value in values), max(value for _, value in values)
        if low == high:
            padding = max(abs(low) * 0.05, 0.05)
            low, high = low - padding, high + padding
        else:
            padding = (high - low) * 0.08
            low, high = low - padding, high + padding
        min_step, max_step = min(steps), max(steps)

        def point(step: float, value: float) -> tuple[float, float]:
            px = x + (step - min_step) / max(max_step - min_step, 1) * panel_w
            py = y + panel_h - (value - low) / (high - low) * panel_h
            return px, py

        for tick_value in (low, (low + high) / 2, high):
            ty = point(min_step, tick_value)[1]
            parts.append(
                f'<line x1="{x}" y1="{ty:.1f}" '
                f'x2="{x + panel_w}" y2="{ty:.1f}" stroke="#E5E5E5"/>'
            )
            parts.append(
                _text(
                    x - 8,
                    ty + 4,
                    f"{tick_value:.3g}",
                    text_anchor="end",
                    font_size="10",
                    fill="#555",
                )
            )
        points = [point(step, value) for step, value in values]
        if len(points) > 1:
            parts.append(
                '<polyline points="'
                + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
                + f'" fill="none" stroke="{COLORS[index]}" stroke-width="2.2"/>'
            )
        for px, py in points:
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" '
                f'r="2.8" fill="{COLORS[index]}"/>'
            )
        parts.append(_text(x, y + panel_h + 18, f"step {min_step}", font_size="10", fill="#555"))
        parts.append(
            _text(
                x + panel_w,
                y + panel_h + 18,
                f"step {max_step}",
                text_anchor="end",
                font_size="10",
                fill="#555",
            )
        )
    parts.extend(["</g>", "</svg>"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="one run output directory")
    parser.add_argument("--output-dir", type=Path, help="defaults to RUN_DIR/curves")
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "curves"
    rows = collect_metrics(args.run_dir)
    if not rows:
        raise ValueError(f"no training metric steps found under {args.run_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "metrics_by_step.csv")
    render_svg(rows, output_dir / "training_curves.svg", f"{args.run_dir.name}: training curves")
    print(f"wrote {output_dir / 'metrics_by_step.csv'}")
    print(f"wrote {output_dir / 'training_curves.svg'}")


if __name__ == "__main__":
    main()
