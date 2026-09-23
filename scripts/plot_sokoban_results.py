#!/usr/bin/env python3
"""Render the committed three-seed Sokoban result as a dependency-free SVG."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from xml.sax.saxutils import escape

METHODS = ("grpo", "random_b2", "surprisal_b2", "linear_ucb_b2")
LABELS = {
    "grpo": "GRPO",
    "random_b2": "Random",
    "surprisal_b2": "Surprisal",
    "linear_ucb_b2": "LinearUCB",
}
COLORS = {
    "grpo": "#4D4D4D",
    "random_b2": "#E69F00",
    "surprisal_b2": "#56B4E9",
    "linear_ucb_b2": "#009E73",
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "seed": int(raw["seed"]),
                    "method": raw["method"],
                    "success": float(raw["final_val_success"]),
                    "overhead": float(raw["rollout_token_overhead"]),
                }
            )
    return rows


def _text(x: float, y: float, value: str, **attrs: object) -> str:
    options = " ".join(
        f'{name.replace("_", "-")}="{escape(str(setting))}"'
        for name, setting in attrs.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {options}>{escape(value)}</text>'


def render_svg(rows: list[dict], output: Path) -> None:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    if any(len(by_method[method]) != 3 for method in METHODS):
        raise ValueError("the main figure requires exactly three rows per method")

    width, height = 980, 450
    left = (70, 45, 440, 330)
    right = (580, 45, 330, 330)
    y_min, y_max = 0.15, 0.38

    def ymap(value: float, panel: tuple[int, int, int, int]) -> float:
        _, y, _, h = panel
        return y + h - (value - y_min) / (y_max - y_min) * h

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<title>ProbeGRPO public Sokoban three-seed results</title>",
        (
            "<desc>Per-seed success rates and the trade-off between mean success "
            "and extra rollout tokens.</desc>"
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#222">',
    ]
    for panel in (left, right):
        x, y, w, h = panel
        parts.append(
            f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#222"/>'
        )
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#222"/>')
        for tick in (0.20, 0.25, 0.30, 0.35):
            ty = ymap(tick, panel)
            parts.append(
                f'<line x1="{x}" y1="{ty:.1f}" x2="{x + w}" y2="{ty:.1f}" stroke="#E5E5E5"/>'
            )
            parts.append(
                _text(x - 10, ty + 4, f"{100 * tick:.0f}", text_anchor="end", font_size="12")
            )

    parts.extend(
        [
            _text(20, 28, "A", font_size="18", font_weight="bold"),
            _text(530, 28, "B", font_size="18", font_weight="bold"),
            _text(
                290,
                24,
                "Final success across seeds",
                text_anchor="middle",
                font_size="16",
                font_weight="bold",
            ),
            _text(
                745,
                24,
                "Accuracy-cost trade-off",
                text_anchor="middle",
                font_size="16",
                font_weight="bold",
            ),
            _text(
                18,
                210,
                "Success rate (%)",
                text_anchor="middle",
                font_size="13",
                transform="rotate(-90 18 210)",
            ),
            _text(
                530,
                210,
                "Success rate (%)",
                text_anchor="middle",
                font_size="13",
                transform="rotate(-90 530 210)",
            ),
        ]
    )

    x_positions = {
        method: left[0] + 55 + index * 108 for index, method in enumerate(METHODS)
    }
    seeds = sorted({row["seed"] for row in rows})
    seed_offsets = dict(zip(seeds, (-11, 0, 11)))
    for seed in seeds:
        points = []
        for method in METHODS:
            row = next(item for item in by_method[method] if item["seed"] == seed)
            points.append(
                (x_positions[method] + seed_offsets[seed], ymap(row["success"], left))
            )
        parts.append(
            '<polyline points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            + '" fill="none" stroke="#B8B8B8" stroke-width="1" opacity="0.65"/>'
        )
        for method, (x, y) in zip(METHODS, points):
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{COLORS[method]}"/>'
            )

    for method in METHODS:
        values = [row["success"] for row in by_method[method]]
        mean, sd = fmean(values), stdev(values)
        x = x_positions[method]
        low, high = ymap(mean - sd, left), ymap(mean + sd, left)
        parts.extend(
            [
                (
                    f'<line x1="{x}" y1="{low:.1f}" x2="{x}" y2="{high:.1f}" '
                    'stroke="#111" stroke-width="2"/>'
                ),
                f'<line x1="{x - 7}" y1="{low:.1f}" x2="{x + 7}" y2="{low:.1f}" stroke="#111"/>',
                f'<line x1="{x - 7}" y1="{high:.1f}" x2="{x + 7}" y2="{high:.1f}" stroke="#111"/>',
                (
                    f'<rect x="{x - 4}" y="{ymap(mean, left) - 4:.1f}" '
                    'width="8" height="8" fill="#111"/>'
                ),
                _text(
                    x,
                    left[1] + left[3] + 22,
                    LABELS[method],
                    text_anchor="middle",
                    font_size="12",
                ),
            ]
        )

    rx, ry, rw, rh = right
    for tick in (0.0, 0.2, 0.4, 0.6):
        tx = rx + tick / 0.65 * rw
        parts.append(
            f'<line x1="{tx:.1f}" y1="{ry + rh}" x2="{tx:.1f}" y2="{ry + rh + 5}" stroke="#222"/>'
        )
        parts.append(
            _text(tx, ry + rh + 22, f"{100 * tick:.0f}", text_anchor="middle", font_size="12")
        )
    parts.append(
        _text(
            rx + rw / 2,
            ry + rh + 43,
            "Extra rollout tokens (%)",
            text_anchor="middle",
            font_size="13",
        )
    )

    label_offsets = {
        "grpo": (9, -9),
        "random_b2": (9, 17),
        "surprisal_b2": (-8, -13),
        "linear_ucb_b2": (-8, 18),
    }
    for method in METHODS:
        values = [row["success"] for row in by_method[method]]
        overhead = fmean(row["overhead"] for row in by_method[method])
        mean, sd = fmean(values), stdev(values)
        x = rx + overhead / 0.65 * rw
        y = ymap(mean, right)
        low, high = ymap(mean - sd, right), ymap(mean + sd, right)
        parts.extend(
            [
                (
                    f'<line x1="{x:.1f}" y1="{low:.1f}" x2="{x:.1f}" '
                    f'y2="{high:.1f}" stroke="{COLORS[method]}" stroke-width="2"/>'
                ),
                (
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" '
                    f'fill="{COLORS[method]}" stroke="white" stroke-width="1.5"/>'
                ),
                _text(
                    x + label_offsets[method][0],
                    y + label_offsets[method][1],
                    LABELS[method],
                    text_anchor="end" if label_offsets[method][0] < 0 else "start",
                    font_size="12",
                    fill=COLORS[method],
                    font_weight="bold",
                ),
            ]
        )

    parts.extend(
        [
            _text(
                490,
                438,
                "n=3 seeds; black/colored error bars show sample SD",
                text_anchor="middle",
                font_size="11",
                fill="#555",
            ),
            "</g>",
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render_svg(load_rows(args.csv), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
