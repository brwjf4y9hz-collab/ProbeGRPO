#!/usr/bin/env python3
"""Plot audited three-seed histories exported from the public Sokoban experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

METHODS = ("grpo", "random_b2", "surprisal_b2", "linear_ucb_b2")
LABELS = ("GRPO", "Random-B2", "Surprisal-B2", "LinearUCB-B2")
COLORS = ("#555555", "#E69F00", "#56B4E9", "#009E73")
SEEDS = (17, 42, 101)
METRICS = (
    "val_success", "val_episodes", "critic/score/mean", "actor/pg_loss",
    "actor/grad_norm", "critic/advantages/mean", "actor/entropy",
    "actor/lr", "response_length/mean", "invalid_action_rate",
    "probe/attempted", "probe/valid", "probe/failed", "probe/extra_tokens",
    "probe/changed_tokens", "probe/credit_abs_sum", "probe/abs_delta_mean",
    "probe/high_impact", "main_generated_tokens", "train_episodes",
    "timing_s/step", "cumulative_rollout_tokens", "cumulative_update_seconds",
)


def load_histories(root: Path, reference: Path) -> tuple[dict, dict]:
    index = json.loads((root / "export_index.json").read_text())
    histories = {}
    for run in index["runs"]:
        identity = (run["method"], int(run["seed"]))
        if identity in histories:
            raise ValueError(f"duplicate run: {identity}")
        with (root / run["path"] / "metrics_by_step.csv").open(newline="") as handle:
            histories[identity] = [
                {key: float(value) for key, value in row.items() if value}
                for row in csv.DictReader(handle)
            ]
    validate_reference(histories, reference)
    return histories, index


def validate_reference(histories: dict, reference: Path) -> None:
    if set(histories) != {(method, seed) for method in METHODS for seed in SEEDS}:
        raise ValueError("comparison requires the four methods and all three frozen seeds")
    for identity, rows in histories.items():
        steps = [int(row["step"]) for row in rows]
        if len(steps) != len(set(steps)) or set(steps) != set(range(51)):
            raise ValueError(f"missing or duplicate history steps: {identity}")
    with reference.open(newline="") as handle:
        for row in csv.DictReader(handle):
            identity = (row["method"], int(row["seed"]))
            final = next(item for item in histories[identity] if item["step"] == 50)
            if abs(final["val_success"] - float(row["final_val_success"])) > 1e-12:
                raise ValueError(f"final success disagrees with published reference: {identity}")
            if final.get("val_episodes") != int(row["eval_count"]):
                raise ValueError(f"final evaluation count disagrees: {identity}")
            tokens = sum(item.get("probe/extra_tokens", 0) for item in histories[identity])
            if tokens != int(row["probe_extra_tokens"]):
                raise ValueError(f"probe tokens disagree with published reference: {identity}")


def load_committed_history(path: Path, reference: Path) -> dict:
    histories = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            identity = (row.pop("method"), int(row.pop("seed")))
            histories[identity].append({key: float(value) for key, value in row.items() if value})
    validate_reference(histories, reference)
    return histories


def aggregate(histories: dict, method: str, metric: str, x_key: str = "step") -> list[dict]:
    by_step = defaultdict(list)
    for seed in SEEDS:
        for row in histories[(method, seed)]:
            if metric in row and x_key in row:
                by_step[int(row["step"])].append((row[x_key], row[metric]))
    result = []
    for step in sorted(by_step):
        values = by_step[step]
        if len(values) != len(SEEDS):
            continue
        result.append({"step": step, "x": fmean(x for x, _ in values),
                       "mean": fmean(y for _, y in values),
                       "sd": stdev(y for _, y in values), "n": len(values)})
    return result


def write_data(histories: dict, index: dict, root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_step.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", "step", *METRICS])
        writer.writeheader()
        for (method, seed), rows in sorted(histories.items()):
            for row in rows:
                writer.writerow({"method": method, "seed": seed, "step": int(row["step"]),
                                 **{key: row[key] for key in METRICS if key in row}})
    with (output_dir / "mean_sd_by_step.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "metric", "step", "mean", "sd", "n"])
        writer.writeheader()
        for method in METHODS:
            for metric in METRICS:
                for row in aggregate(histories, method, metric):
                    writer.writerow({"method": method, "metric": metric,
                                     **{key: row[key] for key in ("step", "mean", "sd", "n")}})
    provenance = {**index, "reference_checks": "12 final success/count/token totals match per_seed.csv",
                  "seed_band": "sample standard deviation, n=3; not a confidence interval",
                  "smoothing": "none", "metric_csv_sha256": {}}
    for run in index["runs"]:
        path = root / run["path"] / "metrics_by_step.csv"
        provenance["metric_csv_sha256"][run["path"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def plot_panel(ax, histories: dict, metric: str, title: str, ylabel: str,
               scale: float = 1, x_key: str = "step") -> None:
    for method, label, color in zip(METHODS, LABELS, COLORS):
        rows = aggregate(histories, method, metric, x_key)
        if not rows:
            continue
        x = [row["x"] / 1000 if x_key == "cumulative_rollout_tokens" else row["x"] for row in rows]
        mean = [row["mean"] * scale for row in rows]
        lower = [(row["mean"] - row["sd"]) * scale for row in rows]
        upper = [(row["mean"] + row["sd"]) * scale for row in rows]
        ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, mean, label=label, color=color, linewidth=1.7,
                marker="o" if metric == "val_success" else None, markersize=4)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Training rollout tokens (thousands)" if x_key != "step" else "Optimizer step")
    ax.grid(axis="both", color="#dddddd", linewidth=0.6, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if metric == "val_success":
        ax.set_ylim(0, 50)


def save_figure(fig, path: Path, title: str, note: str) -> None:
    fig.suptitle(title, y=0.99, fontsize=15, fontweight="bold")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=4, frameon=False)
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.065, 1, 0.9), h_pad=1.5, w_pad=1.5)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=300, bbox_inches="tight", facecolor="white")


def render_figures(histories: dict, figures_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    band_note = "Means across seeds 17, 42, 101; shading = sample SD (n=3); no smoothing."
    groups = (
        ("learning_curves", "ProbeGRPO: public Sokoban learning curves", (
            ("val_success", "A  Held-out validation", "Success (%)", 100),
            ("critic/score/mean", "B  Training reward", "Mean reward", 1),
            ("actor/pg_loss", "C  Actor policy loss", "Policy loss", 1),
            ("actor/grad_norm", "D  Actor gradient", "Gradient norm", 1),
        )),
        ("training_diagnostics", "ProbeGRPO: training diagnostics", (
            ("critic/advantages/mean", "A  Policy advantage after probe credit", "Mean advantage", 1),
            ("actor/entropy", "B  Policy entropy", "Entropy", 1),
            ("response_length/mean", "C  Trainer response length", "Response segment tokens", 1),
            ("invalid_action_rate", "D  Invalid environment actions", "Invalid actions (%)", 100),
        )),
        ("probe_cost_curves", "ProbeGRPO: probe activity and rollout cost", (
            ("probe/valid", "A  Valid probes", "Valid probes / update", 1),
            ("probe/extra_tokens", "B  Extra probing cost", "Probe tokens / update", 1),
            ("probe/credit_abs_sum", "C  Applied turn-local credit", "Sum of absolute token credit", 1),
            ("val_success", "D  Validation versus rollout cost", "Success (%)", 100),
        )),
    )
    for name, title, panels in groups:
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.3))
        for i, (ax, panel) in enumerate(zip(axes.flat, panels)):
            x_key = "cumulative_rollout_tokens" if name == "probe_cost_curves" and i == 3 else "step"
            plot_panel(ax, histories, *panel, x_key=x_key)
        note = band_note
        if name == "probe_cost_curves":
            note += "\nCost counts training main + probe generation only; evaluation generation is excluded."
        save_figure(fig, figures_dir / name, title, note)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    for ax, seed in zip(axes, SEEDS):
        for method, label, color in zip(METHODS, LABELS, COLORS):
            rows = [row for row in histories[(method, seed)] if "val_success" in row]
            ax.plot([row["step"] for row in rows], [row["val_success"] * 100 for row in rows],
                    color=color, label=label, marker="o", markersize=4)
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Validation success (%)")
        ax.set_ylim(0, 50)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, figures_dir / "per_seed_validation", "ProbeGRPO: each seed's held-out validation",
                "128 held-out boards per evaluation; actual evaluation points; no smoothing.")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_root", nargs="?", type=Path)
    parser.add_argument("--history-csv", type=Path, help="regenerate from committed per-step CSV")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="required when reading a raw export")
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.export_root) == bool(args.history_csv):
        parser.error("provide either export_root or --history-csv")
    if args.history_csv:
        histories = load_committed_history(args.history_csv, args.reference)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required for export_root")
        histories, index = load_histories(args.export_root, args.reference)
        write_data(histories, index, args.export_root, args.output_dir)
    render_figures(histories, args.figures_dir)
    print(f"Audited {len(histories)} runs against the published final results")
    print(f"Wrote four figure sets (PNG/SVG/PDF) to {args.figures_dir}")


if __name__ == "__main__":
    main()
