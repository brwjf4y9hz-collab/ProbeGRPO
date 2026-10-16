#!/usr/bin/env python3
"""Read cloud histories back and compare them with the exported source CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from import_history_to_swanlab import ALIASES

KEYS = (
    "training/global_step", "actor/pg_loss", "eval/success_rate",
    "cost/cumulative_training_rollout_tokens", "probe/extra_tokens",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_root", type=Path)
    parser.add_argument("--project", default="probegrpo")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import swanlab

    api = swanlab.Api()
    project = api.project(f"{api.username}/{args.project}")
    cloud_runs = {}
    for run in project.runs_get(all=True):
        if run.name in cloud_runs:
            raise ValueError(f"duplicate experiment name in cloud: {run.name}")
        cloud_runs[run.name] = run
    index = json.loads((args.export_root / "export_index.json").read_text())
    receipt = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "swanlab_version": swanlab.__version__,
        "project_url": project.url, "workspace": api.username,
        "project_experiment_count": project.count.get("experiments"),
        "checked_metrics": list(KEYS), "runs": [],
    }
    for source in index["runs"]:
        name = f"historical-public-sokoban-{source['method']}-seed{source['seed']}"
        run = cloud_runs[name]
        if run.state != "FINISHED":
            raise ValueError(f"cloud experiment is not finished: {name}: {run.state}")
        path = args.export_root / source["path"] / "metrics_by_step.csv"
        expected = {key: {} for key in KEYS}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for key, value in row.items():
                    cloud_key = ALIASES.get(key, key)
                    if cloud_key in expected and value:
                        expected[cloud_key][int(float(row["step"]))] = float(value)
        result = run.metrics(keys=list(KEYS), all=True, ignore_timestamp=True)
        actual = {
            item["key"]: {int(point["step"]): float(point["value"])
                          for point in item.get("metrics", [])}
            for item in result.get("list", [])
        }
        for key in KEYS:
            if set(actual.get(key, {})) != set(expected[key]):
                raise ValueError(f"missing/unexpected cloud metric steps: {name}: {key}")
            for step, value in expected[key].items():
                if not math.isclose(actual[key][step], value, rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError(f"cloud value mismatch: {name}: {key} at step {step}")
        counts = {key: len(expected[key]) for key in KEYS}
        receipt["runs"].append({
            "experiment_name": name, "method": source["method"], "seed": source["seed"],
            "url": run.url, "state": run.state, "metric_point_counts": counts,
            "final_val_success": actual["eval/success_rate"][50],
            "metric_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cloud_values_match_source": True,
        })
        print(f"Verified {name}: {counts['training/global_step']} updates, "
              f"{counts['eval/success_rate']} evaluations", flush=True)
    receipt["verified_runs"] = len(receipt["runs"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified {len(receipt['runs'])} runs. Project: {project.url}")


if __name__ == "__main__":
    main()
