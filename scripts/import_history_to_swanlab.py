#!/usr/bin/env python3
"""Import real historical metric CSVs; this does not run model training."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ALIASES = {
    "val_success": "eval/success_rate",
    "val_episodes": "eval/episodes",
    "train_success": "train/success_rate",
    "train_episodes": "train/episodes",
    "main_generated_tokens": "cost/main_generated_tokens",
    "cumulative_rollout_tokens": "cost/cumulative_training_rollout_tokens",
    "cumulative_update_seconds": "cost/cumulative_logged_update_seconds",
    "invalid_action_rate": "environment/invalid_action_rate",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_root", type=Path)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--project", default="probegrpo")
    parser.add_argument("--mode", choices=("offline", "online", "local"), default="offline")
    args = parser.parse_args()
    if args.log_dir.exists():
        raise FileExistsError(f"refusing duplicate historical import into {args.log_dir}")
    import swanlab

    settings = swanlab.Settings(
        interactive=False,
        probe=swanlab.Settings.Probe(
            hardware=False, monitor=False, runtime=False, requirements=False,
            git=False, conda=False,
        ),
    )
    index = json.loads((args.export_root / "export_index.json").read_text())
    imported = []
    for source in index["runs"]:
        run_dir = args.export_root / source["path"]
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        name = f"historical-public-sokoban-{source['method']}-seed{source['seed']}"
        run = swanlab.init(
            project=args.project, experiment_name=name,
            group="public_sokoban_main_v1", job_type="historical_import",
            tags=["historical-import", source["method"], f"seed-{source['seed']}"],
            description="Imported original training metrics; no new model training was performed.",
            config={**manifest, "historical_import": True, "source_run": source["path"],
                    "source_log_sha256": source["file_sha256"],
                    "cost_definition": "Training main + probe tokens, excluding evaluation"},
            logdir=str(args.log_dir), mode=args.mode, settings=settings,
        )
        count = 0
        try:
            with (run_dir / "metrics_by_step.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    data = {ALIASES.get(key, key): float(value)
                            for key, value in row.items() if key != "step" and value}
                    data = {key: value for key, value in data.items() if math.isfinite(value)}
                    run.log(data, step=int(float(row["step"])))
                    count += 1
        finally:
            swanlab.finish()
        imported.append({"experiment_name": name, "rows": count,
                         "seed": source["seed"], "method": source["method"]})
        print(f"Imported {name}: {count} metric steps")
    (args.log_dir / "imported_runs.json").write_text(
        json.dumps({"swanlab_version": swanlab.__version__, "mode": args.mode,
                    "project": args.project, "runs": imported}, indent=2) + "\n"
    )
    print(f"Imported {len(imported)} historical runs to {args.log_dir} ({args.mode})")


if __name__ == "__main__":
    main()
