#!/usr/bin/env python3
"""Export small, auditable histories without copying large raw episode traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from plot_training_curves import collect_metrics, write_csv


def export_history(root: Path, runs: list[str], output_dir: Path) -> None:
    root = root.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to mix exports in existing directory: {output_dir}")
    index = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
             "runs": []}
    for name in runs:
        run_dir = (root / name).resolve()
        if not run_dir.is_relative_to(root):
            raise ValueError(f"run is outside source root: {name}")
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        rows = collect_metrics(run_dir)
        expected = int(manifest["total_training_steps"])
        training_steps = {int(row["step"]) for row in rows if "training/global_step" in row}
        if training_steps != set(range(1, expected + 1)):
            raise ValueError(f"{name}: incomplete or unexpected training steps")
        final = next(row for row in rows if row["step"] == expected)
        if "val_success" not in final:
            raise ValueError(f"{name}: final validation is missing")
        destination = output_dir / name
        destination.mkdir(parents=True)
        write_csv(rows, destination / "metrics_by_step.csv")
        files = [(run_dir / "run_manifest.json", destination / "run_manifest.json")]
        files.extend((path, destination / "logs" / path.name)
                     for path in sorted((run_dir / "logs").glob("train-*-steps.log")))
        hashes = {}
        for source, target in files:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(source.relative_to(run_dir))] = hashlib.sha256(source.read_bytes()).hexdigest()
        index["runs"].append({
            "path": name, "method": manifest["method"], "seed": manifest["seed"],
            "metric_rows": len(rows), "training_steps": len(training_steps),
            "validation_steps": [int(row["step"]) for row in rows if "val_success" in row],
            "final_val_success": final["val_success"], "file_sha256": hashes,
        })
        print(f"{name}: {len(training_steps)} updates, final success={final['val_success']}")
    (output_dir / "export_index.json").write_text(json.dumps(index, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_history(args.root, args.run, args.output_dir)


if __name__ == "__main__":
    main()
