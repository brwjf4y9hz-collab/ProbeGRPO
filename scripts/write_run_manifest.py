#!/usr/bin/env python3
"""Write immutable run metadata before a public Sokoban experiment starts."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path


def revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--verl-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--lambda-coef", required=True, type=float)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--steps", required=True, type=int)
    args = parser.parse_args()
    data_manifest = json.loads((args.data_dir / "manifest.json").read_text())
    payload = {
        "method": args.method,
        "seed": args.seed,
        "probe": {
            "budget": args.budget,
            "scheduler": args.scheduler,
            "lambda_coef": args.lambda_coef,
        },
        "total_training_steps": args.steps,
        "model": args.model,
        "model_revision": args.model_revision,
        "probegrpo_revision": revision(args.project_dir),
        "verl_revision": revision(args.verl_dir),
        "data": data_manifest,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
