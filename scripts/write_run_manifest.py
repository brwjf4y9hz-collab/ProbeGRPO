#!/usr/bin/env python3
"""在公开 Sokoban 实验开始前记录固定运行元数据，便于追踪代码版本和配置。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
    parser.add_argument("--save-freq", type=int, default=-1)
    parser.add_argument("--hydra-overrides", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    data_manifest = json.loads((args.data_dir / "manifest.json").read_text())
    payload = {
        "method": args.method,
        "save_freq": args.save_freq,
        "hydra_overrides": args.hydra_overrides,
        "source_status": subprocess.check_output(
            ["git", "-C", str(args.project_dir), "status", "--porcelain"], text=True
        ).splitlines(),
        "framework_patch_sha256": hashlib.sha256(
            (args.project_dir / "patches/verl-probegrpo.patch").read_bytes()
        ).hexdigest(),
        "installed_trainer_sha256": hashlib.sha256(
            (args.verl_dir / "verl/trainer/ppo/v1/trainer_base.py").read_bytes()
        ).hexdigest(),
        "launch_script_sha256": hashlib.sha256(
            (args.project_dir / "scripts/run_public_sokoban_train.sh").read_bytes()
        ).hexdigest(),
        "packages": {
            item.metadata["Name"]: item.version
            for item in importlib.metadata.distributions()
            if item.metadata.get("Name", "").lower() in
            {"torch", "vllm", "transformers", "numpy", "ray", "datasets", "verl"}
        },
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
