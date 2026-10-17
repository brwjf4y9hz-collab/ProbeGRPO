#!/usr/bin/env python3
"""在公开 Sokoban 实验前记录实际参数、源码指纹、数据校验与运行环境。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot(root: Path) -> dict:
    """Hash executable project inputs, including uncommitted/untracked source, not secrets/logs."""
    paths = [p for folder in ("src", "scripts", "configs", "experiments/protocols", "patches")
             for p in (root / folder).rglob("*")
             if p.is_file() and p.suffix in {".py", ".sh", ".yaml", ".yml", ".json", ".patch"}]
    paths += [root / "pyproject.toml"] if (root / "pyproject.toml").is_file() else []
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}
    return {"files": files, "sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest()}


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
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--protocol", default="custom")
    parser.add_argument("--hydra-overrides", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    data_manifest = json.loads((args.data_dir / "manifest.json").read_text())
    for split, metadata in data_manifest["derived_splits"].items():
        if sha256(args.data_dir / f"{split}.parquet") != metadata["sha256"]:
            raise ValueError(f"derived dataset checksum mismatch: {split}")
    overrides = dict(item.lstrip("+").split("=", 1) for item in args.hydra_overrides)
    payload = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": args.protocol,
        "hardware_profile": json.loads(os.environ.get("PROBEGRPO_HARDWARE_JSON", "null")),
        "method": args.method,
        "save_freq": int(overrides.get("trainer.save_freq", args.save_freq)),
        "rollouts_per_prompt": int(overrides.get(
            "actor_rollout_ref.rollout.n", args.rollouts_per_prompt)),
        "train_batch_size": int(overrides.get("data.train_batch_size", 4)),
        "hydra_overrides": args.hydra_overrides,
        "source_status": subprocess.check_output(
            ["git", "-C", str(args.project_dir), "status", "--porcelain"], text=True
        ).splitlines(),
        "source_snapshot": source_snapshot(args.project_dir),
        "framework_patch_sha256": sha256(args.project_dir / "patches/verl-probegrpo.patch")
        if (args.project_dir / "patches/verl-probegrpo.patch").is_file() else None,
        "entropy_patch_sha256": sha256(args.project_dir / "patches/verl-dense-entropy.patch")
        if (args.project_dir / "patches/verl-dense-entropy.patch").is_file() else None,
        "installed_entropy_engine_sha256": sha256(
            args.verl_dir / "verl/workers/engine/fsdp/transformer_impl.py")
        if (args.verl_dir / "verl/workers/engine/fsdp/transformer_impl.py").is_file() else None,
        "launch_script_sha256": sha256(args.project_dir / "scripts/run_public_sokoban_train.sh"),
        "installed_trainer_sha256": sha256(
            args.verl_dir / "verl/trainer/ppo/v1/trainer_base.py"),
        "packages": {
            item.metadata["Name"]: item.version
            for item in importlib.metadata.distributions()
            if item.metadata.get("Name", "").lower() in
            {"torch", "vllm", "transformers", "numpy", "ray", "datasets", "verl", "swanlab", "peft"}
        },
        "seed": args.seed,
        "probe": {
            "budget": int(overrides.get("probe.budget", args.budget)),
            "scheduler": overrides.get("probe.scheduler", args.scheduler),
            "lambda_coef": float(overrides.get("probe.lambda_coef", args.lambda_coef)),
        },
        "total_training_steps": int(overrides.get("trainer.total_training_steps", args.steps)),
        "model": overrides.get("actor_rollout_ref.model.path", args.model),
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
