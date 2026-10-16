#!/usr/bin/env python3
"""Plan frozen 200-step controls by default; execute only explicitly on a validated GPU host."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from check_run_complete import check as check_complete
from control_hardware import PROFILE_NAMES, device_environment, hardware_record, load_profile
from write_run_manifest import sha256, source_snapshot

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT / "experiments/protocols/public_sokoban_controls_v3_200.json"


def load_protocol(path: Path = PROTOCOL) -> dict:
    protocol = json.loads(path.read_text())
    steps = protocol["training_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("training_steps must be a positive integer")
    for key in ("evaluation_interval", "checkpoint_interval", "max_actor_checkpoints"):
        value = protocol.get(key, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return protocol


def build_plan(protocol: dict, phase: str, root: Path) -> list[dict]:
    if phase not in {"smoke", "confirm", "controls", "main"}:
        raise ValueError(f"unsupported phase: {phase}")
    rows = []
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            confirmation = arm["method"] == "linear_ucb_b2" or (
                arm["method"] == "grpo" and seed == 17)
            if phase == "confirm" and not confirmation:
                continue
            if phase == "controls" and confirmation:
                continue
            if phase == "smoke" and seed != 17:
                continue
            steps = 1 if phase == "smoke" else protocol["training_steps"]
            folder = root / "smoke" if phase == "smoke" else root
            rows.append({**arm, "seed": seed, "steps": steps,
                         "output_dir": str(folder / f"seed{seed}" / arm["method"])})
    return rows


def launch_settings(run: dict, protocol: dict, model_path: str, verl: Path,
                    profile: dict | None = None, runtime_env: dict | None = None,
                    inventory: dict | None = None) -> tuple[dict, list[str]]:
    """One source of truth for the actual launcher env and Hydra cadence, not only the plan."""
    interval = min(protocol.get("checkpoint_interval", run["steps"]), run["steps"])
    prefix = protocol.get("experiment_prefix", "controls-v2")
    profile = profile if profile is not None else load_profile()
    env = {**(runtime_env if runtime_env is not None else os.environ),
           "METHOD": run["method"], "SEED": str(run["seed"]),
           "PROBE_BUDGET": str(run["budget"]), "PROBE_SCHEDULER": run["scheduler"],
           "PROBE_LAMBDA": str(run["lambda_coef"]), "ROLLOUTS_PER_PROMPT": str(run["rollouts"]),
           "TOTAL_TRAINING_STEPS": str(run["steps"]), "SAVE_FREQ": str(interval),
           "TRAIN_LIMIT": str(protocol["frozen"]["train_boards"]),
           "TEST_LIMIT": str(protocol["validation_episodes"]),
           "MAX_TURNS": str(protocol["frozen"]["max_turns"]),
           "MODEL_PATH": model_path, "MODEL_REVISION": protocol["frozen"]["model_revision"],
           "OUTPUT_DIR": run["output_dir"], "RUN_PROTOCOL": protocol["protocol"],
           "PROBEGRPO_HARDWARE_JSON": json.dumps(hardware_record(profile, inventory)),
           "DATA_DIR": str(verl / "data/probegrpo-ragen-public-sokoban-v1")}
    overrides = [
        f"trainer.experiment_name={prefix}-{profile['name']}-{run['method']}-seed{run['seed']}-steps{run['steps']}",
        f"trainer.test_freq={protocol.get('evaluation_interval', 10)}",
        f"trainer.max_actor_ckpt_to_keep={protocol.get('max_actor_checkpoints', 1)}",
    ]
    overrides.extend(f"{key}={json.dumps(value)}" for key, value in profile["hydra_overrides"].items())
    if env.get("SWANLAB_ENABLED") == "1":
        overrides.append('trainer.logger=["console","swanlab"]')
    return env, overrides


def require_smoke(root: Path, protocol: dict, hardware: dict, fingerprint: str) -> None:
    """All four one-update gates must belong to this code/profile before formal training."""
    for run in build_plan(protocol, "smoke", root):
        output = Path(run["output_dir"])
        receipt = output / "completion_receipt.json"
        if not receipt.is_file():
            raise ValueError(f"run all four --phase smoke gates first: {output}")
        saved = json.loads((output / "run_manifest.json").read_text())
        recorded = json.loads(receipt.read_text())
        previous = saved.get("hardware_profile") or {}
        if (recorded.get("manifest_sha256") != sha256(output / "run_manifest.json")
                or saved["protocol"] != protocol["protocol"]
                or saved["method"] != run["method"] or saved["seed"] != 17
                or saved["total_training_steps"] != 1
                or saved["rollouts_per_prompt"] != run["rollouts"]
                or saved["probe"]["budget"] != run["budget"]
                or saved["probe"]["lambda_coef"] != run["lambda_coef"]
                or saved["source_snapshot"]["sha256"] != fingerprint
                or previous.get("sha256") != hardware["sha256"]
                or previous.get("gpu_signature") != hardware["gpu_signature"]):
            raise ValueError("smoke source/protocol/hardware mismatch; use a fresh output root")
        check_complete(output)


def preflight(verl: Path, protocol: dict) -> dict:
    """No torch/vLLM import, network request, patch, or training on a 2 GB no-card host."""
    if not (verl / ".venv/bin/python").is_file():
        raise ValueError("missing fixed verl environment")
    revision = subprocess.check_output(["git", "-C", str(verl), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != protocol["frozen"]["verl_revision"]:
        raise ValueError("unexpected verl revision")
    data_dir = verl / "data/probegrpo-ragen-public-sokoban-v1"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["source_revision"] != protocol["frozen"]["dataset_revision"]:
        raise ValueError("unexpected dataset revision")
    if manifest["max_turns"] != protocol["frozen"]["max_turns"]:
        raise ValueError("unexpected dataset horizon")
    for split, rows in (("train", 512), ("test", 128)):
        metadata = manifest["derived_splits"][split]
        if metadata["rows"] != rows or sha256(data_dir / f"{split}.parquet") != metadata["sha256"]:
            raise ValueError(f"wrong derived data: {split}")
    hf_root = Path(os.environ.get("HF_HOME", str(verl / "data/huggingface")))
    model = hf_root / "hub/models--Qwen--Qwen3.5-2B/snapshots" / protocol["frozen"]["model_revision"]
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("unexpected model config")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if not (model / shard).is_file():
            raise ValueError(f"missing model shard: {shard}")
    installed = subprocess.check_output(
        [str(verl / ".venv/bin/python"), "-c", "import probegrpo; print(probegrpo.__file__)"],
        text=True).strip()
    if Path(installed).resolve() != (PROJECT / "src/probegrpo/__init__.py").resolve():
        raise ValueError(f"editable ProbeGRPO points elsewhere: {installed}")
    return {"data_checksums": "passed", "model_shards": "present (not loaded)", "model_path": str(model),
            "editable_source": installed, "free_disk_gib": round(shutil.disk_usage(verl).free / 2**30, 2),
            "source_sha256": source_snapshot(PROJECT)["sha256"], "gpu_training": "not started"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-dir", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, default=PROTOCOL)
    parser.add_argument("--output-root", type=Path, help="base directory; profile name is appended")
    parser.add_argument("--hardware-profile", choices=PROFILE_NAMES, default="single_48gb")
    parser.add_argument("--gpu-devices", help="only your assigned indices/UUIDs; never overrides scheduler visibility")
    parser.add_argument("--allow-experimental-profile", action="store_true")
    parser.add_argument("--phase", choices=("smoke", "confirm", "controls", "main"), default="main")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.verl_dir = args.verl_dir.resolve()
    args.protocol_file = args.protocol_file.resolve()
    protocol = load_protocol(args.protocol_file)
    profile = load_profile(args.hardware_profile)
    runtime_env = device_environment(os.environ, args.gpu_devices)
    args.output_root = (args.output_root or PROJECT / "outputs" /
                        protocol.get("output_subdir", "final-controls-v2")).resolve() / profile["name"]
    plan = build_plan(protocol, args.phase, args.output_root)
    print(json.dumps({"protocol": protocol["protocol"], "mode": "execute" if args.execute else "no-training",
                      "evaluation_interval": protocol.get("evaluation_interval", 10),
                      "checkpoint_interval": protocol.get("checkpoint_interval", protocol["training_steps"]),
                      "hardware_profile": hardware_record(profile),
                      "total_optimizer_updates": sum(run["steps"] for run in plan),
                      "runs": plan}, indent=2), flush=True)
    if args.preflight or args.execute:
        checked = preflight(args.verl_dir, protocol)
        print(json.dumps(checked, indent=2), flush=True)
    if not args.execute:
        return
    if profile["experimental"] and not args.allow_experimental_profile:
        raise ValueError("24GB profiles are unvalidated candidates; explicitly use --allow-experimental-profile")
    # This is a paid-training switch, never reached in default/no-card preflight mode.
    inventory = json.loads(subprocess.check_output(
        [str(args.verl_dir / ".venv/bin/python"), str(PROJECT / "scripts/control_hardware.py"),
         "--profile", profile["name"]], env=runtime_env, text=True))
    hardware = hardware_record(profile, inventory)
    fingerprint = source_snapshot(PROJECT)["sha256"]
    if args.phase != "smoke":
        require_smoke(args.output_root, protocol, hardware, fingerprint)
    for run in plan:
        output = Path(run["output_dir"])
        receipt = output / "completion_receipt.json"
        if output.exists():
            if not receipt.is_file():
                raise ValueError(f"partial/unverified run preserved; choose a NEW output root: {output}")
            saved = json.loads((output / "run_manifest.json").read_text())
            if saved["source_snapshot"]["sha256"] != fingerprint:
                raise ValueError("source changed since previous run; use a new output root")
            previous_hardware = saved.get("hardware_profile") or {}
            if (previous_hardware.get("sha256") != hardware["sha256"]
                    or previous_hardware.get("gpu_signature") != hardware["gpu_signature"]):
                raise ValueError("saved run hardware differs; do not mix execution profiles")
            if (saved["protocol"] != protocol["protocol"] or saved["seed"] != run["seed"]
                    or saved["method"] != run["method"] or saved["total_training_steps"] != run["steps"]
                    or saved["rollouts_per_prompt"] != run["rollouts"]
                    or saved["probe"]["budget"] != run["budget"]
                    or saved["probe"]["lambda_coef"] != run["lambda_coef"]):
                raise ValueError("saved run identity/config does not match the requested arm")
            subprocess.run([str(args.verl_dir / ".venv/bin/python"),
                            str(PROJECT / "scripts/check_run_complete.py"), str(output)], check=True)
            print(f"Verified completed run; skipping {output}", flush=True)
            continue
        if source_snapshot(PROJECT)["sha256"] != fingerprint:
            raise ValueError("source changed during experiment matrix")
        if shutil.disk_usage(args.verl_dir).free < 3 * 2**30:
            raise ValueError("less than 3 GiB disk free; preserve results and free space before training")
        env, overrides = launch_settings(run, protocol, checked["model_path"], args.verl_dir,
                                         profile, runtime_env, inventory)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        process = subprocess.run(["bash", str(PROJECT / "scripts/run_public_sokoban_train.sh"),
                                  str(args.verl_dir), *overrides], env=env, check=False)
        if output.is_dir():
            (output / "process_execution.json").write_text(json.dumps({
                "started_at_utc": started_at, "elapsed_seconds": time.monotonic() - started,
                "exit_code": process.returncode,
                "scope": "launcher wall time including setup/model initialization/validation/save/export; not billed GPU time",
            }, indent=2) + "\n")
        process.check_returncode()
        subprocess.run([str(args.verl_dir / ".venv/bin/python"),
                        str(PROJECT / "scripts/check_run_complete.py"), str(output), "--write-receipt"], check=True)
    if args.phase == "main":
        subprocess.run([str(args.verl_dir / ".venv/bin/python"),
                        str(PROJECT / "scripts/summarize_final_controls.py"), str(args.output_root),
                        "--protocol-file", str(args.protocol_file)], check=True)


if __name__ == "__main__":
    main()
