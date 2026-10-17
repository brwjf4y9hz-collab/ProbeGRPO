#!/usr/bin/env python3
"""Reject partial runs: inspect every update/evaluation, credit metrics and final checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from plot_training_curves import read_log_metrics
from control_hardware import profile_hash
from write_run_manifest import sha256


def check(run: Path) -> dict:
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    final = int(manifest["total_training_steps"])
    k = int(manifest["rollouts_per_prompt"])
    train_count = int(manifest["train_batch_size"]) * k
    test_count = int(manifest["data"]["derived_splits"]["test"]["rows"])
    overrides = dict(v.lstrip("+").split("=", 1) for v in manifest["hydra_overrides"])
    hardware = manifest.get("hardware_profile")
    if hardware:
        if hardware["sha256"] != profile_hash(hardware["settings"]):
            raise ValueError("hardware profile hash mismatch")
        for key, expected in hardware["settings"]["hydra_overrides"].items():
            if key not in overrides or json.loads(overrides[key]) != expected:
                raise ValueError(f"resolved launcher override disagrees with hardware profile: {key}")
    interval = int(overrides.get("trainer.test_freq", 10))
    if interval <= 0:
        raise ValueError("final-control runs require validation")
    validation_steps = {0, final, *range(interval, final + 1, interval)}
    counts = {}
    diagnostics = {split: {"episodes": 0, "successes": 0, "stop_reasons": Counter(),
                           "invalid_action_types": Counter(), "probe_outcomes": Counter()}
                   for split in ("train", "test_all_evaluations", "final_test")}
    for step in range(final + 1):
        episodes = [json.loads(p.read_text()) for p in
                    (run / "rollouts/episodes" / f"step-{step}").glob("*.json")]
        training = [e for e in episodes if e.get("dataset_split") == "train"]
        validation = [e for e in episodes if e.get("dataset_split") == "test"]
        expected = (0 if step == 0 else train_count,
                    test_count if step in validation_steps else 0)
        if (len(training), len(validation)) != expected:
            raise ValueError(f"step {step}: train/test counts {(len(training), len(validation))} != {expected}")
        for episode in episodes:
            if episode.get("dataset_split") not in {"train", "test"}:
                raise ValueError("unexpected dataset split in episode sidecar")
            if not math.isfinite(float(episode["final_reward"])) or float(episode["final_reward"]) not in (0, 1):
                raise ValueError("public Sokoban requires finite binary episode rewards")
            groups = ["train" if episode.get("dataset_split") == "train" else "test_all_evaluations"]
            if episode.get("dataset_split") == "test" and step == final:
                groups.append("final_test")
            for group in groups:
                values = diagnostics[group]
                values["episodes"] += 1
                values["successes"] += int(float(episode["final_reward"]) == 1)
                values["stop_reasons"][episode.get("stop_reason", "not_recorded")] += 1
                for turn in episode.get("turns", []):
                    if not turn.get("action_valid", False):
                        category = "malformed_model_action" if turn.get("action") == "invalid" else "blocked_or_illegal_move"
                        values["invalid_action_types"][category] += 1
                probe = episode.get("debug_probe")
                if probe:
                    reason = probe.get("skipped_reason")
                    label = str(reason).split(":", 1)[0] if reason else (
                        "valid" if probe.get("state_match") else "invalid_without_reason")
                    values["probe_outcomes"][label] += 1
        if any(e.get("debug_probe") for e in validation):
            raise ValueError("evaluation must not run probes")
        counts[str(step)] = {"train": len(training), "test": len(validation)}
    metrics = read_log_metrics(run)
    for step in range(1, final + 1):
        values = metrics.get(step, {})
        for key in ("actor/grad_norm", "actor/pg_loss", "training/global_step",
                    "timing_s/step", "probe/changed_tokens", "probe/credit_abs_sum"):
            if key not in values:
                raise ValueError(f"missing/non-finite {key} at update {step}")
        if values["timing_s/step"] <= 0:
            raise ValueError("step time must be positive")
        if manifest["probe"]["lambda_coef"] == 0 or manifest["probe"]["budget"] == 0:
            for key in ("probe/changed_tokens", "probe/credit_abs_sum"):
                if key not in values or values[key] != 0:
                    raise ValueError(f"control must have zero {key} at update {step}")
    actor = run / "checkpoints" / f"global_step_{final}" / "actor"
    world = hardware["settings"]["gpu_count"] if hardware else int(overrides.get("trainer.n_gpus_per_node", 1))
    weights = [actor / f"model_world_size_{world}_rank_{rank}.pt" for rank in range(world)]
    if any(not p.is_file() or p.stat().st_size == 0 for p in weights):
        raise ValueError("missing final actor model weights for one or more ranks")
    if not (actor.parent / "data.pt").is_file():
        raise ValueError("missing final dataloader checkpoint")
    metrics_csv = run / "curves/metrics_by_step.csv"
    if not metrics_csv.is_file():
        raise ValueError("missing exported metrics CSV")
    return {"status": "complete", "manifest_sha256": sha256(manifest_path),
            "metrics_csv_sha256": sha256(metrics_csv),
            "diagnostics": diagnostics,
            "final_step": final, "episode_counts": counts,
            "checkpoint_weights": [str(p.relative_to(run)) for p in weights],
            "note": "structural/finite-metric acceptance, not a performance or resume-validation claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    receipt = check(args.run_dir)
    if args.write_receipt:
        (args.run_dir / "completion_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
