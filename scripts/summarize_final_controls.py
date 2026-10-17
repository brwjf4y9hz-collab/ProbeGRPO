#!/usr/bin/env python3
"""Report frozen 200-step paired outcomes and measured costs; refuse a partial main result."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev

from run_final_controls import PROTOCOL, load_protocol
from control_hardware import hardware_record, profile_hash
from write_run_manifest import sha256

PROJECT = Path(__file__).resolve().parents[1]


def summarize(root: Path, protocol: dict | None = None) -> dict:
    protocol = protocol if protocol is not None else load_protocol()
    records, missing, fingerprints, hardware_fingerprints = [], [], set(), set()
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            run = root / f"seed{seed}" / arm["method"]
            if not (run / "completion_receipt.json").is_file():
                missing.append(str(run))
                continue
            manifest = json.loads((run / "run_manifest.json").read_text())
            receipt = json.loads((run / "completion_receipt.json").read_text())
            if receipt["status"] != "complete" or receipt["manifest_sha256"] != sha256(run / "run_manifest.json"):
                raise ValueError(f"stale/invalid receipt: {run}")
            if (manifest["protocol"] != protocol["protocol"] or manifest["seed"] != seed
                    or manifest["total_training_steps"] != protocol["training_steps"]
                    or manifest["method"] != arm["method"]
                    or manifest["rollouts_per_prompt"] != arm["rollouts"]
                    or manifest["probe"]["budget"] != arm["budget"]
                    or manifest["probe"]["lambda_coef"] != arm["lambda_coef"]):
                raise ValueError(f"run does not match frozen main protocol: {run}")
            hardware = manifest.get("hardware_profile")
            if not hardware or not hardware.get("inventory"):
                raise ValueError(f"missing recorded hardware profile/inventory: {run}")
            if (hardware["sha256"] != profile_hash(hardware["settings"])
                    or hardware["gpu_signature"] != hardware_record(
                        hardware["settings"], hardware["inventory"])["gpu_signature"]):
                raise ValueError("hardware record signature mismatch")
            hardware_fingerprints.add((hardware["sha256"], json.dumps(hardware["gpu_signature"])))
            fingerprints.add(manifest["source_snapshot"]["sha256"])
            metrics_csv = run / "curves/metrics_by_step.csv"
            if receipt.get("metrics_csv_sha256") != sha256(metrics_csv):
                raise ValueError("metrics CSV changed after acceptance; re-audit the run")
            with metrics_csv.open() as handle:
                rows = list(csv.DictReader(handle))
                history = {int(row["step"]): row for row in rows}
            if len(rows) != len(history):
                raise ValueError("duplicate step records in metrics CSV")
            if set(history) != set(range(protocol["training_steps"] + 1)):
                raise ValueError(f"step history does not cover the full protocol: {run}")
            for step, row in history.items():
                if step:
                    for key in ("main_generated_tokens", "probe/extra_tokens", "timing_s/step"):
                        value = float(row[key])
                        if not math.isfinite(value) or value < 0:
                            raise ValueError(f"invalid measured {key} at step {step}")
                    if float(row["timing_s/step"]) <= 0:
                        raise ValueError("missing measured step time")
                if step % protocol.get("evaluation_interval", 10) == 0 or step == protocol["training_steps"]:
                    success = float(row["val_success"])
                    if not math.isfinite(success) or not 0 <= success <= 1:
                        raise ValueError("invalid validation success")
            initial = float(history[0]["val_success"])
            final = float(history[protocol["training_steps"]]["val_success"])
            main = sum(float(r.get("main_generated_tokens") or 0) for r in history.values())
            probe = sum(float(r.get("probe/extra_tokens") or 0) for r in history.values())
            if main <= 0:
                raise ValueError("missing measured main generated tokens")
            execution = run / "process_execution.json"
            checkpoint_times = [float(r["timing_s/save_checkpoint"]) for r in history.values()
                                if r.get("timing_s/save_checkpoint")]
            records.append({"seed": seed, "method": arm["method"],
                            "hardware_profile": hardware["name"], "gpu_count": hardware["settings"]["gpu_count"],
                            "failure_diagnostics": receipt.get("diagnostics", {}),
                            "initial_success": initial,
                            "final_success": final, "self_gain_pp": 100 * (final - initial),
                            "prespecified_validation_success": {
                                str(step): float(history[step]["val_success"])
                                for step in protocol.get("reporting_steps", [protocol["training_steps"]])},
                            "main_tokens": main, "probe_tokens": probe, "total_tokens": main + probe,
                            "logged_checkpoint_seconds": sum(checkpoint_times) if checkpoint_times else None,
                            "logged_step_seconds": sum(float(r.get("timing_s/step") or 0) for r in history.values()),
                            "launcher_wall_seconds": json.loads(execution.read_text())["elapsed_seconds"]
                            if execution.is_file() else None})
    if len(fingerprints) > 1:
        raise ValueError("mixed source snapshots: do not aggregate different executable protocols")
    if len(hardware_fingerprints) > 1:
        raise ValueError("mixed hardware profiles/devices: aggregate each execution setup separately")
    payload = {"protocol": protocol["protocol"], "training_steps": protocol["training_steps"],
               "status": "incomplete" if missing else "complete", "missing_runs": missing,
               "records": records, "cost_scope": "main + probe generated tokens, excludes validation/prefill; logged step time is not billed GPU hours"}
    if missing:
        return payload
    lookup = {(r["seed"], r["method"]): r for r in records}
    payload["aggregates"] = {}
    for arm in protocol["arms"]:
        values = [lookup[(s, arm["method"])] for s in protocol["seeds"]]
        paired = [100 * (v["final_success"] - lookup[(v["seed"], "grpo")]["final_success"]) for v in values]
        payload["aggregates"][arm["method"]] = {
            "final_success_mean": fmean(v["final_success"] for v in values),
            "final_success_sample_sd": stdev(v["final_success"] for v in values),
            "paired_vs_grpo_pp_mean": fmean(paired), "paired_vs_grpo_pp_sample_sd": stdev(paired)}
    payload["k6_vs_probegrpo_cost"] = []
    for seed in protocol["seeds"]:
        k6, probe = lookup[(seed, "grpo_k6")], lookup[(seed, "linear_ucb_b2")]
        ratio = k6["total_tokens"] / probe["total_tokens"]
        payload["k6_vs_probegrpo_cost"].append({
            "seed": seed, "generated_token_ratio": ratio,
            "approximately_token_matched_within_10_percent": abs(ratio - 1) <= 0.1,
            "logged_step_time_ratio": k6["logged_step_seconds"] / probe["logged_step_seconds"],
            "success_difference_k6_minus_probe_pp": 100 * (k6["final_success"] - probe["final_success"])})
    payload["causal_limit"] = "K changes GRPO estimation as well as cost; no automatic mechanism or scheduler-superiority claim"
    return payload


def markdown(payload: dict) -> str:
    lines = [f"# {payload['protocol']}: {payload['training_steps']} updates", "",
             f"Status: {payload['status']}. Costs are generated tokens / logged time, not billed GPU hours.", ""]
    if payload["status"] != "complete":
        lines += ["Missing or unaccepted runs (no main-result aggregate):", ""]
        lines += [f"- `{run}`" for run in payload["missing_runs"]]
        return "\n".join(lines) + "\n"
    lines += ["Three-seed descriptive means +/- sample SD; no significance claim.", "",
              "| Method | Final success (%) | Paired vs GRPO (pp) |",
              "|---|---:|---:|"]
    for method, row in payload["aggregates"].items():
        lines.append(f"| {method} | {100*row['final_success_mean']:.2f} +/- "
                     f"{100*row['final_success_sample_sd']:.2f} | "
                     f"{row['paired_vs_grpo_pp_mean']:+.2f} +/- {row['paired_vs_grpo_pp_sample_sd']:.2f} |")
    lines += ["", "K6 is not automatically equal-compute; compare actual per-seed token ratios:", "",
              "| Seed | K6 / Probe tokens | Within 10% | K6 - Probe success (pp) |",
              "|---|---:|---|---:|"]
    for row in payload["k6_vs_probegrpo_cost"]:
        lines.append(f"| {row['seed']} | {row['generated_token_ratio']:.3f} | "
                     f"{row['approximately_token_matched_within_10_percent']} | "
                     f"{row['success_difference_k6_minus_probe_pp']:+.2f} |")
    lines += ["", payload["causal_limit"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--protocol-file", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    payload = summarize(args.root, load_protocol(args.protocol_file))
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "control_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.root / "control_summary.md").write_text(markdown(payload))
    print(json.dumps(payload, indent=2))
    if payload["status"] != "complete":
        raise SystemExit("Incomplete matrix recorded; no main-result aggregate produced")


if __name__ == "__main__":
    main()
