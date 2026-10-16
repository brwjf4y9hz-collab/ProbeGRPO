#!/usr/bin/env python3
"""Audit real saved anchors and their eligible-position opportunities, without model imports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean


def bucket(fraction):
    return min(2, int(max(0, min(1, fraction)) * 3))


def audit(run: Path):
    manifest = json.loads((run / "run_manifest.json").read_text())
    rows, digest = [], hashlib.sha256()
    for path in sorted((run / "rollouts/episodes").glob("step-*/*.json")):
        raw = path.read_bytes()
        digest.update(str(path.relative_to(run)).encode() + b"\0" + hashlib.sha256(raw).digest())
        episode = json.loads(raw)
        if episode.get("dataset_split") != "train":
            continue
        probe = episode.get("debug_probe") or {}
        anchor = probe.get("scheduler_turn")
        if not anchor:
            continue  # No eligible anchor is not an actual UCB-ranking selection.
        label = probe.get("scheduler", anchor.get("scheduler_label", "unknown"))
        if not label.startswith("linear_ucb"):
            continue
        index = int(anchor["turn_id"])
        length = len(episode["turns"])
        horizon = int(episode["max_horizon"])
        opportunities = {bucket(i / max(1, horizon))
                         for i, t in enumerate(episode["turns"])
                         if t["action"] in t["legal_actions"] and len(set(t["legal_actions"])) >= 2}
        rows.append({
            "seed": manifest["seed"], "step": int(path.parent.name.removeprefix("step-")),
            "scheduler": label, "turn_id": index, "trajectory_turns": length,
            "max_horizon": horizon, "horizon_bucket": bucket(index / max(1, horizon)),
            "observed_bucket": bucket(index / max(1, length - 1)),
            "eligible_horizon_buckets": ",".join(map(str, sorted(opportunities))),
            "valid": probe.get("skipped_reason") is None and bool(probe.get("state_match")),
            "extra_tokens": int(probe.get("additional_rollout_tokens", 0)),
            "source": str(path.relative_to(run)), "source_sha256": hashlib.sha256(raw).hexdigest(),
        })
    warmup = [r for r in rows if r["scheduler"] == "linear_ucb_warmup"]
    ranked = [r for r in rows if r["scheduler"] == "linear_ucb"]
    summary = {
        "seed": manifest["seed"], "run": str(run), "sidecars_sha256": digest.hexdigest(),
        "anchored_probes": len(rows), "warmup_anchors": len(warmup),
        "warmup_horizon_buckets": dict(sorted(Counter(r["horizon_bucket"] for r in warmup).items())),
        "warmup_observed_buckets": dict(sorted(Counter(r["observed_bucket"] for r in warmup).items())),
        "warmup_eligible_bucket_sets": dict(sorted(Counter(r["eligible_horizon_buckets"] for r in warmup).items())),
        "warmup_last_anchor_step": max((r["step"] for r in warmup), default=None),
        "first_ranked_anchor_step": min((r["step"] for r in ranked), default=None),
        "warmup_mean_turn_id": fmean(r["turn_id"] for r in warmup) if warmup else None,
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows, summaries = [], []
    for run in args.run_dirs:
        selected, summary = audit(run)
        rows.extend(selected)
        summaries.append(summary)
    if not rows:
        raise ValueError("no actual saved UCB anchors found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "anchors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"runs": summaries, "warmup_anchors": sum(s["warmup_anchors"] for s in summaries),
               "warmup_horizon_buckets": dict(sorted(Counter(
                   r["horizon_bucket"] for r in rows if r["scheduler"] == "linear_ucb_warmup").items())),
               "scope": "saved v1 episodes; not corrected-v2 training; both valid and failed selected probes counted"}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
