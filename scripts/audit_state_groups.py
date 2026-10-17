#!/usr/bin/env python3
"""Offline state-reuse diagnostics, NOT a GiGPO implementation or performance result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

MODES = ("full_state", "observation", "horizon_observation", "history")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def summarize(groups):
    total = sum(len(values) for values in groups.values())
    repeated = cross = varied = useful = later_cross = 0
    for values in groups.values():
        multi = len(values) > 1
        across = len({v["trajectory"] for v in values}) > 1
        different_return = len({v["return"] for v in values}) > 1
        different_action = len({v["action"] for v in values}) > 1
        repeated += len(values) * multi
        cross += len(values) * across
        varied += len(values) * (multi and different_return)
        useful += len(values) * (across and different_return and different_action)
        later_cross += sum(v["turn"] > 0 for v in values) * across
    later = sum(v["turn"] > 0 for values in groups.values() for v in values)
    return {
        "groups": len(groups), "occurrences": total,
        "group_size_histogram": dict(sorted(Counter(len(v) for v in groups.values()).items())),
        "repeated_occurrence_fraction": repeated / total if total else None,
        "cross_trajectory_occurrence_fraction": cross / total if total else None,
        "heterogeneous_return_occurrence_fraction": varied / total if total else None,
        "cross_trajectory_diverse_action_return_fraction": useful / total if total else None,
        "noninitial_cross_trajectory_fraction": later_cross / later if later else None,
    }


def audit(run: Path, *, rollouts=4, prompts=4):
    if min(rollouts, prompts) <= 0:
        raise ValueError("rollouts and prompts must be positive")
    manifest = json.loads((run / "run_manifest.json").read_text())
    paths = sorted((run / "rollouts/episodes").glob("step-*/*.json"))
    if not paths:
        raise ValueError("missing full episode sidecars; logs/selected cases cannot establish state coverage")
    groups = {mode: defaultdict(list) for mode in MODES}
    batches, sources = defaultdict(list), []
    for path in paths:
        raw = path.read_bytes()
        episode = json.loads(raw)
        if episode.get("dataset_split") != "train":
            continue  # Never combine validation trajectories or separate policy updates.
        step = int(path.parent.name.removeprefix("step-"))
        trajectory = episode["trajectory_id"]
        uid, separator, session = trajectory.rpartition("_")
        if not separator or not uid or not session.isdigit():
            raise ValueError("trajectory_id must preserve the original uid_session grouping")
        batches[step].append((uid, int(session), episode["task_id"], episode["seed"]))
        sources.append({"path": str(path.relative_to(run)), "sha256": hashlib.sha256(raw).hexdigest()})
        namespace = (step, uid, episode["task_id"], episode["seed"])
        turns = episode["turns"]
        returns, value = [], 0.0
        for turn in reversed(turns):
            reward = float(turn["reward"])
            if not math.isfinite(reward) or reward not in (0, 1):
                raise ValueError("expected finite binary Sokoban step rewards")
            value += reward  # Explicit gamma=1 diagnostic, not an assumed official gamma.
            returns.append(value)
        final = float(episode["final_reward"])
        if not math.isfinite(final) or final not in (0, 1) or value != final:
            raise ValueError("step returns do not match terminal reward")
        returns.reverse()
        history = []
        for index, turn in enumerate(turns):
            history.append(["observation", turn["observation"]])
            keys = {
                "full_state": turn["state_hash"],
                "observation": turn["observation"],
                "horizon_observation": [turn["observation"], episode["max_horizon"] - index],
                "history": history,
            }
            item = {"trajectory": trajectory, "turn": index, "action": turn["action"],
                    "return": returns[index]}
            for mode, key in keys.items():
                groups[mode][(*namespace, fingerprint(key))].append(item)
            history.append(["assistant", turn["raw_text"]])
    expected_steps = set(range(1, int(manifest["total_training_steps"]) + 1))
    if set(batches) != expected_steps:
        raise ValueError("incomplete training step coverage; no aggregate emitted")
    for step, batch in batches.items():
        by_uid = defaultdict(list)
        for uid, session, task, seed in batch:
            by_uid[uid].append((session, task, seed))
        if len(by_uid) != prompts:
            raise ValueError(f"step {step}: prompt count mismatch")
        for entries in by_uid.values():
            if (len(entries) != rollouts or sorted(e[0] for e in entries) != list(range(rollouts))
                    or len({(e[1], e[2]) for e in entries}) != 1):
                raise ValueError(f"step {step}: missing/duplicate rollout or mismatched initial task")
    rows = []
    for mode, mapping in groups.items():
        for key, values in sorted(mapping.items()):
            rows.append({"mode": mode, "step": key[0], "group_sha256": fingerprint(key),
                         "size": len(values), "trajectories": len({v["trajectory"] for v in values}),
                         "actions": len({v["action"] for v in values}),
                         "returns": len({v["return"] for v in values})})
    return {"status": "complete_offline_diagnostic", "method": manifest["method"],
            "seed": manifest["seed"], "training_steps": len(batches),
            "episodes": sum(map(len, batches.values())), "rollouts": rollouts, "prompts": prompts,
            "gamma": 1.0, "modes": {mode: summarize(mapping) for mode, mapping in groups.items()},
            "manifest_sha256": hashlib.sha256((run / "run_manifest.json").read_bytes()).hexdigest(),
            "sources": sources, "sources_sha256": fingerprint(sources),
            "limitations": ["not GiGPO training or an advantage implementation",
                "observation matching omits remaining horizon and policy history",
                "full_state is the replay hash, not a canonical GiGPO state",
                "history uses recorded observation/raw text, not verified identical token inputs",
                "heterogeneous returns do not isolate an action's causal effect",
                "initial states repeat by construction; noninitial coverage reported separately"]}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("use a new output directory; never overwrite audit evidence")
    payload, rows = audit(args.run, rollouts=args.rollouts, prompts=args.prompts)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (args.output_dir / "groups.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "step", "group_sha256", "size", "trajectories", "actions", "returns"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in payload.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
