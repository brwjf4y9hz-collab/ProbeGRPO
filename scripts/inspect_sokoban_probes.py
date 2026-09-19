"""Summarize the optional paired-suffix AgentLoop debug gate."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="An episodes/step-N directory")
    args = parser.parse_args()
    files = sorted(args.directory.glob("*.json"))
    if not files:
        raise SystemExit("No episode sidecars found")

    episodes = [json.loads(path.read_text()) for path in files]
    probes = [episode["debug_probe"] for episode in episodes if "debug_probe" in episode]
    reasons = Counter(probe.get("skipped_reason") or "valid" for probe in probes)
    print(f"Episodes: {len(episodes)}; probe attempts: {len(probes)}")
    print(f"Probe outcomes: {dict(reasons)}")
    for probe in probes:
        print(
            f"anchor={probe.get('anchor_id', 'none')} "
            f"{probe.get('factual_action', '?')} -> {probe.get('counterfactual_action', '?')} "
            f"delta={probe['delta']} extra_tokens={probe.get('additional_rollout_tokens', 0)} "
            f"applied={probe.get('advantage_applied', False)}"
        )
    if not probes:
        raise SystemExit("Debug probe gate was not enabled or did not run")
    if any(probe.get("advantage_applied") for probe in probes):
        raise SystemExit("Debug probe must not modify the GRPO advantage")


if __name__ == "__main__":
    main()
