"""Validate response masks and replay every recorded pre-action state without a GPU."""

import argparse
import json
from pathlib import Path

from probegrpo.envs.sokoban import SokobanEnv


def inspect(path):
    episode = json.loads(path.read_text())
    stream = episode["stream"]
    mask = stream["response_mask"]
    env = SokobanEnv(max_steps=episode["max_horizon"])
    state = env.reset(episode["task_id"], episode["seed"])
    prefix, owned = [], set()
    for turn in episode["turns"]:
        assert turn["action_prefix"] == prefix, path
        assert state.state_hash == turn["state_hash"], path
        assert state.observation == turn["observation"], path
        indices = set(turn["token_indices"])
        assert indices and not (owned & indices), path
        assert all(0 <= i < len(mask) and mask[i] == 1 for i in indices), path
        owned.update(indices)
        state = env.step(turn["action"])
        assert state.state_hash == turn["next_state_hash"], path
        assert state.observation == turn["next_observation"], path
        prefix.append(turn["action"])
    assert owned == {i for i, flag in enumerate(mask) if flag}, path
    assert state.reward == episode["final_reward"] and mask[-1] == 1, path
    return episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="One episodes/step-N directory")
    args = parser.parse_args()
    files = sorted(args.directory.glob("*.json"))
    if not files:
        raise SystemExit("No episode sidecars found")
    episodes = [inspect(path) for path in files]
    print(f"Validated {len(episodes)} episodes; all state hashes and assistant masks match")
    print(f"Successes: {sum(e['final_reward'] == 1 for e in episodes)}/{len(episodes)}")
    print(f"Assistant turns: {[len(e['turns']) for e in episodes]}")
    print("Fixture smoke results only; not a held-out benchmark score.")


if __name__ == "__main__":
    main()
