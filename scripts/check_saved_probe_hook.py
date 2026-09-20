"""Replay saved sidecars through the torch advantage adapter without loading a model."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from probegrpo.integration.verl_v1_sidecar import apply_sidecar_probe_credits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sidecar_dir", type=Path, help="episodes/step-N directory")
    args = parser.parse_args()
    episodes = [json.loads(path.read_text()) for path in sorted(args.sidecar_dir.glob("*.json"))]
    if not episodes:
        raise SystemExit("No episode sidecars found")

    import torch
    from tensordict import TensorDict

    token_count = max(len(e["stream"]["response_mask"]) for e in episodes)
    masks = [
        e["stream"]["response_mask"]
        + [0] * (token_count - len(e["stream"]["response_mask"]))
        for e in episodes
    ]
    keys = [f'{e["trajectory_id"]}_0' for e in episodes]
    base = torch.zeros(len(episodes), token_count)

    def trial(budget: int, lambda_coef: float):
        data = SimpleNamespace(batch=TensorDict({
            "advantages": base.clone(),
            "response_mask": torch.tensor(masks, dtype=torch.bool),
        }, batch_size=[len(episodes)]))
        return apply_sidecar_probe_credits(
            data, keys, sidecar_dir=args.sidecar_dir,
            budget=budget, lambda_coef=lambda_coef,
        )

    budget_zero, _ = trial(0, 0.5)
    lambda_zero, _ = trial(1, 0.0)
    assert torch.equal(budget_zero.batch["advantages"], base)
    assert torch.equal(lambda_zero.batch["advantages"], base)
    trained, metrics = trial(1, 0.5)
    print(f"N={len(episodes)} A=1 T={token_count}")
    print("budget=0 and lambda=0 exactly preserve GRPO: True")
    print(f"metrics: {metrics}")
    zero_delta_anchors = 0
    for index, episode in enumerate(episodes):
        changed = torch.nonzero(trained.batch["advantages"][index], as_tuple=True)[0].tolist()
        probe = episode.get("debug_probe")
        if probe and not probe.get("skipped_reason"):
            anchor = episode["turns"][probe["anchor_turn_id"]]["token_indices"]
            assert set(changed).issubset(anchor), "credit escaped the selected assistant turn"
            if float(probe["delta"]) == 0.0:
                zero_delta_anchors += 1
                assert not changed, "zero-delta anchor received nonzero credit"
        if changed:
            values = trained.batch["advantages"][index, changed].tolist()
            print(f'{episode["trajectory_id"]}: indices={changed} credit={values}')
    print(f"zero-delta valid anchors unchanged: {zero_delta_anchors}")


if __name__ == "__main__":
    main()
