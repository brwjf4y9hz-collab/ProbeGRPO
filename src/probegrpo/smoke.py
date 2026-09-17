"""End-to-end CPU smoke test used by `make smoke`."""

from __future__ import annotations

import json

from .advantage import ProbeCredit, blend_probe_advantages
from .envs import TinySokobanEnv, TinySokobanSuffixPolicy
from .probing import CounterfactualProber
from .schedulers import EntropyScheduler, LinearUCBScheduler, choose_alternative_action
from .types import TurnRecord


def _turn(turn_id: int, entropy: float, state_hash: str, legal_actions: tuple) -> TurnRecord:
    return TurnRecord(
        trajectory_id="smoke-trajectory",
        task_id="tiny-0",
        seed=7,
        turn_id=turn_id,
        action="left",
        action_prefix=(),
        state_hash=state_hash,
        token_indices=(2, 3),
        mean_entropy=entropy,
        logprob_margin=0.2,
        invalid_actions_before=0,
        legal_actions=legal_actions,
        max_horizon=3,
        suffix_tokens=2,
        final_reward=1.0,
    )


def run_smoke() -> dict:
    env = TinySokobanEnv()
    initial = env.reset("tiny-0", seed=7)
    replayed = env.replay("tiny-0", seed=7, action_prefix=())
    assert initial == replayed

    candidates = [
        _turn(0, 0.2, initial.state_hash, initial.legal_actions),
        _turn(1, 1.4, initial.state_hash, initial.legal_actions),
    ]
    anchor = EntropyScheduler().select(candidates, budget=1)[0]
    alternative = choose_alternative_action(anchor.turn, seed=3)
    assert alternative is not None

    prober = CounterfactualProber(TinySokobanEnv, TinySokobanSuffixPolicy())
    result = prober.probe(anchor, anchor.turn.action, alternative, sampling_seed=11)
    assert result.valid and result.delta == 1.0

    advantages = blend_probe_advantages(
        [[0.25] * 6],
        [ProbeCredit(sample_index=0, token_indices=(2, 3), delta=result.delta)],
        lambda_coef=0.5,
    )
    assert advantages[0] == [0.25, 0.25, 0.75, 0.75, 0.25, 0.25]

    learner = LinearUCBScheduler(warmup_updates=0, warmup_probes=0, seed=5)
    learner.observe(anchor.turn, result)
    learned = learner.select(candidates, budget=1, training_update=1)
    assert len(learned) == 1

    return {
        "anchor": anchor.turn.anchor_id,
        "alternative": alternative,
        "delta": result.delta,
        "additional_rollout_tokens": result.additional_rollout_tokens,
        "shaped_advantages": advantages[0],
        "learned_anchor": learned[0].turn.anchor_id,
    }


def main() -> None:
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))
    print("ProbeGRPO smoke test passed")


if __name__ == "__main__":
    main()

