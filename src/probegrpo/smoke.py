"""End-to-end CPU data-flow demo used by ``make smoke``."""

from __future__ import annotations

import json

from .batching import blend_packed_probe_advantages, pack_probe_credits
from .envs import TinySokobanEnv, TinySokobanSuffixPolicy
from .pipeline import ProbePipeline
from .probing import CounterfactualProber
from .rollout import RawAssistantTurn, TrajectoryTrace, extract_turn_records
from .schedulers import EntropyScheduler


def _trace(trajectory_id: str, entropy: float, seed: int = 7) -> TrajectoryTrace:
    initial = TinySokobanEnv().reset("tiny-0", seed)
    return TrajectoryTrace(
        trajectory_id=trajectory_id,
        task_id="tiny-0",
        seed=seed,
        token_count=6,
        max_horizon=2,
        final_reward=1.0,
        turns=(
            RawAssistantTurn(
                action="left",
                state_hash=initial.state_hash,
                token_indices=(2, 3),
                token_entropies=(entropy, entropy),
                top1_logprobs=(-0.1, -0.2),
                top2_logprobs=(-0.3, -0.4),
                legal_actions=initial.legal_actions,
            ),
        ),
    )


def run_smoke() -> dict:
    traces = (
        _trace("trajectory-low-entropy", entropy=0.2),
        _trace("trajectory-high-entropy", entropy=1.4),
    )
    candidates = [record for trace in traces for record in extract_turn_records(trace)]
    trajectory_to_sample = {
        trace.trajectory_id: sample_index for sample_index, trace in enumerate(traces)
    }

    pipeline = ProbePipeline(
        EntropyScheduler(),
        CounterfactualProber(TinySokobanEnv, TinySokobanSuffixPolicy()),
        trajectory_to_sample,
        base_seed=7,
    )
    outcome = pipeline.run(candidates, budget=1, training_update=3)
    packed = pack_probe_credits(
        outcome.credits,
        batch_size=len(traces),
        token_count=traces[0].token_count,
        anchors_per_sample=1,
    )
    base_advantages = [[0.25] * traces[0].token_count for _ in traces]
    final_advantages = blend_packed_probe_advantages(
        base_advantages,
        packed,
        lambda_coef=0.5,
    )

    assert outcome.valid_probes == 1
    assert outcome.results[0].delta == 1.0
    assert outcome.anchors[0].turn.trajectory_id == "trajectory-high-entropy"
    assert packed.mask_shape == (2, 1, 6)
    assert final_advantages == [
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.75, 0.75, 0.25, 0.25],
    ]

    return {
        "symbols": {
            "N_batch_trajectories": packed.batch_size,
            "A_anchor_slots_per_trajectory": packed.anchors_per_sample,
            "T_padded_tokens": packed.token_count,
            "C_candidate_turns": len(candidates),
        },
        "turn_records": [
            {
                "anchor_id": turn.anchor_id,
                "action_prefix": turn.action_prefix,
                "token_indices": turn.token_indices,
                "mean_entropy": turn.mean_entropy,
            }
            for turn in candidates
        ],
        "selected_anchor": outcome.anchors[0].turn.anchor_id,
        "probe": {
            "factual_action": outcome.results[0].factual_action,
            "counterfactual_action": outcome.results[0].counterfactual_action,
            "delta": outcome.results[0].delta,
            "additional_rollout_tokens": outcome.additional_rollout_tokens,
        },
        "packed_batch": {
            "probe_turn_masks_shape": packed.mask_shape,
            "probe_deltas_shape": packed.matrix_shape,
            "probe_valid_shape": packed.matrix_shape,
            "probe_turn_masks": packed.probe_turn_masks,
            "probe_deltas": packed.probe_deltas,
            "probe_valid": packed.probe_valid,
        },
        "base_advantages": base_advantages,
        "final_advantages": final_advantages,
    }


def main() -> None:
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))
    print("ProbeGRPO CPU data-flow smoke test passed")


if __name__ == "__main__":
    main()
