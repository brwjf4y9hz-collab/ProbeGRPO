import unittest
from dataclasses import replace

from probegrpo.rollout import RawAssistantTurn, TrajectoryTrace, extract_turn_records


def raw_turn(
    action,
    token_indices,
    *,
    entropies=None,
    top1=None,
    top2=None,
    action_valid=True,
):
    count = len(token_indices)
    return RawAssistantTurn(
        action=action,
        state_hash=f"state-{action}",
        token_indices=tuple(token_indices),
        token_entropies=tuple(entropies if entropies is not None else [1.0] * count),
        top1_logprobs=tuple(top1 if top1 is not None else [-0.1] * count),
        top2_logprobs=tuple(top2 if top2 is not None else [-0.4] * count),
        legal_actions=("left", "right", "wait"),
        action_valid=action_valid,
    )


class RolloutExtractionTest(unittest.TestCase):
    def test_extracts_prefix_statistics_and_suffix_cost(self):
        trace = TrajectoryTrace(
            trajectory_id="trajectory-0",
            task_id="task-0",
            seed=9,
            token_count=8,
            max_horizon=3,
            final_reward=1.0,
            turns=(
                raw_turn(
                    "right",
                    (0, 1),
                    entropies=(0.2, 0.4),
                    top1=(-0.1, -0.2),
                    top2=(-0.4, -0.4),
                ),
                raw_turn("wait", (3, 4), action_valid=False),
                raw_turn("left", (6,)),
            ),
        )

        records = extract_turn_records(trace)

        self.assertEqual(
            [record.action_prefix for record in records],
            [(), ("right",), ("right", "wait")],
        )
        self.assertEqual([record.invalid_actions_before for record in records], [0, 0, 1])
        self.assertEqual([record.suffix_tokens for record in records], [5, 3, 1])
        self.assertAlmostEqual(records[0].mean_entropy, 0.3)
        self.assertAlmostEqual(records[0].logprob_margin, 0.25)
        self.assertTrue(all(record.final_reward == 1.0 for record in records))

    def test_rejects_misaligned_token_statistics(self):
        turn = raw_turn("left", (0, 1), entropies=(0.5,))
        trace = self._single_turn_trace(turn)
        with self.assertRaisesRegex(ValueError, "identical lengths"):
            extract_turn_records(trace)

    def test_rejects_duplicate_or_out_of_range_tokens(self):
        duplicate = self._single_turn_trace(raw_turn("left", (1, 1)))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            extract_turn_records(duplicate)

        out_of_range = replace(
            self._single_turn_trace(raw_turn("left", (0,))),
            turns=(raw_turn("left", (4,)),),
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            extract_turn_records(out_of_range)

    def test_rejects_tokens_shared_by_two_turns(self):
        trace = replace(
            self._single_turn_trace(raw_turn("left", (0, 1))),
            max_horizon=2,
            turns=(raw_turn("right", (0, 1)), raw_turn("left", (1, 2))),
        )
        with self.assertRaisesRegex(ValueError, "more than one turn"):
            extract_turn_records(trace)

    @staticmethod
    def _single_turn_trace(turn):
        return TrajectoryTrace(
            trajectory_id="trajectory-0",
            task_id="task-0",
            seed=1,
            token_count=4,
            max_horizon=1,
            final_reward=0.0,
            turns=(turn,),
        )


if __name__ == "__main__":
    unittest.main()
