"""本文件用自动化断言锁定 schedulers 相关组件的输入约定、边界条件和预期行为；测试数据为验证样例，不代表正式实验结果。"""

import unittest
from collections import Counter
from dataclasses import replace

from probegrpo.schedulers import (
    EntropyScheduler,
    LinearUCBScheduler,
    RandomScheduler,
    choose_alternative_action,
    turn_features,
)
from probegrpo.types import ProbeResult, TurnRecord


def turn(index, entropy=1.0, action="left", legal_actions=("left", "right")):
    return TurnRecord(
        trajectory_id="traj",
        task_id="tiny-0",
        seed=1,
        turn_id=index,
        action=action,
        action_prefix=(),
        state_hash=f"state-{index}",
        token_indices=(index,),
        mean_entropy=entropy,
        logprob_margin=0.2,
        invalid_actions_before=0,
        legal_actions=legal_actions,
        max_horizon=6,
        suffix_tokens=10,
        final_reward=0.0,
    )


class SchedulerTest(unittest.TestCase):
    def test_short_episode_strata_use_observed_length_not_horizon(self):
        candidates = [replace(turn(i), max_horizon=12, metadata={"trajectory_turns": 3})
                      for i in range(3)]
        self.assertEqual([t.trajectory_fraction for t in candidates], [0, 0.5, 1])
        self.assertLess(candidates[-1].turn_fraction, 1 / 3)
        counts = Counter(
            LinearUCBScheduler(seed=seed).select(candidates, 1)[0].turn.turn_id
            for seed in range(600)
        )
        self.assertEqual(set(counts), {0, 1, 2})
        self.assertTrue(all(140 <= count <= 260 for count in counts.values()), counts)

    def test_single_anchor_warmup_covers_all_position_strata(self):
        candidates = [turn(index) for index in range(6)]
        # Production creates a seeded scheduler per episode and asks for one turn.
        buckets = Counter(
            min(2, int(LinearUCBScheduler(seed=seed).select(candidates, 1)[0].turn.turn_fraction * 3))
            for seed in range(600)
        )
        self.assertEqual(set(buckets), {0, 1, 2})
        self.assertTrue(all(140 <= count <= 260 for count in buckets.values()), buckets)

    def test_warmup_rounds_are_balanced_unique_and_reproducible(self):
        candidates = [turn(index) for index in range(6)]
        for count in range(1, 8):
            first = LinearUCBScheduler(seed=91).select(candidates, count)
            second = LinearUCBScheduler(seed=91).select(candidates, count)
            ids = [anchor.turn.turn_id for anchor in first]
            self.assertEqual(ids, [anchor.turn.turn_id for anchor in second])
            self.assertEqual(len(set(ids)), min(count, 6))
            if count >= 3:
                self.assertEqual({int(anchor.turn.turn_fraction * 3) for anchor in first[:3]},
                                 {0, 1, 2})

    def test_warmup_handles_empty_strata_and_ineligible_turns(self):
        candidates = [turn(4), turn(5), turn(0, legal_actions=("left",))]
        selected = LinearUCBScheduler(seed=7).select(candidates, 20)
        self.assertEqual({anchor.turn.turn_id for anchor in selected}, {4, 5})
        self.assertEqual(LinearUCBScheduler().select([], 1), [])

    def test_budget_zero_selects_nothing(self):
        candidates = [turn(0)]
        self.assertEqual(RandomScheduler().select(candidates, 0), [])
        self.assertEqual(EntropyScheduler().select(candidates, 0), [])
        self.assertEqual(LinearUCBScheduler().select(candidates, 0), [])

    def test_entropy_scheduler_selects_highest_entropy(self):
        candidates = [turn(0, 0.1), turn(1, 2.5), turn(2, 1.2)]
        selected = EntropyScheduler().select(candidates, 2)
        self.assertEqual([item.turn.turn_id for item in selected], [1, 2])

    def test_random_scheduler_is_reproducible(self):
        candidates = [turn(index) for index in range(6)]
        first = RandomScheduler(seed=99).select(candidates, 3)
        second = RandomScheduler(seed=99).select(candidates, 3)
        self.assertEqual(
            [item.turn.anchor_id for item in first],
            [item.turn.anchor_id for item in second],
        )

    def test_linear_ucb_updates_only_from_valid_probe(self):
        scheduler = LinearUCBScheduler(warmup_updates=0, warmup_probes=0)
        candidate = turn(1)
        invalid = ProbeResult(
            candidate.anchor_id,
            "left",
            "right",
            0.0,
            0.0,
            0.0,
            0,
            False,
            skipped_reason="bad_replay",
        )
        scheduler.observe(candidate, invalid)
        self.assertEqual(scheduler.observations, 0)
        valid = ProbeResult(
            candidate.anchor_id,
            "left",
            "right",
            1.0,
            0.0,
            1.0,
            20,
            True,
        )
        scheduler.observe(candidate, valid)
        self.assertEqual(scheduler.observations, 1)
        self.assertTrue(any(abs(value) > 0 for value in scheduler.theta))

    def test_features_are_bounded(self):
        features = turn_features(turn(5, entropy=100.0))
        self.assertEqual(len(features), 7)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in features))

    def test_alternative_is_legal_and_not_factual(self):
        candidate = turn(0, legal_actions=("left", "right", "up"))
        alternative = choose_alternative_action(candidate, seed=3)
        self.assertIn(alternative, candidate.legal_actions)
        self.assertNotEqual(alternative, candidate.action)


if __name__ == "__main__":
    unittest.main()

