import unittest

from probegrpo.envs import TinySokobanEnv, TinySokobanSuffixPolicy
from probegrpo.probing import CounterfactualProber
from probegrpo.types import Anchor, TurnRecord


def make_turn(state_hash, legal_actions, *, action="left"):
    return TurnRecord(
        trajectory_id="trajectory-0",
        task_id="tiny-0",
        seed=42,
        turn_id=0,
        action=action,
        action_prefix=(),
        state_hash=state_hash,
        token_indices=(1, 2),
        mean_entropy=1.0,
        logprob_margin=0.1,
        invalid_actions_before=0,
        legal_actions=legal_actions,
        max_horizon=2,
        suffix_tokens=2,
        final_reward=1.0,
    )


class ReplayAndProbeTest(unittest.TestCase):
    def test_replay_is_deterministic(self):
        env = TinySokobanEnv()
        first = env.replay("tiny-0", 9, ["right"])
        second = env.replay("tiny-0", 9, ["right"])
        self.assertEqual(first.observation, second.observation)
        self.assertEqual(first.state_hash, second.state_hash)

    def test_probe_uses_same_anchor_state(self):
        env = TinySokobanEnv()
        initial = env.reset("tiny-0", 42)
        anchor = Anchor(make_turn(initial.state_hash, initial.legal_actions), "test", 0.0)
        result = CounterfactualProber(
            TinySokobanEnv, TinySokobanSuffixPolicy(), state_hash_strict=True
        ).probe(anchor, "left", "right", sampling_seed=3)
        self.assertTrue(result.valid)
        self.assertTrue(result.state_match)
        self.assertEqual(result.factual_reward, 1.0)
        self.assertEqual(result.counterfactual_reward, 0.0)
        self.assertEqual(result.delta, 1.0)
        self.assertGreater(result.additional_rollout_tokens, 0)

    def test_state_mismatch_falls_back_to_zero_credit(self):
        env = TinySokobanEnv()
        initial = env.reset("tiny-0", 42)
        turn = make_turn("not-the-real-hash", initial.legal_actions)
        result = CounterfactualProber(TinySokobanEnv, TinySokobanSuffixPolicy()).probe(
            Anchor(turn, "test", 0.0), "left", "right", sampling_seed=3
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.delta, 0.0)
        self.assertEqual(result.skipped_reason, "anchor_state_mismatch")

    def test_missing_alternative_is_safe(self):
        env = TinySokobanEnv()
        initial = env.reset("tiny-0", 42)
        turn = make_turn(initial.state_hash, ("left",))
        result = CounterfactualProber(TinySokobanEnv, TinySokobanSuffixPolicy()).probe(
            Anchor(turn, "test", 0.0), "left", None, sampling_seed=3
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.additional_rollout_tokens, 0)


if __name__ == "__main__":
    unittest.main()

