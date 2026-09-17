import unittest

from probegrpo.envs import TinySokobanEnv, TinySokobanSuffixPolicy
from probegrpo.pipeline import ProbePipeline
from probegrpo.probing import CounterfactualProber
from probegrpo.schedulers import EntropyScheduler
from probegrpo.types import TurnRecord


class PipelineTest(unittest.TestCase):
    def test_end_to_end_budget_and_credit(self):
        initial = TinySokobanEnv().reset("tiny-0", 4)
        turn = TurnRecord(
            trajectory_id="trajectory-a",
            task_id="tiny-0",
            seed=4,
            turn_id=0,
            action="left",
            action_prefix=(),
            state_hash=initial.state_hash,
            token_indices=(2, 3),
            mean_entropy=1.0,
            logprob_margin=0.1,
            invalid_actions_before=0,
            legal_actions=initial.legal_actions,
            max_horizon=2,
            suffix_tokens=2,
            final_reward=1.0,
        )
        pipeline = ProbePipeline(
            EntropyScheduler(),
            CounterfactualProber(TinySokobanEnv, TinySokobanSuffixPolicy()),
            {"trajectory-a": 0},
            base_seed=7,
        )
        outcome = pipeline.run([turn], budget=1, training_update=3)
        self.assertEqual(len(outcome.anchors), 1)
        self.assertEqual(outcome.valid_probes, 1)
        self.assertEqual(len(outcome.credits), 1)
        self.assertEqual(outcome.credits[0].sample_index, 0)
        self.assertEqual(outcome.credits[0].token_indices, (2, 3))

    def test_budget_zero_does_no_work(self):
        pipeline = ProbePipeline(
            EntropyScheduler(),
            CounterfactualProber(TinySokobanEnv, TinySokobanSuffixPolicy()),
            {},
        )
        outcome = pipeline.run([], budget=0, training_update=0)
        self.assertEqual(outcome.valid_probes, 0)
        self.assertEqual(outcome.additional_rollout_tokens, 0)


if __name__ == "__main__":
    unittest.main()

