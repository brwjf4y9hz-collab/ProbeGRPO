import unittest
from dataclasses import replace

from probegrpo.agent_episode import GeneratedAction, TokenStream, run_episode
from probegrpo.agent_smoke import ScriptedTokenIO
from probegrpo.envs.sokoban import SokobanEnv
from probegrpo.episode_probing import probe_episode


class EpisodeProbingTest(unittest.IsolatedAsyncioTestCase):
    async def make_episode(self):
        return await run_episode(
            ScriptedTokenIO(["up", "right", "right", "right"]),
            SokobanEnv(max_steps=4),
            task_id="push-up",
            seed=17,
            trajectory_id="trajectory-1",
            max_turns=4,
        )

    async def test_paired_suffix_uses_same_seed_and_replays_prefix(self):
        episode = await self.make_episode()
        seeds = []

        def io_factory(seed, branch):
            seeds.append((seed, branch))
            return ScriptedTokenIO(["right"] * 4)

        result = await probe_episode(
            episode,
            1,
            GeneratedAction((ord("u"), ord("p")), "up", (-0.1, -0.1)),
            env_factory=lambda: SokobanEnv(max_steps=4),
            io_factory=io_factory,
            sampling_seed=29,
            response_budget=2048,
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.anchor_id, "trajectory-1:1")
        self.assertEqual(result.factual_reward, 0.0)
        self.assertEqual(result.counterfactual_reward, 1.0)
        self.assertEqual(result.delta, -1.0)
        self.assertEqual(result.counterfactual_steps, 1)
        self.assertGreater(result.additional_rollout_tokens, 0)
        self.assertEqual(seeds, [(29, "factual"), (29, "counterfactual")])

    async def test_invalid_alternative_skips_without_generation(self):
        episode = await self.make_episode()
        calls = []

        def io_factory(seed, branch):
            calls.append((seed, branch))
            return ScriptedTokenIO(["right"] * 4)

        result = await probe_episode(
            episode,
            1,
            GeneratedAction((ord("z"),), "invalid", (-0.1,)),
            env_factory=lambda: SokobanEnv(max_steps=4),
            io_factory=io_factory,
            sampling_seed=29,
            response_budget=2048,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.delta, 0.0)
        self.assertEqual(result.skipped_reason, "alternative_action_is_not_legal")
        self.assertEqual(calls, [])

    async def test_anchor_context_mismatch_falls_back_to_zero_credit(self):
        episode = await self.make_episode()
        changed_tokens = (999,) + episode.stream.token_ids[1:]
        changed = replace(
            episode,
            stream=TokenStream(
                changed_tokens,
                episode.stream.response_mask,
                episode.stream.logprobs,
            ),
        )
        result = await probe_episode(
            changed,
            1,
            GeneratedAction((ord("u"), ord("p")), "up", (-0.1, -0.1)),
            env_factory=lambda: SokobanEnv(max_steps=4),
            io_factory=lambda seed, branch: ScriptedTokenIO(["right"] * 4),
            sampling_seed=29,
            response_budget=2048,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.delta, 0.0)
        self.assertEqual(result.skipped_reason, "anchor_context_mismatch")

    async def test_suffix_error_falls_back_to_zero_credit(self):
        episode = await self.make_episode()

        class FailingIO(ScriptedTokenIO):
            async def generate(self, stream, limit):
                raise TimeoutError("simulated generation timeout")

        result = await probe_episode(
            episode,
            1,
            GeneratedAction((ord("u"), ord("p")), "up", (-0.1, -0.1)),
            env_factory=lambda: SokobanEnv(max_steps=4),
            io_factory=lambda seed, branch: FailingIO([]),
            sampling_seed=29,
            response_budget=2048,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.additional_rollout_tokens, 0)
        self.assertIn("TimeoutError", result.skipped_reason)


if __name__ == "__main__":
    unittest.main()
