import unittest
from dataclasses import replace

from probegrpo.agent_episode import TokenStream, run_episode, validate_merge
from probegrpo.agent_smoke import ScriptedTokenIO
from probegrpo.envs.sokoban import SokobanEnv
from probegrpo.rollout import extract_turn_records


class SokobanTest(unittest.TestCase):
    def test_replay_full_state_and_success(self):
        for task, actions in [
            ("push-up", ["up", "up"]),
            ("push-right", ["right"] * 3),
            ("push-left", ["left"] * 3),
        ]:
            first = SokobanEnv().replay(task, 17, actions)
            second = SokobanEnv().replay(task, 17, actions)
            self.assertEqual(first, second)
            self.assertTrue(first.terminal)
            self.assertEqual(first.reward, 1)

    def test_invalid_attempt_changes_hash_and_consumes_horizon(self):
        env = SokobanEnv(max_steps=2)
        state = env.reset("push-up", 17)
        blocked = env.step("down")
        self.assertEqual(state.observation, blocked.observation)
        self.assertNotEqual(state.state_hash, blocked.state_hash)
        state = env.step("nonsense")
        self.assertTrue(state.terminal)
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.metadata["invalid_actions"], 2)
        with self.assertRaises(RuntimeError):
            env.step("up")

    def test_wall_blocks_push(self):
        env = SokobanEnv(max_steps=8)
        env.reset("push-right", 17)
        env.step("up")
        for _ in range(4):
            env.step("right")
        self.assertNotIn("right", env.state().legal_actions)


class EpisodeTest(unittest.IsolatedAsyncioTestCase):
    async def episode(self, actions=("up", "up"), **kwargs):
        return await run_episode(
            ScriptedTokenIO(actions),
            SokobanEnv(),
            task_id="push-up",
            seed=17,
            trajectory_id="test",
            **kwargs,
        )

    async def test_multiturn_masks_replay_and_bridge(self):
        episode = await self.episode()
        self.assertEqual(len(episode.turns), 2)
        self.assertEqual(episode.final_reward, 1)
        self.assertEqual(episode.turns[1].action_prefix, ("up",))
        masks = episode.stream.response_mask
        owned = {i for t in episode.turns for i in t.token_indices}
        self.assertEqual(owned, {i for i, flag in enumerate(masks) if flag})
        self.assertGreater(masks.count(0), 0)
        self.assertEqual(masks[-1], 1)
        for turn in episode.turns:
            state = SokobanEnv().replay("push-up", 17, turn.action_prefix)
            self.assertEqual(state.state_hash, turn.state_hash)
        n = len(masks)
        trace = episode.to_trace(
            entropies=[0.5] * n, top1_logprobs=[-0.1] * n, top2_logprobs=[-1.0] * n
        )
        records = extract_turn_records(trace)
        self.assertFalse(records[-1].terminal)  # Winning action still has a valid anchor.
        self.assertEqual(records[1].token_indices, episode.turns[1].token_indices)
        self.assertIsNone(episode.as_dict()["token_entropy"])
        with self.assertRaises(ValueError):
            episode.to_trace(entropies=[], top1_logprobs=[], top2_logprobs=[])

    async def test_invalid_model_text_and_horizon(self):
        episode = await self.episode(["I choose up", "up"], max_turns=2)
        self.assertFalse(episode.turns[0].action_valid)
        self.assertEqual(episode.turns[1].action_prefix, ("invalid",))
        self.assertEqual(episode.stop_reason, "horizon")
        self.assertEqual(episode.final_reward, 0)

    async def test_budget_does_not_execute_unseen_action(self):
        episode = await self.episode(response_budget=40)
        self.assertEqual(len(episode.turns), 1)
        self.assertEqual(episode.stop_reason, "token_budget")
        self.assertEqual(episode.stream.response_mask, (1, 1))

    async def test_empty_generation_after_feedback_strips_context(self):
        episode = await self.episode(["up", ""])
        self.assertEqual(episode.stop_reason, "empty_generation")
        self.assertEqual(episode.stream.response_mask, (1, 1))

    async def test_no_generation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "No assistant"):
            await self.episode([""])

    async def test_reproducible_scripted_episode_and_concurrent_isolation(self):
        import asyncio

        a, b = await asyncio.gather(self.episode(), self.episode())
        self.assertEqual(a, b)

    def test_context_cannot_rewrite_or_train_old_tokens(self):
        stream = TokenStream((10, 20, 30), (1, 1), (-0.1, -0.2))
        rewrite = replace(stream, token_ids=(10, 99, 30))
        with self.assertRaisesRegex(ValueError, "rewrote"):
            validate_merge(stream, rewrite, assistant=False)
        context = TokenStream((10, 20, 30, 40), (1, 1, 1), (-0.1, -0.2, 0))
        with self.assertRaisesRegex(ValueError, "Observation"):
            validate_merge(stream, context, assistant=False)
