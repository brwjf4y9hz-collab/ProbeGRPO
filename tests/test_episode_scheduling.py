import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from probegrpo.agent_episode import run_episode
from probegrpo.agent_smoke import ScriptedTokenIO
from probegrpo.envs.sokoban import SokobanEnv
from probegrpo.episode_scheduling import (
    episode_turn_records,
    select_episode_anchor,
    serialized_scheduler_turn,
)
from probegrpo.types import ProbeResult


class EpisodeSchedulingTest(unittest.IsolatedAsyncioTestCase):
    async def episode(self):
        return await run_episode(
            ScriptedTokenIO(("up", "up")),
            SokobanEnv(),
            task_id="push-up",
            seed=17,
            trajectory_id="trajectory",
        )

    async def test_records_use_chosen_token_surprisal(self):
        episode = await self.episode()
        records = episode_turn_records(episode)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1].turn_id, 1)
        self.assertEqual(records[1].metadata["uncertainty_statistic"], "chosen_token_surprisal")

    async def test_all_runnable_schedulers_select_eligible_anchor(self):
        episode = await self.episode()
        with tempfile.TemporaryDirectory() as directory:
            for scheduler in ("random", "surprisal", "linear_ucb"):
                anchor = select_episode_anchor(
                    episode,
                    scheduler,
                    training_update=21,
                    settings={"warmup_updates": 0, "warmup_probes": 0, "exploration_rate": 0},
                    sidecar_root=Path(directory),
                )
                self.assertIsNotNone(anchor)
                self.assertIn(anchor.turn.action, anchor.turn.legal_actions)

    async def test_linear_ucb_replays_only_previous_step_history(self):
        episode = await self.episode()
        anchor = select_episode_anchor(
            episode,
            "linear_ucb",
            training_update=0,
            settings={},
            sidecar_root=Path("missing"),
        )
        probe = ProbeResult(
            anchor_id=anchor.turn.anchor_id,
            factual_action=anchor.turn.action,
            counterfactual_action="left",
            factual_reward=1.0,
            counterfactual_reward=0.0,
            delta=1.0,
            additional_rollout_tokens=10,
            state_match=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            step = Path(directory) / "step-1"
            step.mkdir()
            (step / "episode.json").write_text(
                json.dumps(
                    {
                        "debug_probe": {
                            **asdict(probe),
                            "scheduler": "linear_ucb_warmup",
                            "scheduler_turn": serialized_scheduler_turn(anchor),
                        }
                    }
                )
            )
            selected = select_episode_anchor(
                episode,
                "linear_ucb",
                training_update=2,
                settings={"warmup_updates": 0, "warmup_probes": 0, "exploration_rate": 0},
                sidecar_root=Path(directory),
            )
            self.assertIsNotNone(selected)


if __name__ == "__main__":
    unittest.main()
