import json
import runpy
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_sokoban_ablation.py"
SUMMARIZE = runpy.run_path(str(SCRIPT))["summarize"]


class AblationSummaryTest(unittest.TestCase):
    def test_combines_training_log_and_episode_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "random_b2"
            sidecars = run / "rollouts" / "episodes" / "step-1"
            logs = run / "logs"
            sidecars.mkdir(parents=True)
            logs.mkdir()
            training = {
                "dataset_split": "train",
                "final_reward": 1.0,
                "turns": [
                    {"action_valid": True, "generated_token_ids": [1, 2]},
                    {"action_valid": False, "generated_token_ids": [3]},
                ],
                "debug_probe": {
                    "additional_rollout_tokens": 10,
                    "skipped_reason": None,
                    "state_match": True,
                    "delta": 1.0,
                },
            }
            validation = {
                "dataset_split": "test",
                "final_reward": 0.0,
                "oracle_shortest_steps": 6,
                "turns": [],
            }
            (sidecars / "train.json").write_text(json.dumps(training))
            (sidecars / "test.json").write_text(json.dumps(validation))
            (logs / "train-1-steps.log").write_text(
                "training/global_step:1 - critic/score/mean:0.5 - "
                "timing_s/step:36.0 - probe/valid:1\n"
            )
            row = SUMMARIZE(root)[0]
            self.assertEqual(row["valid_probes"], 1)
            self.assertEqual(row["probe_extra_tokens"], 10)
            self.assertEqual(row["main_generated_tokens"], 3)
            self.assertEqual(row["final_val_success"], 0.0)
            self.assertEqual(row["final_val_success_hard"], 0.0)
            self.assertAlmostEqual(row["invalid_action_rate"], 0.5)
            self.assertAlmostEqual(row["gpu_hours"], 0.01)


if __name__ == "__main__":
    unittest.main()
