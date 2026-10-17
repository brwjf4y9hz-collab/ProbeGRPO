"""Synthetic structural fixtures only; no model results or empirical state-coverage claims."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from test_final_controls import load_script

AUDIT = load_script("audit_state_groups")


def fixture(root, steps=1):
    (root / "run_manifest.json").write_text(json.dumps({"method": "CPU-fixture", "seed": 17,
                                                       "total_training_steps": steps}))
    folder = root / "rollouts/episodes/step-1"
    folder.mkdir(parents=True)
    for session, final in ((0, 1), (1, 0)):
        turn = {"observation": "same board", "state_hash": "same strict state", "raw_text": "up",
                "action": "up", "reward": 0}
        episode = {"dataset_split": "train", "trajectory_id": f"prompt_with_underscore_{session}",
                   "task_id": "board", "seed": 100, "max_horizon": 12, "final_reward": final,
                   "turns": [turn, {**turn, "state_hash": "later strict state", "reward": final,
                                    "action": "down" if session else "up"}]}
        (folder / f"{session}.json").write_text(json.dumps(episode))
    return folder


class StateGroupAuditTest(unittest.TestCase):
    def test_modes_separate_horizon_but_keep_original_prompt_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            payload, rows = AUDIT.audit(root, rollouts=2, prompts=1)
            self.assertEqual(payload["episodes"], 2)
            self.assertEqual(payload["modes"]["observation"]["groups"], 1)
            self.assertEqual(payload["modes"]["horizon_observation"]["groups"], 2)
            self.assertEqual(payload["modes"]["history"]["groups"], 2)
            self.assertEqual(payload["modes"]["full_state"]["cross_trajectory_occurrence_fraction"], 1)
            self.assertEqual(payload["modes"]["observation"]["cross_trajectory_diverse_action_return_fraction"], 1)
            self.assertEqual(len(payload["sources"]), 2)
            self.assertTrue(rows)

    def test_does_not_pool_updates_or_prompt_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = fixture(root, steps=2)
            for source in list(folder.glob("*.json")):
                episode = json.loads(source.read_text())
                episode["trajectory_id"] = "another_" + episode["trajectory_id"]
                (folder / f"another-{source.name}").write_text(json.dumps(episode))
            second = folder.parent / "step-2"
            second.mkdir()
            for source in folder.glob("*.json"):
                (second / source.name).write_bytes(source.read_bytes())
            payload, _ = AUDIT.audit(root, rollouts=2, prompts=2)
            self.assertEqual(payload["modes"]["observation"]["groups"], 4)

    def test_duplicate_or_missing_rollouts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = fixture(root)
            (folder / "duplicate.json").write_bytes((folder / "0.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                AUDIT.audit(root, rollouts=2, prompts=1)

    def test_incomplete_history_and_missing_sidecars_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, steps=2)
            with self.assertRaisesRegex(ValueError, "step coverage"):
                AUDIT.audit(root, rollouts=2, prompts=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "missing full episode"):
                AUDIT.audit(root)

    def test_invalid_returns_rejected_and_test_split_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = fixture(root)
            original = json.loads((folder / "0.json").read_text())
            test = copy.deepcopy(original)
            test["dataset_split"] = "test"
            (folder / "test.json").write_text(json.dumps(test))
            self.assertEqual(AUDIT.audit(root, rollouts=2, prompts=1)[0]["episodes"], 2)
            original["final_reward"] = float("nan")
            (folder / "0.json").write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "terminal reward"):
                AUDIT.audit(root, rollouts=2, prompts=1)
