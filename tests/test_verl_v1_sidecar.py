import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from probegrpo.integration.verl_v1_sidecar import (
    apply_sidecar_probe_credits,
    pack_sidecar_probes,
)


def write_episode(directory, trajectory_id, mask, indices, delta, *, anchor=0):
    payload = {
        "trajectory_id": trajectory_id,
        "stream": {"response_mask": mask},
        "turns": [{"token_indices": indices}],
        "debug_probe": {
            "anchor_turn_id": anchor,
            "anchor_id": f"{trajectory_id}:{anchor}",
            "state_match": True,
            "delta": delta,
            "additional_rollout_tokens": 12,
            "skipped_reason": None,
        },
    }
    filename = hashlib.sha256(trajectory_id.encode()).hexdigest() + ".json"
    (directory / filename).write_text(json.dumps(payload))


class VerlV1SidecarTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_matches_rows_and_only_assistant_turn_tokens(self):
        write_episode(self.directory, "first_0", [1, 0, 1], [2], 1.0)
        write_episode(self.directory, "second_0", [1, 1], [0, 1], 0.0)
        packed, metrics = pack_sidecar_probes(
            ["first_0_0", "second_0_0"],
            [[1, 0, 1, 0], [1, 1, 0, 0]],
            sidecar_dir=self.directory,
            token_count=4,
            budget=1,
        )
        self.assertEqual(packed.mask_shape, (2, 1, 4))
        self.assertEqual(packed.probe_turn_masks[0][0], (False, False, True, False))
        self.assertEqual(packed.probe_turn_masks[1][0], (True, True, False, False))
        self.assertEqual(packed.probe_deltas, ((1.0,), (0.0,)))
        self.assertEqual(metrics["probe/extra_tokens"], 24)

    def test_budget_zero_needs_no_sidecars(self):
        packed, metrics = pack_sidecar_probes(
            ["missing_0_0"], [[1, 0]],
            sidecar_dir=self.directory, token_count=2, budget=0,
        )
        self.assertEqual(packed.mask_shape, (1, 0, 2))
        self.assertEqual(metrics["probe/valid"], 0)

    def test_rejects_misaligned_mask_before_credit(self):
        write_episode(self.directory, "first_0", [1, 0, 1], [2], 1.0)
        with self.assertRaisesRegex(ValueError, "response mask mismatch"):
            pack_sidecar_probes(
                ["first_0_0"], [[1, 1, 1]],
                sidecar_dir=self.directory, token_count=3, budget=1,
            )

    def test_rejects_multi_span_episode(self):
        with self.assertRaisesRegex(ValueError, "multi-span"):
            pack_sidecar_probes(
                ["first_0_0", "first_0_1"], [[1], [1]],
                sidecar_dir=self.directory, token_count=1, budget=1,
            )

    def test_tensor_hook_changes_only_anchor_tokens(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is installed only in the verl runtime")
        write_episode(self.directory, "first_0", [1, 0, 1], [2], 1.0)
        write_episode(self.directory, "second_0", [1, 1], [0, 1], 0.0)
        base = torch.tensor([[0.5, 0.0, 0.5, 0.0], [0.2, 0.2, 0.0, 0.0]])
        data = SimpleNamespace(batch={
            "advantages": base.clone(),
            "response_mask": torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]]),
        })
        data, metrics = apply_sidecar_probe_credits(
            data, ["first_0_0", "second_0_0"],
            sidecar_dir=self.directory, budget=1, lambda_coef=0.5,
        )
        expected = torch.tensor([[0.5, 0.0, 1.0, 0.0], [0.2, 0.2, 0.0, 0.0]])
        self.assertTrue(torch.allclose(data.batch["advantages"], expected))
        self.assertEqual(metrics["probe/changed_tokens"], 1)
        self.assertEqual(metrics["probe/credit_abs_sum"], 0.5)
        self.assertEqual(data.batch["probe_turn_masks"].shape, (2, 1, 4))

    def test_lambda_zero_keeps_original_tensor(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is installed only in the verl runtime")
        base = torch.tensor([[0.5, 0.0]])
        data = SimpleNamespace(batch={"advantages": base, "response_mask": torch.tensor([[1, 0]])})
        data, _ = apply_sidecar_probe_credits(
            data, ["missing_0_0"], sidecar_dir=self.directory, lambda_coef=0,
        )
        self.assertIs(data.batch["advantages"], base)


if __name__ == "__main__":
    unittest.main()
