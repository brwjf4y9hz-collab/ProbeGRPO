import json
import tempfile
import unittest
from pathlib import Path

from test_final_controls import load_script

AUDIT = load_script("audit_engineering_logs")


class EngineeringLogTest(unittest.TestCase):
    def test_descriptive_values_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_manifest.json").write_text(json.dumps({"method": "grpo", "seed": 17,
                                                               "total_training_steps": 2}))
            (root / "logs").mkdir()
            (root / "logs/train-2-steps.log").write_text(
                "training/global_step:1 - actor/perf/max_memory_allocated_gb:20 - actor/perf/max_memory_reserved_gb:22 - timing_s/step:10\n"
                "training/global_step:2 - actor/perf/max_memory_allocated_gb:21 - actor/perf/max_memory_reserved_gb:23 - timing_s/step:12\n")
            result = AUDIT.audit(root)
            self.assertEqual(result["step_seconds_mean"], 11)
            self.assertEqual(result["logged_allocated_peak_gb"], 21)
            self.assertEqual(len(result["log_sha256"]), 64)
            self.assertEqual(result["configuration_caveat"], "pre-padding-fix GRPO")

    def test_missing_memory_is_not_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_manifest.json").write_text(json.dumps({"total_training_steps": 1}))
            (root / "logs").mkdir()
            (root / "logs/train-1-steps.log").write_text("training/global_step:1 - timing_s/step:10\n")
            with self.assertRaisesRegex(ValueError, "coverage"):
                AUDIT.audit(root)
