import tempfile
import unittest
from pathlib import Path

from probegrpo.config import ProbeConfig
from probegrpo.metrics import ExperimentRecord, append_jsonl, load_jsonl, summarize_records


class MetricsAndConfigTest(unittest.TestCase):
    def test_probe_config_validation(self):
        self.assertEqual(ProbeConfig().budget, 2)
        with self.assertRaises(ValueError):
            ProbeConfig(budget=-1)
        with self.assertRaises(ValueError):
            ProbeConfig(scheduler="magic")

    def test_jsonl_round_trip_and_summary(self):
        records = [
            ExperimentRecord("grpo", "sokoban", 1, 10, 0.4, 0.0, 5, 0.1, 100, 0, 4),
            ExperimentRecord("grpo", "sokoban", 2, 10, 0.6, 1.0, 4, 0.0, 120, 0, 5),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            for record in records:
                append_jsonl(path, record)
            loaded = load_jsonl(path)
        self.assertEqual(loaded, records)
        summary = summarize_records(loaded)[("grpo", "sokoban")]
        self.assertAlmostEqual(summary["reward_mean"], 0.5)
        self.assertAlmostEqual(summary["success_mean"], 0.5)
        self.assertEqual(summary["total_rollout_tokens"], 220.0)


if __name__ == "__main__":
    unittest.main()

