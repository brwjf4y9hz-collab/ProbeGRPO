import runpy
import tempfile
import unittest
from pathlib import Path

AGGREGATE_SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_sokoban_seeds.py"
PLOT_SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_sokoban_results.py"
AGGREGATE = runpy.run_path(str(AGGREGATE_SCRIPT))
PLOT = runpy.run_path(str(PLOT_SCRIPT))


def row(seed, method, success, overhead=0.0):
    return {
        "seed": seed,
        "method": method,
        "success_count": round(success * 100),
        "eval_count": 100,
        "final_val_success": success,
        "probe_extra_tokens": round(overhead * 1000),
        "rollout_token_overhead": overhead,
        "valid_probes": 1 if method != "grpo" else 0,
        "probe_attempts": 1 if method != "grpo" else 0,
        "high_impact_per_1k_probe_tokens": 2.0 if method != "grpo" else None,
        "gpu_hours": 0.25,
    }


class SeedAggregationTest(unittest.TestCase):
    def setUp(self):
        self.rows = []
        values = {
            "grpo": (0.2, 0.2, 0.2),
            "random_b2": (0.2, 0.3, 0.4),
            "surprisal_b2": (0.3, 0.3, 0.3),
            "linear_ucb_b2": (0.25, 0.30, 0.35),
        }
        for method, successes in values.items():
            for seed, success in zip((17, 42, 101), successes):
                self.rows.append(row(seed, method, success, overhead=0.5))

    def test_uses_paired_seed_uplift_and_sample_sd(self):
        result = {item["method"]: item for item in AGGREGATE["aggregate"](self.rows)}
        linear = result["linear_ucb_b2"]
        self.assertAlmostEqual(linear["success_mean"], 0.30)
        self.assertAlmostEqual(linear["success_sample_sd"], 0.05)
        self.assertAlmostEqual(linear["paired_uplift_pp_mean"], 10.0)
        self.assertAlmostEqual(linear["paired_uplift_pp_sample_sd"], 5.0)

    def test_requires_all_methods_for_every_seed(self):
        with self.assertRaisesRegex(ValueError, "methods"):
            AGGREGATE["aggregate"](self.rows[:-1])

    def test_dependency_free_svg_contains_accessible_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "rows.csv"
            svg_path = Path(temporary) / "result.svg"
            AGGREGATE["write_csv"](csv_path, self.rows)
            PLOT["render_svg"](PLOT["load_rows"](csv_path), svg_path)
            svg = svg_path.read_text()
            self.assertIn("<title>ProbeGRPO public Sokoban", svg)
            self.assertIn("LinearUCB", svg)
            self.assertIn("sample SD", svg)


if __name__ == "__main__":
    unittest.main()
