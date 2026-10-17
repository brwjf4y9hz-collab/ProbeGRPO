"""Check real compact historical artifacts, without SSH, plotting, torch or training."""

import hashlib
import json
import unittest
from pathlib import Path

from test_final_controls import load_script

ROOT = Path(__file__).resolve().parents[1]
PLOT = load_script("plot_training_comparison")


class PublishedEvidenceTest(unittest.TestCase):
    def test_twelve_real_histories_match_exact_published_results(self):
        result = ROOT / "experiments/results/public_sokoban_main_v1"
        histories = PLOT.load_committed_history(result / "training_history/per_step.csv", result / "per_seed.csv")
        self.assertEqual(len(histories), 12)
        self.assertEqual(sum(len(rows) for rows in histories.values()), 612)

    def test_six_state_audits_have_complete_provenance_and_expected_scope(self):
        folder = ROOT / "experiments/results/public_sokoban_main_v1/state_group_audit"
        reports = [json.loads(path.read_text()) for path in sorted(folder.glob("ablation-*/*/summary.json"))]
        self.assertEqual(len(reports), 6)
        self.assertEqual({(r["method"], r["seed"]) for r in reports},
                         {(method, seed) for method in ("grpo", "linear_ucb_b2") for seed in (17, 42, 101)})
        self.assertEqual(sum(r["episodes"] for r in reports), 4800)
        for report in reports:
            encoded = json.dumps(report["sources"], sort_keys=True, ensure_ascii=False).encode()
            self.assertEqual(report["sources_sha256"], hashlib.sha256(encoded).hexdigest())
            self.assertEqual(len(report["sources"]), report["episodes"])
            self.assertEqual(report["training_steps"], 50)

    def test_upstream_preservation_assets_are_retained(self):
        for relative in ("REPRODUCE.md", "REPRODUCIBILITY.md", "REPO_AUDIT.md", "repro/verl.commit",
                         "repro/verl.remote", "patches/verl-probegrpo.patch", "scripts/setup_verl.sh",
                         "docs/incidents/README.md", "docs/results/figure-provenance.json"):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_original_incident_zip_remains_intact_after_live_source_changes(self):
        verifier = load_script("verify_incident_archive")
        result = verifier.verify(ROOT)
        self.assertEqual((result["status"], result["files"], result["incidents"]), ("passed", 32, 8))
