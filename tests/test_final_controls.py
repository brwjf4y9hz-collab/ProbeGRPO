import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


RUNNER = load_script("run_final_controls")
COMPLETE = load_script("check_run_complete")
AUDIT = load_script("audit_anchor_positions")
MANIFEST = load_script("write_run_manifest")
SUMMARY = load_script("summarize_final_controls")
CURVES = load_script("plot_training_curves")
HARDWARE = load_script("control_hardware")


class FinalControlTest(unittest.TestCase):
    def test_framework_patches_are_part_of_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "patches").mkdir()
            target = root / "patches/entropy.patch"
            target.write_text("fixture-v1")
            before = MANIFEST.source_snapshot(root)
            target.write_text("fixture-v2")
            after = MANIFEST.source_snapshot(root)
            self.assertIn("patches/entropy.patch", before["files"])
            self.assertNotEqual(before["sha256"], after["sha256"])

    def test_200_step_protocol_preserves_all_other_scientific_settings(self):
        protocol = RUNNER.load_protocol()
        previous = RUNNER.load_protocol(RUNNER.PROJECT / "experiments/protocols/public_sokoban_controls_v2.json")
        self.assertEqual(protocol["training_steps"], 200)
        for key in ("frozen", "arms", "seeds", "train_batch_size", "validation_episodes"):
            self.assertEqual(protocol[key], previous[key])
        self.assertNotEqual(protocol["protocol"], previous["protocol"])
        plan = RUNNER.build_plan(protocol, "main", Path("outputs/final-controls-v3-200"))
        self.assertEqual(len(plan), 12)
        with patch.dict("os.environ", {"TOTAL_TRAINING_STEPS": "1", "SAVE_FREQ": "-1",
                                       "TRAIN_LIMIT": "1", "TEST_LIMIT": "1", "MODEL_PATH": "wrong"}):
            for run in plan:
                env, overrides = RUNNER.launch_settings(run, protocol, "/fixed/model", Path("/fixed/verl"))
                self.assertEqual(env["TOTAL_TRAINING_STEPS"], "200")
                self.assertEqual(env["SAVE_FREQ"], "50")
                self.assertEqual((env["TRAIN_LIMIT"], env["TEST_LIMIT"]), ("512", "128"))
                self.assertEqual(env["MODEL_PATH"], "/fixed/model")
                self.assertIn("trainer.test_freq=10", overrides)
                self.assertIn("trainer.max_actor_ckpt_to_keep=2", overrides)
                self.assertIn(f"trainer.experiment_name=controls-v3-200-single_48gb-{run['method']}-seed{run['seed']}-steps200",
                              overrides)

    def test_smoke_save_point_is_step_one_not_step_fifty(self):
        protocol = RUNNER.load_protocol()
        run = RUNNER.build_plan(protocol, "smoke", Path("outputs/test"))[0]
        env, _ = RUNNER.launch_settings(run, protocol, "/fixed/model", Path("/fixed/verl"))
        self.assertEqual((env["TOTAL_TRAINING_STEPS"], env["SAVE_FREQ"]), ("1", "1"))

    def test_200_step_acceptance_rejects_a_missing_middle_update(self):
        # Tiny structural fixtures, not downscaled training or model checkpoints.
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = {"total_training_steps": 200, "rollouts_per_prompt": 4, "train_batch_size": 1,
                        "probe": {"budget": 0, "lambda_coef": 0},
                        "data": {"derived_splits": {"test": {"rows": 2}}},
                        "hydra_overrides": ["trainer.test_freq=10"]}
            (run / "run_manifest.json").write_text(json.dumps(manifest))
            for step in range(201):
                folder = run / "rollouts/episodes" / f"step-{step}"
                folder.mkdir(parents=True)
                splits = (["test"] * 2 if step % 10 == 0 else []) + (["train"] * 4 if step else [])
                for i, split in enumerate(splits):
                    (folder / f"{i}.json").write_text(json.dumps({"dataset_split": split, "final_reward": 0}))
            actor = run / "checkpoints/global_step_200/actor"
            actor.mkdir(parents=True)
            (actor / "model_world_size_1_rank_0.pt").write_bytes(b"fixture")
            (actor.parent / "data.pt").write_bytes(b"fixture")
            (run / "logs").mkdir()
            log = run / "logs/train-200-steps.log"
            lines = [f"training/global_step:{step} - actor/grad_norm:1 - actor/pg_loss:0 "
                     "- probe/changed_tokens:0 - probe/credit_abs_sum:0 - timing_s/step:1\n"
                     for step in range(1, 201)]
            log.write_text("".join(lines))
            (run / "curves").mkdir()
            CURVES.write_csv(CURVES.collect_metrics(run), run / "curves/metrics_by_step.csv")
            receipt = COMPLETE.check(run)
            self.assertEqual(receipt["final_step"], 200)
            self.assertEqual(sum(bool(c["test"]) for c in receipt["episode_counts"].values()), 21)
            curve_rows = CURVES.collect_metrics(run)
            self.assertEqual(len(curve_rows), 201)
            self.assertEqual(curve_rows[-1]["step"], 200)
            CURVES.render_svg(curve_rows, run / "fixture.svg", "CPU test fixture, not training")
            self.assertIn(">step 200</text>", (run / "fixture.svg").read_text())
            log.write_text("".join(lines[:50]))
            with self.assertRaisesRegex(ValueError, "update 51"):
                COMPLETE.check(run)

    def test_50_step_run_cannot_enter_200_step_aggregate(self):
        protocol = RUNNER.load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "seed17/grpo"
            run.mkdir(parents=True)
            manifest = {"protocol": protocol["protocol"], "seed": 17, "total_training_steps": 50}
            (run / "run_manifest.json").write_text(json.dumps(manifest))
            (run / "completion_receipt.json").write_text(json.dumps({"status": "complete",
                "manifest_sha256": MANIFEST.sha256(run / "run_manifest.json")}))
            with self.assertRaisesRegex(ValueError, "frozen main protocol"):
                SUMMARY.summarize(Path(directory))

    def test_complete_control_has_zero_credit_and_real_final_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = {"total_training_steps": 1, "rollouts_per_prompt": 4, "train_batch_size": 1,
                        "probe": {"budget": 2, "lambda_coef": 0},
                        "data": {"derived_splits": {"test": {"rows": 2}}}, "hydra_overrides": []}
            (run / "run_manifest.json").write_text(json.dumps(manifest))
            for step in (0, 1):
                folder = run / "rollouts/episodes" / f"step-{step}"
                folder.mkdir(parents=True)
                splits = ["test"] * 2 + (["train"] * 4 if step else [])
                for index, split in enumerate(splits):
                    (folder / f"{index}.json").write_text(json.dumps({"dataset_split": split, "final_reward": 0}))
            actor = run / "checkpoints/global_step_1/actor"
            actor.mkdir(parents=True)
            (actor / "model_world_size_1_rank_0.pt").write_bytes(b"fixture-not-a-model")
            (actor.parent / "data.pt").write_bytes(b"fixture-not-a-dataloader")
            logs = run / "logs"
            logs.mkdir()
            log = logs / "train-1-steps.log"
            line = "training/global_step:1 - actor/grad_norm:1 - actor/pg_loss:0 - probe/changed_tokens:0 - probe/credit_abs_sum:0 - timing_s/step:1\n"
            log.write_text(line)
            (run / "curves").mkdir()
            CURVES.write_csv(CURVES.collect_metrics(run), run / "curves/metrics_by_step.csv")
            receipt = COMPLETE.check(run)
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["diagnostics"]["train"]["episodes"], 4)
            self.assertEqual(receipt["diagnostics"]["final_test"]["episodes"], 2)
            self.assertEqual(receipt["diagnostics"]["test_all_evaluations"]["episodes"], 4)
            log.write_text(line.replace("probe/changed_tokens:0", "probe/changed_tokens:2"))
            with self.assertRaisesRegex(ValueError, "zero probe/changed_tokens"):
                COMPLETE.check(run)

    def test_summary_pairs_seeds_and_measures_token_match_instead_of_assuming_it(self):
        protocol = json.loads(RUNNER.PROTOCOL.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in protocol["seeds"]:
                for arm in protocol["arms"]:
                    run = root / f"seed{seed}" / arm["method"]
                    (run / "curves").mkdir(parents=True)
                    manifest = {"protocol": protocol["protocol"], "seed": seed,
                                "method": arm["method"],
                                "total_training_steps": protocol["training_steps"], "rollouts_per_prompt": arm["rollouts"],
                                "probe": {"budget": arm["budget"], "lambda_coef": arm["lambda_coef"]},
                                "source_snapshot": {"sha256": "unit-fixture"},
                                "hardware_profile": HARDWARE.hardware_record(HARDWARE.load_profile(), {
                                    "devices": [{"name": "unit-fixture-not-real-GPU", "total_memory_mib": 48000,
                                                 "free_memory_mib": 47000}]})}
                    (run / "run_manifest.json").write_text(json.dumps(manifest))
                    (run / "completion_receipt.json").write_text(json.dumps({"status": "complete",
                        "manifest_sha256": MANIFEST.sha256(run / "run_manifest.json")}))
                    is_probe = arm["budget"] > 0
                    is_k6 = arm["method"] == "grpo_k6"
                    with (run / "curves/metrics_by_step.csv").open("w", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(["step", "val_success", "main_generated_tokens", "probe/extra_tokens", "timing_s/step"])
                        writer.writerow([0, 0.2, 0, 0, 0])
                        for step in range(1, protocol["training_steps"] + 1):
                            writer.writerow([step, 0.3 if is_probe else 0.2,
                                             180 if is_k6 else 100, 50 if is_probe else 0, 10])
                    (run / "completion_receipt.json").write_text(json.dumps({"status": "complete",
                        "manifest_sha256": MANIFEST.sha256(run / "run_manifest.json"),
                        "metrics_csv_sha256": MANIFEST.sha256(run / "curves/metrics_by_step.csv")}))
            result = SUMMARY.summarize(root)
            self.assertEqual(result["status"], "complete")
            self.assertAlmostEqual(result["aggregates"]["linear_ucb_b2"]["paired_vs_grpo_pp_mean"], 10)
            cost = result["k6_vs_probegrpo_cost"][0]
            self.assertAlmostEqual(cost["generated_token_ratio"], 1.2)
            self.assertFalse(cost["approximately_token_matched_within_10_percent"])

    def test_incomplete_control_matrix_never_produces_performance_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            result = SUMMARY.summarize(Path(directory))
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(len(result["missing_runs"]), 12)
            self.assertNotIn("aggregates", result)

    def test_confirmation_and_controls_partition_full_matrix(self):
        protocol = json.loads(RUNNER.PROTOCOL.read_text())
        root = Path("outputs/test-plan")
        main = RUNNER.build_plan(protocol, "main", root)
        split = RUNNER.build_plan(protocol, "confirm", root) + RUNNER.build_plan(protocol, "controls", root)
        self.assertEqual(len(main), 12)
        self.assertEqual({r["output_dir"] for r in main}, {r["output_dir"] for r in split})
        self.assertEqual(len(split), 12)
        self.assertTrue(all(row["steps"] == 200 for row in split))
        for row in main:
            if row["method"] == "grpo_k6":
                self.assertEqual((row["rollouts"], row["budget"]), (6, 0))
            if row["method"] == "linear_ucb_b2_l0":
                self.assertEqual((row["lambda_coef"], row["budget"]), (0, 2))

    def test_smoke_never_collides_with_200_step_main(self):
        protocol = json.loads(RUNNER.PROTOCOL.read_text())
        root = Path("outputs/test-plan")
        rows = RUNNER.build_plan(protocol, "smoke", root)
        main = RUNNER.build_plan(protocol, "main", root)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["seed"] == 17 and r["steps"] == 1 for r in rows))
        self.assertFalse({r["output_dir"] for r in rows} & {r["output_dir"] for r in main})
        with self.assertRaisesRegex(ValueError, "unsupported phase"):
            RUNNER.build_plan(protocol, "extended", root)

    def test_uncommitted_source_changes_alter_snapshot_not_output_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            file = root / "src/example.py"
            file.write_text("a = 1\n")
            before = MANIFEST.source_snapshot(root)["sha256"]
            (root / "outputs").mkdir()
            (root / "outputs/secret.log").write_text("not a source input")
            self.assertEqual(before, MANIFEST.source_snapshot(root)["sha256"])
            file.write_text("a = 2\n")
            self.assertNotEqual(before, MANIFEST.source_snapshot(root)["sha256"])

    def test_partial_final_step_cannot_be_reported_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_manifest.json").write_text(json.dumps({
                "total_training_steps": 1, "rollouts_per_prompt": 4, "train_batch_size": 4,
                "data": {"derived_splits": {"test": {"rows": 128}}}, "hydra_overrides": []}))
            (run / "rollouts/episodes/step-1").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "counts"):
                COMPLETE.check(run)

    def test_audit_ignores_unanchored_skips_and_records_available_late_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_manifest.json").write_text(json.dumps({"seed": 17}))
            step = run / "rollouts/episodes/step-1"
            step.mkdir(parents=True)
            base = {"dataset_split": "train", "max_horizon": 12,
                    "turns": [{"action": "up", "legal_actions": ["up", "down"]}] * 12}
            (step / "skip.json").write_text(json.dumps({**base, "debug_probe": {"scheduler": "linear_ucb"}}))
            (step / "anchor.json").write_text(json.dumps({**base, "debug_probe": {
                "scheduler": "linear_ucb_warmup", "scheduler_turn": {"turn_id": 0}, "state_match": True}}))
            rows, summary = AUDIT.audit(run)
            self.assertEqual(len(rows), 1)
            self.assertEqual(summary["warmup_eligible_bucket_sets"], {"0,1,2": 1})
            self.assertIsNone(summary["first_ranked_anchor_step"])


if __name__ == "__main__":
    unittest.main()
