"""CPU fixtures only: never allocate GPUs or present synthetic outcomes as experiments."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_final_controls import COMPLETE, CURVES, HARDWARE, MANIFEST, RUNNER, SUMMARY


def inventory(count=1):
    return {"devices": [{"name": "CPU-unit-fixture-not-real-GPU", "total_memory_mib": 48000,
                         "free_memory_mib": 47000} for _ in range(count)]}


def aggregate_fixture(root):
    protocol = RUNNER.load_protocol()
    hardware = HARDWARE.hardware_record(HARDWARE.load_profile(), inventory())
    for run in RUNNER.build_plan(protocol, "main", root):
        path = Path(run["output_dir"])
        (path / "curves").mkdir(parents=True)
        manifest = {"protocol": protocol["protocol"], "method": run["method"], "seed": run["seed"],
                    "total_training_steps": 200, "rollouts_per_prompt": run["rollouts"],
                    "probe": {"budget": run["budget"], "lambda_coef": run["lambda_coef"]},
                    "source_snapshot": {"sha256": "CPU-unit-fixture"}, "hardware_profile": hardware}
        (path / "run_manifest.json").write_text(json.dumps(manifest))
        with (path / "curves/metrics_by_step.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "val_success", "main_generated_tokens", "probe/extra_tokens", "timing_s/step"])
            for step in range(201):
                writer.writerow([step, 0.2 if step % 10 == 0 else "", 10 if step else 0,
                                 2 if step and run["budget"] else 0, 1 if step else 0])
        refresh_receipt(path)
    return root / "seed17/grpo"


def refresh_receipt(path):
    (path / "completion_receipt.json").write_text(json.dumps({"status": "complete",
        "manifest_sha256": MANIFEST.sha256(path / "run_manifest.json"),
        "metrics_csv_sha256": MANIFEST.sha256(path / "curves/metrics_by_step.csv")}))


class HardwareControlTest(unittest.TestCase):
    def test_all_profiles_preserve_formal_training_budget_and_sequence_limits(self):
        protocol = RUNNER.load_protocol()
        for name in HARDWARE.PROFILE_NAMES:
            profile = HARDWARE.load_profile(name)
            for run in RUNNER.build_plan(protocol, "main", Path("outputs/unit-fixture")):
                env, overrides = RUNNER.launch_settings(run, protocol, "/fixed/model", Path("/fixed/verl"), profile)
                self.assertEqual(env["TOTAL_TRAINING_STEPS"], "200")
                self.assertEqual((env["TRAIN_LIMIT"], env["TEST_LIMIT"]), ("512", "128"))
                self.assertEqual(int(env["ROLLOUTS_PER_PROMPT"]), run["rollouts"])
                self.assertNotIn("data.train_batch_size", profile["hydra_overrides"])
                self.assertGreaterEqual(profile["hydra_overrides"]["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"], 2048)
                self.assertGreaterEqual(profile["hydra_overrides"]["actor_rollout_ref.rollout.max_num_batched_tokens"], 2048)
                self.assertFalse(any("response_length" in key or "model.path" in key
                                     for key in profile["hydra_overrides"]))
                self.assertIn(f"trainer.n_gpus_per_node={profile['gpu_count']}", overrides)
                self.assertEqual((4 * run["rollouts"]) % profile["gpu_count"], 0)

    def test_visibility_never_expands_scheduler_allocation(self):
        base = {"CUDA_VISIBLE_DEVICES": "1,2"}
        self.assertEqual(HARDWARE.device_environment(base), base)
        self.assertEqual(HARDWARE.device_environment(base, "1,2"), base)
        for requested in ("0,1", "0,1,2"):
            with self.assertRaisesRegex(ValueError, "override"):
                HARDWARE.device_environment(base, requested)
        self.assertEqual(base, {"CUDA_VISIBLE_DEVICES": "1,2"})
        with self.assertRaises(ValueError):
            HARDWARE.device_environment({"CUDA_VISIBLE_DEVICES": ""}, "0")

    def test_invalid_or_duplicate_devices_are_rejected(self):
        for value in ("", "0,0", "0; echo bad", "-1", "0,"):
            with self.assertRaises(ValueError):
                HARDWARE.device_environment({}, value)
        self.assertEqual(HARDWARE.device_environment({}, "0, 2")["CUDA_VISIBLE_DEVICES"], "0,2")

    def test_wrong_count_insufficient_memory_and_busy_cards_are_rejected(self):
        profile = HARDWARE.load_profile("dual_24gb")
        HARDWARE.validate_inventory(profile, inventory(2))
        with self.assertRaisesRegex(ValueError, "count"):
            HARDWARE.validate_inventory(profile, inventory(3))
        insufficient = inventory(2)
        insufficient["devices"][0]["total_memory_mib"] = 12000
        with self.assertRaisesRegex(ValueError, "memory"):
            HARDWARE.validate_inventory(profile, insufficient)
        busy = inventory(2)
        busy["devices"][0]["free_memory_mib"] = 1000
        with self.assertRaisesRegex(ValueError, "exclusive"):
            HARDWARE.validate_inventory(profile, busy)

    def test_dual_card_plan_needs_neither_gpu_nor_runtime_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not-created"
            env = dict(os.environ)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            completed = subprocess.run([sys.executable, str(RUNNER.PROJECT / "scripts/run_final_controls.py"),
                "--verl-dir", "/not-a-runtime", "--hardware-profile", "dual_24gb", "--gpu-devices", "0,2",
                "--output-root", str(root)], capture_output=True, text=True, env=env, check=True)
            plan = json.loads(completed.stdout)
            self.assertEqual(plan["total_optimizer_updates"], 2400)
            self.assertEqual(plan["hardware_profile"]["name"], "dual_24gb")
            self.assertEqual(plan["mode"], "no-training")
            self.assertTrue(all("/dual_24gb/" in r["output_dir"] for r in plan["runs"]))
            self.assertFalse(root.exists())

    def test_formal_training_requires_all_smoke_arms_and_same_profile(self):
        protocol = RUNNER.load_protocol()
        hardware = HARDWARE.hardware_record(HARDWARE.load_profile(), inventory())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "smoke"):
                RUNNER.require_smoke(root, protocol, hardware, "fixture")
            for run in RUNNER.build_plan(protocol, "smoke", root):
                path = Path(run["output_dir"])
                path.mkdir(parents=True)
                manifest = {"method": run["method"], "seed": 17, "protocol": protocol["protocol"],
                            "total_training_steps": 1, "rollouts_per_prompt": run["rollouts"],
                            "probe": {"budget": run["budget"], "lambda_coef": run["lambda_coef"]},
                            "source_snapshot": {"sha256": "fixture"}, "hardware_profile": hardware}
                (path / "run_manifest.json").write_text(json.dumps(manifest))
                (path / "completion_receipt.json").write_text(json.dumps({
                    "manifest_sha256": MANIFEST.sha256(path / "run_manifest.json")}))
            with patch.object(RUNNER, "check_complete") as structural_check:
                RUNNER.require_smoke(root, protocol, hardware, "fixture")
                self.assertEqual(structural_check.call_count, 4)
                other = HARDWARE.hardware_record(HARDWARE.load_profile("dual_24gb"), inventory(2))
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    RUNNER.require_smoke(root, protocol, other, "fixture")

    def test_two_rank_checkpoint_requires_both_model_shards(self):
        profile = HARDWARE.load_profile("dual_24gb")
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            manifest = {"total_training_steps": 1, "train_batch_size": 1, "rollouts_per_prompt": 4,
                        "data": {"derived_splits": {"test": {"rows": 2}}},
                        "probe": {"budget": 0, "lambda_coef": 0},
                        "hardware_profile": HARDWARE.hardware_record(profile, inventory(2)),
                        "hydra_overrides": [f"{k}={json.dumps(v)}" for k,v in profile["hydra_overrides"].items()]}
            (run / "run_manifest.json").write_text(json.dumps(manifest))
            for step in (0, 1):
                folder = run / "rollouts/episodes" / f"step-{step}"
                folder.mkdir(parents=True)
                for index, split in enumerate(["test"] * 2 + (["train"] * 4 if step else [])):
                    (folder / f"{index}.json").write_text(json.dumps({"dataset_split": split, "final_reward": 0}))
            (run / "logs").mkdir()
            (run / "logs/train-1-steps.log").write_text("training/global_step:1 - actor/grad_norm:1 "
                "- actor/pg_loss:0 - timing_s/step:1 - probe/changed_tokens:0 - probe/credit_abs_sum:0\n")
            (run / "curves").mkdir()
            CURVES.write_csv(CURVES.collect_metrics(run), run / "curves/metrics_by_step.csv")
            actor = run / "checkpoints/global_step_1/actor"
            actor.mkdir(parents=True)
            (actor.parent / "data.pt").write_bytes(b"unit-fixture-not-model")
            (actor / "model_world_size_2_rank_0.pt").write_bytes(b"unit-fixture-not-model")
            with self.assertRaisesRegex(ValueError, "ranks"):
                COMPLETE.check(run)
            (actor / "model_world_size_2_rank_1.pt").write_bytes(b"unit-fixture-not-model")
            self.assertEqual(COMPLETE.check(run)["status"], "complete")

    def test_summary_rejects_mixed_hardware_even_when_source_is_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = aggregate_fixture(root)
            saved = json.loads((run / "run_manifest.json").read_text())
            saved["hardware_profile"] = HARDWARE.hardware_record(HARDWARE.load_profile("dual_24gb"), inventory(2))
            (run / "run_manifest.json").write_text(json.dumps(saved))
            refresh_receipt(run)
            with self.assertRaisesRegex(ValueError, "mixed hardware"):
                SUMMARY.summarize(root)

    def test_summary_rejects_modified_csv_and_nonfinite_or_duplicate_steps(self):
        for kind in ("checksum", "nan", "duplicate", "missing"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run = aggregate_fixture(root)
                path = run / "curves/metrics_by_step.csv"
                with path.open() as handle:
                    rows = list(csv.DictReader(handle))
                if kind == "duplicate":
                    rows.append(rows[-1])
                elif kind == "missing":
                    rows.pop()
                else:
                    rows[1]["timing_s/step"] = "nan"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                if kind != "checksum":
                    refresh_receipt(run)
                with self.assertRaises(ValueError):
                    SUMMARY.summarize(root)

    def test_human_readable_report_is_generated_only_for_complete_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete = SUMMARY.summarize(root)
            self.assertNotIn("Final success", SUMMARY.markdown(incomplete))
            aggregate_fixture(root)
            report = SUMMARY.markdown(SUMMARY.summarize(root))
            self.assertIn("200 updates", report)
            self.assertIn("Final success", report)
            self.assertIn("not automatically equal-compute", report)


if __name__ == "__main__":
    unittest.main()
