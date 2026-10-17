import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_final_controls import load_script

CLEAN = load_script("check_clean_cpu")


class CleanCpuTest(unittest.TestCase):
    def test_snapshot_excludes_artifacts_and_secret_file_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            for name in ("README.md", "scripts/a.py", "configs/a.json", "outputs/raw.json", ".env",
                         "docs/.DS_Store", "scripts/private.pem", "repro_outputs/status.json"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            self.assertEqual({str(p) for p in CLEAN.source_files(root)},
                             {"README.md", "scripts/a.py", "configs/a.json"})

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "scripts").mkdir()
            (root / "scripts/link.py").symlink_to("/tmp/not-a-source.py")
            with self.assertRaisesRegex(ValueError, "symlink"):
                CLEAN.source_files(root)

    def test_child_environment_does_not_inherit_credentials_or_gpu_visibility(self):
        env = CLEAN.clean_env()
        self.assertNotIn("SWANLAB_API_KEY", env)
        self.assertNotIn("HF_TOKEN", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["SWANLAB_ENABLED"], "0")

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                CLEAN.verify(Path(directory), Path(directory), "working-tree")

    def test_snapshot_keeps_fixed_framework_and_reproduction_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            files = {"REPRODUCE.md", "repro/verl.commit", "repro/verl.remote",
                     "patches/verl-probegrpo.patch", "data/manifests/public.json",
                     "docs/results/requirements-viz.txt"}
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("CPU fixture, not framework evidence")
            self.assertEqual({str(path) for path in CLEAN.source_files(root)}, files)
