"""Offline shell-flow checks; mocked tools do not validate a real training install."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        (self.project / "scripts").mkdir(parents=True)
        shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/bootstrap_verl.sh",
                        self.project / "scripts/bootstrap_verl.sh")
        (self.project / "patches").mkdir()
        (self.project / "patches/verl-probegrpo.patch").write_text("fixture patch")
        self.executable(self.project / "scripts/setup_verl.sh",
                        'printf checked > "$1/checked-source"\n')
        self.runtime = self.root / "runtime"
        self.fake_bin = self.root / "tools"
        self.fake_bin.mkdir()
        self.executable(self.fake_bin / "python3", "exit 77\n")
        self.executable(self.fake_bin / "uv", "printf 'uv 0.0.0\\n'\n")
        self.executable(self.fake_bin / "git", "printf 'fixture-commit\\n'\n")
        # macOS names this utility shasum; the Linux production script uses sha256sum.
        self.executable(self.fake_bin / "sha256sum", "printf 'fixture-sha  patch\\n'\n")
        self.env = {**os.environ, "PATH": str(self.fake_bin) + os.pathsep + os.environ["PATH"]}
        for key in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_BIN_DIR",
                    "PIP_CACHE_DIR", "HF_HOME"):
            self.env.pop(key, None)

    def executable(self, path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\nset -eu\n" + body)
        path.chmod(0o755)

    def partial(self, version="0.12.15"):
        (self.runtime / "verl/.git").mkdir(parents=True)
        self.executable(self.runtime / "verl/.venv/bin/python", "printf 'Python 3.12.3\\n'\n")
        self.executable(self.runtime / "bootstrap/bin/uv", """
if [[ "${1:-}" == --version ]]; then
  printf 'uv VERSION\n'
  exit
fi
printf '%s\n' "$*" >> "$UV_CACHE_DIR/calls"
printf '%s\n' "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_BIN_DIR" "$PIP_CACHE_DIR" "$HF_HOME" > "$UV_CACHE_DIR/paths"
""".replace("VERSION", version))

    def run_bootstrap(self, *arguments):
        return subprocess.run(["bash", str(self.project / "scripts/bootstrap_verl.sh"),
                               str(self.runtime), *arguments], env=self.env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_existing_runtime_requires_explicit_resume(self):
        self.partial()
        result = self.run_bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to overwrite", result.stderr)
        self.assertFalse((self.runtime / "verl/checked-source").exists())

    def test_resume_requires_existing_framework_and_rejects_completed_runtime(self):
        result = self.run_bootstrap("--resume")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.runtime.exists())
        self.runtime.mkdir()
        self.assertNotEqual(self.run_bootstrap("--resume").returncode, 0)
        self.partial()
        (self.runtime / "runtime_manifest.txt").write_text("completed fixture")
        result = self.run_bootstrap("--resume")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.runtime / "runtime_manifest.txt").read_text(), "completed fixture")

    def test_resume_uses_pinned_local_uv_without_system_ensurepip(self):
        self.partial()
        result = self.run_bootstrap("--resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.runtime / "verl/checked-source").exists())
        calls = (self.runtime / "cache/uv/calls").read_text()
        self.assertIn("sync --python 3.12.3 --frozen --all-packages --extra vllm --extra fsdp", calls)
        self.assertIn("numpy==2.3.5", calls)
        self.assertIn("pip check", calls)
        paths = (self.runtime / "cache/uv/paths").read_text().splitlines()
        self.assertEqual(paths, [str(self.runtime / suffix) for suffix in
                                ("python", "bin", "cache/pip", "cache/huggingface")])
        self.assertIn("runtime_override=numpy==2.3.5",
                      (self.runtime / "runtime_manifest.txt").read_text())

    def test_unexpected_local_uv_preserved_and_not_executed_for_sync(self):
        self.partial(version="0.0.0")
        result = self.run_bootstrap("--resume")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unexpected local uv version", result.stderr)
        self.assertFalse((self.runtime / "cache/uv/calls").exists())

    def test_failed_framework_check_stops_before_dependency_sync(self):
        self.partial()
        self.executable(self.project / "scripts/setup_verl.sh", "exit 23\n")
        result = self.run_bootstrap("--resume")
        self.assertEqual(result.returncode, 23)
        self.assertFalse((self.runtime / "cache/uv/calls").exists())

    def test_missing_ensurepip_reports_standalone_recovery_and_preserves_source(self):
        self.partial()
        (self.runtime / "bootstrap/bin/uv").unlink()
        result = self.run_bootstrap("--resume")
        self.assertEqual(result.returncode, 2)
        self.assertIn("lacks ensurepip", result.stderr)
        self.assertIn("--resume", result.stderr)
        self.assertTrue((self.runtime / "verl/checked-source").exists())
        self.assertFalse((self.runtime / "runtime_manifest.txt").exists())


if __name__ == "__main__":
    unittest.main()
