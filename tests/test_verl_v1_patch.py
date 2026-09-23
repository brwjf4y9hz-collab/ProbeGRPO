import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class VerlV1PatchTest(unittest.TestCase):
    def test_injects_once_and_keeps_original(self):
        script = Path(__file__).resolve().parents[1] / "scripts/install_verl_v1_probe_hook.py"
        namespace = runpy.run_path(str(script))
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            target = checkout / namespace["TARGET"]
            target.parent.mkdir(parents=True)
            original = "def update(self):\n" + namespace["MARKER"] + "        return self\n"
            target.write_text(original)
            with patch.object(
                namespace["subprocess"],
                "check_output",
                side_effect=[namespace["VERL_COMMIT"] + "\n", "", namespace["VERL_COMMIT"] + "\n"],
            ):
                self.assertIn("installed", namespace["install"](checkout))
                self.assertEqual(namespace["install"](checkout), "already installed")
            self.assertEqual(target.read_text().count("apply_sidecar_probe_credits"), 2)
            self.assertEqual(target.with_suffix(".py.probegrpo.backup").read_text(), original)


if __name__ == "__main__":
    unittest.main()
