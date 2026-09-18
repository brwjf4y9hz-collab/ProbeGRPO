import unittest

from probegrpo.integration import ragen, verl


class VerlIntegrationTest(unittest.TestCase):
    def test_legacy_ragen_import_reexports_verl_adapter(self) -> None:
        self.assertIs(ragen.apply_probe_credits_tensor, verl.apply_probe_credits_tensor)
        self.assertIs(ragen.apply_to_dataproto, verl.apply_to_dataproto)


if __name__ == "__main__":
    unittest.main()
