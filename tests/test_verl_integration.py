"""本文件用自动化断言锁定 verl integration 相关组件的输入约定、边界条件和预期行为；测试数据为验证样例，不代表正式实验结果。"""

import unittest

from probegrpo.integration import ragen, verl


class VerlIntegrationTest(unittest.TestCase):
    def test_legacy_ragen_import_reexports_verl_adapter(self) -> None:
        self.assertIs(ragen.apply_probe_credits_tensor, verl.apply_probe_credits_tensor)
        self.assertIs(ragen.apply_to_dataproto, verl.apply_to_dataproto)


if __name__ == "__main__":
    unittest.main()
