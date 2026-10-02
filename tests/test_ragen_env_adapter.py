"""本文件用自动化断言锁定 ragen env adapter 相关组件的输入约定、边界条件和预期行为；测试数据为验证样例，不代表正式实验结果。"""

import unittest

from probegrpo.integration.ragen_env import RagenReplayableEnv


class FakeRagenEnv:
    def __init__(self):
        self.position = 0
        self.steps = 0

    def reset(self, seed=None, mode="train"):
        self.position = int(seed) % 2
        self.steps = 0
        return f"position={self.position};mode={mode}"

    def step(self, action):
        self.position += int(action)
        self.steps += 1
        done = self.steps >= 2
        return f"position={self.position}", float(self.position == 3), done, {"valid": True}

    def get_all_actions(self):
        return [1, 2]


class RagenEnvAdapterTest(unittest.TestCase):
    def test_adapter_replays_prefix(self):
        adapter = RagenReplayableEnv(FakeRagenEnv)
        first = adapter.replay("fake", seed=2, action_prefix=("1",))
        second = adapter.replay("fake", seed=2, action_prefix=("1",))
        self.assertEqual(first.state_hash, second.state_hash)
        self.assertEqual(first.observation, second.observation)
        self.assertEqual(first.legal_actions, ("1", "2"))


if __name__ == "__main__":
    unittest.main()

