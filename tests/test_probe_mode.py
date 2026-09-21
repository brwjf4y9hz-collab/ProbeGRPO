import unittest

from probegrpo.probe_mode import resolve_probe_mode


class ProbeModeTest(unittest.TestCase):
    def test_disabled_baseline_spends_no_probe_rollouts(self):
        for session_id in range(4):
            mode = resolve_probe_mode({"enabled": False}, session_id=session_id)
            self.assertFalse(mode.run_probe)
            self.assertFalse(mode.credit_requested)

    def test_budget_zero_overrides_debug_switch_and_spends_nothing(self):
        mode = resolve_probe_mode(
            {"enabled": True, "budget": 0, "scheduler": "random"},
            session_id=0,
            debug_probe=True,
        )
        self.assertFalse(mode.run_probe)
        self.assertFalse(mode.credit_requested)

    def test_budget_one_runs_once_per_four_trajectory_group(self):
        modes = [
            resolve_probe_mode(
                {"enabled": True, "budget": 1, "scheduler": "random"},
                session_id=index,
            )
            for index in range(4)
        ]
        self.assertEqual([mode.run_probe for mode in modes], [True, False, False, False])
        self.assertTrue(all(mode.credit_requested for mode in modes))
        self.assertTrue(all(mode.scheduler == "random" for mode in modes))

    def test_budget_two_runs_first_two_sessions_for_supported_schedulers(self):
        for scheduler in ("random", "surprisal", "linear_ucb"):
            modes = [
                resolve_probe_mode(
                    {"enabled": True, "budget": 2, "scheduler": scheduler},
                    session_id=index,
                )
                for index in range(4)
            ]
            self.assertEqual([mode.run_probe for mode in modes], [True, True, False, False])
            self.assertTrue(all(mode.group_budget == 2 for mode in modes))

    def test_debug_probe_preserves_rollout_only_gate(self):
        mode = resolve_probe_mode({}, session_id=0, debug_probe=True)
        self.assertTrue(mode.run_probe)
        self.assertFalse(mode.credit_requested)
        self.assertEqual(mode.scheduler, "random_debug")

    def test_unsupported_budget_and_scheduler_fail_before_generation(self):
        for settings in (
            {"enabled": True, "budget": 5},
            {"enabled": True, "budget": -1},
            {"enabled": True, "budget": 1.0},
            {"enabled": True, "scheduler": "entropy"},
            {"enabled": "true"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                resolve_probe_mode(settings, session_id=0)


if __name__ == "__main__":
    unittest.main()
