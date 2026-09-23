import unittest

from probegrpo.advantage import (
    ProbeCredit,
    blend_probe_advantages,
    normalize_probe_deltas,
    standardize_probe_deltas,
)


class AdvantageTest(unittest.TestCase):
    def test_budget_zero_is_identical_to_grpo(self):
        base = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(blend_probe_advantages(base, [], 0.5), base)

    def test_only_selected_turn_tokens_change(self):
        base = [[0.0] * 5, [1.0] * 5]
        credits = [
            ProbeCredit(0, (1, 2), delta=1.0),
            ProbeCredit(1, (3,), delta=-1.0),
        ]
        shaped = blend_probe_advantages(base, credits, lambda_coef=0.5)
        self.assertEqual(shaped[0], [0.0, 0.5, 0.5, 0.0, 0.0])
        self.assertEqual(shaped[1], [1.0, 1.0, 1.0, 0.5, 1.0])

    def test_zero_delta_stays_zero_with_other_valid_probes(self):
        self.assertEqual(normalize_probe_deltas([1.0, 0.0]), [1.0, 0.0])

    def test_equal_nonzero_deltas_keep_causal_credit(self):
        self.assertEqual(normalize_probe_deltas([0.7, 0.7]), [0.7, 0.7])

    def test_all_zero_deltas_add_no_credit(self):
        self.assertEqual(normalize_probe_deltas([0.0, 0.0]), [0.0, 0.0])

    def test_preserves_sign_and_relative_magnitude(self):
        self.assertEqual(normalize_probe_deltas([1.0, -0.25, 0.0]), [1.0, -0.25, 0.0])

    def test_larger_reward_scale_is_bounded(self):
        self.assertEqual(normalize_probe_deltas([2.0, -1.0, 0.0]), [1.0, -0.5, 0.0])

    def test_single_small_delta_keeps_its_magnitude(self):
        self.assertEqual(standardize_probe_deltas([-0.3]), [-0.3])

    def test_zero_delta_does_not_change_any_token(self):
        base = [[0.0, 0.0], [0.0, 0.0]]
        credits = [ProbeCredit(0, (1,), 1.0), ProbeCredit(1, (0,), 0.0)]
        self.assertEqual(blend_probe_advantages(base, credits), [[0.0, 0.5], [0.0, 0.0]])

    def test_nonfinite_delta_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_probe_deltas([float("nan")])

    def test_invalid_credit_is_ignored(self):
        base = [[0.0, 0.0]]
        credit = ProbeCredit(0, (1,), delta=1.0, valid=False)
        self.assertEqual(blend_probe_advantages(base, [credit]), base)

    def test_invalid_token_index_is_rejected(self):
        with self.assertRaises(IndexError):
            blend_probe_advantages([[0.0]], [ProbeCredit(0, (3,), 1.0)])


if __name__ == "__main__":
    unittest.main()
