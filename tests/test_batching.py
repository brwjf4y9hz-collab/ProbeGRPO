import unittest

from probegrpo.advantage import ProbeCredit
from probegrpo.batching import blend_packed_probe_advantages, pack_probe_credits


class ProbeBatchingTest(unittest.TestCase):
    def test_packs_masks_deltas_validity_and_padding(self):
        packed = pack_probe_credits(
            [
                ProbeCredit(0, (1, 2), delta=1.0),
                ProbeCredit(1, (3,), delta=-0.5),
                ProbeCredit(0, (4,), delta=0.25),
            ],
            batch_size=2,
            token_count=5,
            anchors_per_sample=2,
        )

        self.assertEqual(packed.mask_shape, (2, 2, 5))
        self.assertEqual(packed.matrix_shape, (2, 2))
        self.assertEqual(packed.probe_deltas, ((1.0, 0.25), (-0.5, 0.0)))
        self.assertEqual(packed.probe_valid, ((True, True), (True, False)))
        self.assertEqual(packed.probe_turn_masks[0][0], (False, True, True, False, False))
        self.assertEqual(packed.probe_turn_masks[1][1], (False, False, False, False, False))

    def test_invalid_credit_is_ignored(self):
        packed = pack_probe_credits(
            [ProbeCredit(0, (1,), delta=1.0, valid=False)],
            batch_size=1,
            token_count=3,
            anchors_per_sample=1,
        )
        self.assertEqual(packed.probe_valid, ((False,),))
        self.assertEqual(packed.probe_deltas, ((0.0,),))
        self.assertFalse(any(packed.probe_turn_masks[0][0]))

    def test_rejects_overflow_out_of_range_and_overlap(self):
        with self.assertRaisesRegex(ValueError, "more valid credits"):
            pack_probe_credits(
                [ProbeCredit(0, (0,), 1.0), ProbeCredit(0, (1,), -1.0)],
                batch_size=1,
                token_count=2,
                anchors_per_sample=1,
            )
        with self.assertRaisesRegex(IndexError, "token index"):
            pack_probe_credits(
                [ProbeCredit(0, (2,), 1.0)],
                batch_size=1,
                token_count=2,
                anchors_per_sample=1,
            )
        with self.assertRaisesRegex(ValueError, "multiple anchors"):
            pack_probe_credits(
                [ProbeCredit(0, (0,), 1.0), ProbeCredit(0, (0,), -1.0)],
                batch_size=1,
                token_count=2,
                anchors_per_sample=2,
            )

    def test_budget_zero_preserves_standard_grpo(self):
        packed = pack_probe_credits(
            [],
            batch_size=2,
            token_count=2,
            anchors_per_sample=0,
        )
        base = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(blend_packed_probe_advantages(base, packed), base)

    def test_dense_round_trip_modifies_only_masked_tokens(self):
        packed = pack_probe_credits(
            [ProbeCredit(0, (1, 2), delta=1.0), ProbeCredit(1, (3,), delta=-1.0)],
            batch_size=2,
            token_count=5,
            anchors_per_sample=1,
        )
        shaped = blend_packed_probe_advantages(
            [[0.0] * 5, [1.0] * 5],
            packed,
            lambda_coef=0.5,
        )
        self.assertEqual(shaped[0], [0.0, 0.5, 0.5, 0.0, 0.0])
        self.assertEqual(shaped[1], [1.0, 1.0, 1.0, 0.5, 1.0])
    def test_lambda_zero_ignores_valid_probe(self):
        packed = pack_probe_credits(
            [ProbeCredit(0, (1,), delta=1.0)],
            batch_size=1,
            token_count=3,
            anchors_per_sample=1,
        )
        base = [[0.2, 0.2, 0.2]]

        shaped = blend_packed_probe_advantages(
            base,
            packed,
            lambda_coef=0.0,
            )

        self.assertEqual(shaped, base)


if __name__ == "__main__":
    unittest.main()
