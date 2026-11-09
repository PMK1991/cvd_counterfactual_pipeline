import unittest

from scripts.report_simulation_diagnostics import acceptance_status, describe, extreme_counts


class CompactReportTests(unittest.TestCase):
    def test_record_delta_statistics_keep_signed_changes_and_max_absolute(self):
        result = describe([-.2, 0, .1, .3])
        self.assertAlmostEqual(result["mean"], .05)
        self.assertEqual(result["max_absolute"], .3)
        self.assertEqual(result["q00"], -.2)
        self.assertEqual(result["q50"], .05)
        self.assertEqual(result["q100"], .3)
        self.assertEqual(result["negative_records"], 1)
        self.assertEqual(result["zero_records"], 1)
        self.assertEqual(result["positive_records"], 2)
        self.assertEqual(result["n_records"], 4)

    def test_never_always_and_switches_use_inclusive_historical_threshold(self):
        self.assertEqual(acceptance_status([0, .49]), "never")
        self.assertEqual(acceptance_status([.5, 1]), "always")
        self.assertEqual(acceptance_status([.49, .5]), "switches")
        with self.assertRaises(ValueError):
            acceptance_status([])
        with self.assertRaises(ValueError):
            acceptance_status([float("nan")])

    def test_exact_zero_one_counts_do_not_round_near_extremes(self):
        self.assertEqual(
            extreme_counts([0, 1, 1e-16, 1-1e-15, .5]),
            {"p0_exactly_zero": 1, "p0_exactly_one": 1},
        )

    def test_identity_delta_is_exactly_zero_for_all_records(self):
        result = describe([0] * 48)
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["max_absolute"], 0)
        self.assertEqual(result["zero_records"], 48)


if __name__ == "__main__":
    unittest.main()
