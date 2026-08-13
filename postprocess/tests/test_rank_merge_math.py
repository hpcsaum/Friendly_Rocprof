import importlib.util
import os
import sys
import unittest

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "rank_merge_math.py")

spec = importlib.util.spec_from_file_location("rank_merge_math", MODULE_PATH)
rmm = importlib.util.module_from_spec(spec)
sys.modules["rank_merge_math"] = rmm
spec.loader.exec_module(rmm)


class StatsAcrossRanksTests(unittest.TestCase):
    def test_empty_list_returns_all_zeros(self):
        stats = rmm.stats_across_ranks([])
        self.assertEqual(stats, {"avg": 0.0, "std_dev": 0.0, "min": 0.0, "max": 0.0})

    def test_normal_values_match_hand_computed_stats(self):
        stats = rmm.stats_across_ranks([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(stats["avg"], 2.5)
        self.assertAlmostEqual(stats["std_dev"], 1.1180339887498949)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 4.0)

    def test_single_value_has_zero_std_dev(self):
        stats = rmm.stats_across_ranks([7.0])
        self.assertEqual(stats["avg"], 7.0)
        self.assertEqual(stats["std_dev"], 0.0)
        self.assertEqual(stats["min"], 7.0)
        self.assertEqual(stats["max"], 7.0)

    def test_missing_rank_as_explicit_zero_pulls_avg_down(self):
        # Mirrors how callers build this list: a rank absent from the source data
        # contributes an explicit 0.0 entry, not a shorter list.
        stats = rmm.stats_across_ranks([10.0, 0.0])
        self.assertAlmostEqual(stats["avg"], 5.0)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 10.0)


if __name__ == "__main__":
    unittest.main()
