"""Tests for stage4_rank_merge_math.py's cross-rank stats primitive (stats_across_ranks()) --
average/std_dev/min/max over a list of per-rank numeric values, including the empty-list,
single-value, and explicit-zero-for-a-missing-rank edge cases.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

rmm = load_module_by_path("stage4_rank_merge_math", "stage4", "stage4_rank_merge_math.py")


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
