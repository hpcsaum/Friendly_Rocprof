"""Tests for stage4_rocprofsys_trace_ranks.py's trace-CSV rank discovery (discover_ranks()) --
numeric rank-key sorting, preferring a rank's partitioned (gpu/mpi/other) file set over its
unfiltered file when both exist, falling back to whichever form is present, and raising when a
directory has no matching files at all.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

ranks_mod = load_module_by_path("stage4_rocprofsys_trace_ranks", "stage4", "stage4_rocprofsys_trace_ranks.py")

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
MIXED_DIR = os.path.join(FIXTURES, "trace_discovery_mixed")


class DiscoverRanksTests(unittest.TestCase):
    def test_returns_one_entry_per_rank_sorted_numerically(self):
        result = ranks_mod.discover_ranks(MIXED_DIR)
        self.assertEqual([rank_key for rank_key, _paths in result], ["0", "1", "2"])

    def test_partitioned_set_wins_when_both_forms_exist(self):
        # Confirmed against this project's own real trace-CSV export: the unfiltered file isn't
        # guaranteed to carry the wide GPU-arg columns (corr_id, grid_size, ...) the partitioned
        # files do -- preferring the partitioned set avoids silently losing corr_id-join data
        # purely because of which file happened to exist.
        result = dict(ranks_mod.discover_ranks(MIXED_DIR))
        self.assertEqual(
            result["1"],
            sorted([
                os.path.join(MIXED_DIR, "run-1-gpu.csv"),
                os.path.join(MIXED_DIR, "run-1-mpi.csv"),
                os.path.join(MIXED_DIR, "run-1-other.csv"),
            ]),
        )

    def test_rank_with_only_unfiltered_file_returns_a_single_path(self):
        result = dict(ranks_mod.discover_ranks(MIXED_DIR))
        self.assertEqual(result["0"], os.path.join(MIXED_DIR, "run-0.csv"))

    def test_partitioned_set_used_when_no_unfiltered_file_exists(self):
        result = dict(ranks_mod.discover_ranks(MIXED_DIR))
        self.assertEqual(
            result["2"],
            sorted([
                os.path.join(MIXED_DIR, "run-2-gpu.csv"),
                os.path.join(MIXED_DIR, "run-2-mpi.csv"),
            ]),
        )

    def test_numeric_sort_order_for_ten_plus_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in (1, 2, 10, 11):
                open(os.path.join(tmp, f"run-{n}.csv"), "w").close()
            result = ranks_mod.discover_ranks(tmp)
            self.assertEqual([rank_key for rank_key, _paths in result], ["1", "2", "10", "11"])

    def test_no_matching_files_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "unrelated.csv"), "w").close()
            with self.assertRaises(SystemExit):
                ranks_mod.discover_ranks(tmp)


if __name__ == "__main__":
    unittest.main()
