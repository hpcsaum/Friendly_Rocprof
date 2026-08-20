import importlib.util
import os
import sys
import tempfile
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "stage4", "stage4_rocprofsys_trace_ranks.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("stage4_rocprofsys_trace_ranks", MODULE_PATH)
ranks_mod = importlib.util.module_from_spec(spec)
sys.modules["stage4_rocprofsys_trace_ranks"] = ranks_mod
spec.loader.exec_module(ranks_mod)

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
