import importlib.util
import os
import sys
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "stage4", "stage4_rocprofsys_trace_flat.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("stage4_rocprofsys_trace_flat", MODULE_PATH)
flat = importlib.util.module_from_spec(spec)
sys.modules["stage4_rocprofsys_trace_flat"] = flat
spec.loader.exec_module(flat)

from stage5_load_imbalance_table import compute_load_imbalance  # noqa: E402
from stage5_pop_metrics_table import compute_metrics_from_per_rank  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
TWO_RANK_DIR = os.path.join(FIXTURES, "trace_two_rank")
RANK_INPUTS = [
    ("r0", os.path.join(TWO_RANK_DIR, "rank0.csv")),
    ("r1", os.path.join(TWO_RANK_DIR, "rank1.csv")),
]


def by_label(entries, label):
    return next(e for e in entries if e["label"] == label)


class AggregateTests(unittest.TestCase):
    def test_total_runtime_is_the_sum_of_each_ranks_root_sum(self):
        _entries, total_runtime = flat.aggregate(RANK_INPUTS)
        self.assertAlmostEqual(total_runtime, 22.0)  # rank0 main.sum=10.0 + rank1 main.sum=12.0

    def test_counts_and_sums_add_up_across_ranks(self):
        entries, _total_runtime = flat.aggregate(RANK_INPUTS)
        kernel = by_label(entries, "jacobi_kernel.kd")
        self.assertEqual(kernel["count"], 2)
        self.assertAlmostEqual(kernel["self_sum"], 7.0)  # 3.0 (r0) + 4.0 (r1)
        self.assertAlmostEqual(kernel["sum"], 7.0)

    def test_no_label_is_dropped(self):
        entries, _total_runtime = flat.aggregate(RANK_INPUTS)
        self.assertEqual(
            {e["label"] for e in entries},
            {"main", "jacobi_sweep", "hipLaunchKernel", "MPI_Barrier", "jacobi_kernel.kd"},
        )


class AggregatePerRankTests(unittest.TestCase):
    def test_self_sum_by_default(self):
        per_rank, rank_keys = flat.aggregate_per_rank(RANK_INPUTS)
        self.assertEqual(rank_keys, ["r0", "r1"])
        self.assertAlmostEqual(per_rank[0]["jacobi_sweep"], 6.5)
        self.assertAlmostEqual(per_rank[1]["jacobi_sweep"], 6.4)

    def test_unfiltered_switches_to_inclusive_sum(self):
        per_rank, _rank_keys = flat.aggregate_per_rank(RANK_INPUTS, unfiltered=True)
        self.assertAlmostEqual(per_rank[0]["jacobi_sweep"], 8.0)
        self.assertAlmostEqual(per_rank[1]["jacobi_sweep"], 9.0)

    def test_feeds_compute_load_imbalance_unmodified(self):
        per_rank, _rank_keys = flat.aggregate_per_rank(RANK_INPUTS)
        selected, _description = compute_load_imbalance(per_rank)
        self.assertEqual(len(selected), 5)


class GatherTimingSummaryPerRankTests(unittest.TestCase):
    def test_per_rank_values(self):
        summary = flat.gather_timing_summary_per_rank(RANK_INPUTS)
        r0, r1 = summary
        self.assertEqual(r0["rank_key"], "r0")
        self.assertAlmostEqual(r0["total_time"], 10.0)
        self.assertAlmostEqual(r0["comm_time"], 1.0)
        self.assertAlmostEqual(r0["gpu_busy_time"], 3.0)
        self.assertAlmostEqual(r0["cpu_only_time"], 6.0)
        self.assertAlmostEqual(r0["useful_compute"], 9.0)
        self.assertAlmostEqual(r1["total_time"], 12.0)
        self.assertAlmostEqual(r1["gpu_busy_time"], 4.0)

    def test_gpu_busy_time_is_never_none(self):
        # Unlike the sample pipeline's paired-rocprofv3 case, GPU visibility is always part of
        # the same trace -- gpu_busy_time is a real number even when it happens to be 0.0.
        summary = flat.gather_timing_summary_per_rank(RANK_INPUTS)
        for entry in summary:
            self.assertIsNotNone(entry["gpu_busy_time"])
            self.assertIsNotNone(entry["cpu_only_time"])

    def test_feeds_compute_metrics_from_per_rank_unmodified(self):
        summary = flat.gather_timing_summary_per_rank(RANK_INPUTS)
        metrics = compute_metrics_from_per_rank(summary)
        self.assertAlmostEqual(metrics["load_balance"], 19.0 / 2 / 10.0)
        self.assertIsNotNone(metrics["gpu_utilization"])


if __name__ == "__main__":
    unittest.main()
