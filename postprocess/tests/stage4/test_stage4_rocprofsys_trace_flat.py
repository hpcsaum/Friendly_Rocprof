"""Tests for stage4_rocprofsys_trace_flat.py's cross-rank flat/per-rank derivations for the
Perfetto trace-CSV pipeline.

AggregateTests                   -- global by-label totals across ranks, total_runtime, CPU/GPU domain tagging
AggregatePerRankTests            -- per-rank {label: value} breakdown, GPU rows included, self vs unfiltered time
GatherTimingSummaryPerRankTests  -- per-rank total/comm/gpu_busy/cpu_only/useful-compute timing summary
AggregateGpuKernelsTests         -- kernel-only aggregation excluding CPU and GPU-API (launch-call) labels, pct_total against kernel-only time
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

flat = load_module_by_path("stage4_rocprofsys_trace_flat", "stage4", "stage4_rocprofsys_trace_flat.py")

from stage5_load_imbalance_table import compute_load_imbalance  # noqa: E402
from stage5_pop_metrics_table import compute_metrics_from_per_rank  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
TWO_RANK_DIR = os.path.join(FIXTURES, "trace_two_rank")
RANK_INPUTS = [
    ("r0", os.path.join(TWO_RANK_DIR, "rank0.csv")),
    ("r1", os.path.join(TWO_RANK_DIR, "rank1.csv")),
]

KERNEL_SELECTION_DIR = os.path.join(FIXTURES, "trace_gpu_kernel_selection")
KERNEL_SELECTION_RANK_INPUTS = [
    ("r0", os.path.join(KERNEL_SELECTION_DIR, "perfetto-trace-0.csv")),
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

    def test_domain_is_gpu_for_gpu_tagged_labels_cpu_otherwise(self):
        entries, _total_runtime = flat.aggregate(RANK_INPUTS)
        self.assertEqual(by_label(entries, "hipLaunchKernel")["domain"], "GPU")
        self.assertEqual(by_label(entries, "jacobi_kernel.kd")["domain"], "GPU")
        self.assertEqual(by_label(entries, "main")["domain"], "CPU")
        self.assertEqual(by_label(entries, "jacobi_sweep")["domain"], "CPU")
        self.assertEqual(by_label(entries, "MPI_Barrier")["domain"], "CPU")


class AggregatePerRankTests(unittest.TestCase):
    def test_gpu_labels_are_included_not_excluded(self):
        # Unlike the sample pipeline's own aggregate_per_rank() (which drops GPU rows because
        # rocprofv3's data has no compatible per-rank breakdown to fold in), trace data gives
        # real per-rank GPU timing -- GPU load imbalance belongs in the same breakdown, not
        # excluded to match an older data source's limitation.
        per_rank, _rank_keys = flat.aggregate_per_rank(RANK_INPUTS)
        self.assertIn("hipLaunchKernel", per_rank[0])
        self.assertIn("jacobi_kernel.kd", per_rank[0])
        self.assertAlmostEqual(per_rank[0]["jacobi_kernel.kd"], 3.0)

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


class AggregateGpuKernelsTests(unittest.TestCase):
    def test_only_gpu_kernel_tagged_labels_are_included(self):
        entries, _total = flat.aggregate_gpu_kernels(KERNEL_SELECTION_RANK_INPUTS)
        self.assertEqual(
            {e["label"] for e in entries}, {"kernel_a.kd", "kernel_b.kd", "kernel_c.kd"}
        )

    def test_cpu_and_gpu_api_labels_are_excluded_even_when_bigger(self):
        # cpu_heavy_function's self_sum (60.0) dwarfs every kernel's, and hipLaunchKernel is
        # domain-GPU in aggregate() despite being a launch call, not a dispatch -- neither may
        # leak into a kernel-only result.
        entries, _total = flat.aggregate_gpu_kernels(KERNEL_SELECTION_RANK_INPUTS)
        labels = {e["label"] for e in entries}
        self.assertNotIn("cpu_heavy_function", labels)
        self.assertNotIn("hipLaunchKernel", labels)
        self.assertNotIn("main", labels)

    def test_counts_and_sums_add_up_per_kernel(self):
        entries, _total = flat.aggregate_gpu_kernels(KERNEL_SELECTION_RANK_INPUTS)
        kernel_a = by_label(entries, "kernel_a.kd")
        self.assertEqual(kernel_a["count"], 2)
        self.assertAlmostEqual(kernel_a["self_sum"], 20.0)
        kernel_b = by_label(entries, "kernel_b.kd")
        self.assertEqual(kernel_b["count"], 1)
        self.assertAlmostEqual(kernel_b["self_sum"], 5.0)

    def test_pct_total_is_against_kernel_only_time_not_app_runtime(self):
        # Total kernel-only time is 20 (kernel_a) + 5 (kernel_b) + 2 (kernel_c) = 27, NOT the
        # ~100s the app's own "main" root spans -- ranking kernels against each other, not
        # diluted by CPU-side time.
        entries, total_kernel_time = flat.aggregate_gpu_kernels(KERNEL_SELECTION_RANK_INPUTS)
        self.assertAlmostEqual(total_kernel_time, 27.0)
        kernel_a = by_label(entries, "kernel_a.kd")
        self.assertAlmostEqual(kernel_a["pct_total"], 20.0 / 27.0 * 100.0)


if __name__ == "__main__":
    unittest.main()
