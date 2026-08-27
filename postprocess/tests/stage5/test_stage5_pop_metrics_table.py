import os
import statistics
import sys
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

pop_table = load_module_by_path("stage5_pop_metrics_table", "stage5", "stage5_pop_metrics_table.py")

import stage4_rocprofv3  # noqa: E402  (needs sys.path insert above first)

REF_DIR = os.path.join(FIXTURES, "pop_ref_2rank")
SCALED_DIR = os.path.join(FIXTURES, "pop_scaled_4rank")
COMBINED_DIR = os.path.join(FIXTURES, "pop_combined_2rank")
MISMATCH_DIR = os.path.join(FIXTURES, "pop_combined_mismatch")
GPU_ONLY_DIR = os.path.join(FIXTURES, "pop_gpu_only")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")
MPI_PREFIX_DIR = os.path.join(FIXTURES, "pop_mpi_prefix_reconciliation")
MPI_FORTRAN_SHIM_DIR = os.path.join(FIXTURES, "pop_mpi_fortran_shim_suffix")


class ComputeRunMetricsTests(unittest.TestCase):
    def test_per_rank_totals_match_fixture_arithmetic(self):
        # pop_ref_2rank/wall_clock-3001.txt: main(10.0, self 0) with children
        # compute_stencil(8.0, leaf), MPI_Isend(1.0, leaf), PMPI_Waitall(1.0,
        # self 50% = 0.5) -> MPIR_Typerep_icopy(0.5, leaf) as its one child.
        # comm_time = MPI_Isend(1.0) + PMPI_Waitall self(0.5) + MPIR_Typerep_icopy(0.5) = 2.0
        # useful_compute = total_time(10.0) - comm_time(2.0) = 8.0
        # wall_clock-3002.txt is the same shape scaled up: total 12.0, comm 3.0, useful 9.0.
        metrics = pop_table.compute_run_metrics(REF_DIR)
        self.assertEqual(metrics["num_ranks"], 2)
        self.assertIsNone(metrics["gpu_dir"])

        by_total = sorted(metrics["per_rank"], key=lambda r: r["total_time"])
        self.assertAlmostEqual(by_total[0]["total_time"], 10.0)
        self.assertAlmostEqual(by_total[0]["comm_time"], 2.0)
        self.assertAlmostEqual(by_total[0]["useful_compute"], 8.0)
        self.assertAlmostEqual(by_total[1]["total_time"], 12.0)
        self.assertAlmostEqual(by_total[1]["comm_time"], 3.0)
        self.assertAlmostEqual(by_total[1]["useful_compute"], 9.0)

    def test_load_balance_communication_and_parallel_efficiency(self):
        metrics = pop_table.compute_run_metrics(REF_DIR)
        useful = [8.0, 9.0]
        total = [10.0, 12.0]
        expected_lb = statistics.mean(useful) / max(useful)
        expected_comm_e = max(useful) / max(total)
        expected_pe = expected_lb * expected_comm_e

        self.assertAlmostEqual(metrics["load_balance"], expected_lb)
        self.assertAlmostEqual(metrics["communication_efficiency"], expected_comm_e)
        self.assertAlmostEqual(metrics["parallel_efficiency"], expected_pe)
        self.assertAlmostEqual(metrics["total_useful_compute"], sum(useful))
        self.assertAlmostEqual(metrics["avg_useful_compute"], statistics.mean(useful))

    def test_raises_on_completely_empty_input(self):
        with self.assertRaises(SystemExit):
            pop_table.compute_run_metrics(EMPTY_DIR)

    def test_raises_with_clear_message_on_gpu_only_input(self):
        with self.assertRaises(SystemExit) as ctx:
            pop_table.compute_run_metrics(GPU_ONLY_DIR)
        self.assertIn("CPU-side timing", str(ctx.exception))

    def test_gpu_extension_fields_are_none_without_gpu_pairing(self):
        metrics = pop_table.compute_run_metrics(REF_DIR)
        self.assertIsNone(metrics["gpu_offload_efficiency"])
        self.assertIsNone(metrics["gpu_utilization"])
        self.assertIsNone(metrics["gpu_load_balance"])
        self.assertIsNone(metrics["total_gpu_busy_time"])
        self.assertIsNone(metrics["avg_gpu_busy_time"])
        for r in metrics["per_rank"]:
            self.assertIsNone(r["cpu_only_time"])
            self.assertIsNone(r["gpu_busy_time"])


class GatherComputeSplitTests(unittest.TestCase):
    """compute_run_metrics() is gather_timing_summary_per_rank() + compute_metrics_from_per_rank()
    -- the point of the split (see stage5_pop_metrics_table.py's own docstrings) is that a new
    data source can call compute_metrics_from_per_rank() directly, without going through
    gather_timing_summary_per_rank()'s rocprof-sys-specific data collection at all. Confirmed here
    by calling it against a plain, hand-built per-rank list -- no fixture directory involved."""

    def test_gather_returns_the_same_per_rank_shape_compute_run_metrics_uses(self):
        per_rank, cpu_dir, gpu_dir, rank_keys = pop_table.gather_timing_summary_per_rank(REF_DIR)
        self.assertIsNone(gpu_dir)
        self.assertEqual(len(rank_keys), 2)
        self.assertEqual(len(per_rank), 2)
        for r in per_rank:
            self.assertEqual(
                set(r.keys()),
                {"rank_key", "total_time", "comm_time", "useful_compute", "cpu_only_time", "gpu_busy_time"},
            )
        self.assertTrue(cpu_dir)

    def test_compute_metrics_from_per_rank_works_without_any_gather_step(self):
        # Same two ranks as pop_ref_2rank's own arithmetic, built by hand instead of parsed --
        # compute_metrics_from_per_rank() has no idea (or opinion) where these numbers came from.
        per_rank = [
            {"rank_key": "r0", "total_time": 10.0, "comm_time": 2.0, "useful_compute": 8.0,
             "cpu_only_time": None, "gpu_busy_time": None},
            {"rank_key": "r1", "total_time": 12.0, "comm_time": 3.0, "useful_compute": 9.0,
             "cpu_only_time": None, "gpu_busy_time": None},
        ]
        metrics = pop_table.compute_metrics_from_per_rank(per_rank)
        useful = [8.0, 9.0]
        total = [10.0, 12.0]
        self.assertAlmostEqual(metrics["load_balance"], statistics.mean(useful) / max(useful))
        self.assertAlmostEqual(metrics["communication_efficiency"], max(useful) / max(total))
        self.assertIsNone(metrics["gpu_offload_efficiency"])
        self.assertIsNone(metrics["total_gpu_busy_time"])

    def test_compute_run_metrics_is_exactly_gather_then_compute(self):
        gathered_per_rank, cpu_dir, gpu_dir, rank_keys = pop_table.gather_timing_summary_per_rank(REF_DIR)
        expected = pop_table.compute_metrics_from_per_rank(gathered_per_rank)
        actual = pop_table.compute_run_metrics(REF_DIR)
        for key, value in expected.items():
            self.assertEqual(actual[key], value)
        self.assertEqual(actual["per_rank"], gathered_per_rank)
        self.assertEqual(actual["num_ranks"], len(rank_keys))
        self.assertEqual(actual["gpu_dir"], gpu_dir)


class MpiPrefixReconciliationTests(unittest.TestCase):
    def test_mpidi_and_differently_cased_labels_count_as_communication(self):
        # pop_mpi_prefix_reconciliation/wall_clock-4001.txt: main(10.0) with
        # compute_stencil(6.0, leaf), mpidi_cray_shm_coll(2.0, leaf), and
        # Mpi_Allreduce(2.0, leaf, mixed-case) -- comm_time = 2.0 + 2.0 = 4.0.
        metrics = pop_table.compute_run_metrics(MPI_PREFIX_DIR)
        self.assertEqual(len(metrics["per_rank"]), 1)
        rank = metrics["per_rank"][0]
        self.assertAlmostEqual(rank["total_time"], 10.0)
        self.assertAlmostEqual(rank["comm_time"], 4.0)
        self.assertAlmostEqual(rank["useful_compute"], 6.0)

    def test_fortran_shim_suffixed_labels_count_as_communication(self):
        # pop_mpi_fortran_shim_suffix/wall_clock-6001.txt: main(10.0) with
        # compute_stencil(7.0, leaf), mpi_isend_f08_(1.5, leaf), and
        # mpi_wait_f08ts_(1.5, leaf) -- comm_time = 1.5 + 1.5 = 3.0.
        metrics = pop_table.compute_run_metrics(MPI_FORTRAN_SHIM_DIR)
        self.assertEqual(len(metrics["per_rank"]), 1)
        rank = metrics["per_rank"][0]
        self.assertAlmostEqual(rank["total_time"], 10.0)
        self.assertAlmostEqual(rank["comm_time"], 3.0)
        self.assertAlmostEqual(rank["useful_compute"], 7.0)


class CombinedPoolTests(unittest.TestCase):
    def test_gpu_kernel_time_and_sync_wait_subtraction_applied_per_rank(self):
        # rocprof-sys/wall_clock-2001.txt: main(6.5) = compute_stencil(5.0) +
        # MPI_Isend(0.5) + hipStreamSynchronize(1.0). comm_time = 0.5 (only the
        # MPI_ label -- hipStreamSynchronize must NOT be misclassified as comm).
        # cpu_pure = max(0, 6.5 - 0.5 - 1.0) = 5.0; useful = cpu_pure + gpu kernel total.
        metrics = pop_table.compute_run_metrics(COMBINED_DIR)
        self.assertEqual(metrics["gpu_dir"], os.path.join(COMBINED_DIR, "rocprofv3"))

        gpu_per_rank, gpu_scanned = stage4_rocprofv3.aggregate_per_rank(os.path.join(COMBINED_DIR, "rocprofv3"))
        self.assertEqual(len(gpu_scanned), 2)
        gpu_totals_by_scan = sorted(sum(ft.values()) for ft in gpu_per_rank)

        by_total = sorted(metrics["per_rank"], key=lambda r: r["total_time"])
        self.assertAlmostEqual(by_total[0]["comm_time"], 0.5)  # MPI_Isend only
        self.assertAlmostEqual(by_total[1]["comm_time"], 0.6)
        expected_useful = sorted([
            max(0.0, 6.5 - 0.5 - 1.0) + gpu_totals_by_scan[0],
            max(0.0, 7.0 - 0.6 - 0.9) + gpu_totals_by_scan[1],
        ])
        actual_useful = sorted(r["useful_compute"] for r in metrics["per_rank"])
        for exp, act in zip(expected_useful, actual_useful):
            self.assertAlmostEqual(exp, act)

    def test_mismatched_rank_counts_falls_back_to_cpu_only(self):
        # 2 rocprof-sys ranks paired with only 1 rocprofv3 file -- must not
        # guess a pairing; falls back to the CPU-only pool instead.
        metrics = pop_table.compute_run_metrics(MISMATCH_DIR)
        self.assertIsNone(metrics["gpu_dir"])
        by_total = sorted(metrics["per_rank"], key=lambda r: r["total_time"])
        self.assertAlmostEqual(by_total[0]["useful_compute"], max(0.0, 6.5 - 0.5))  # no GPU term added
        self.assertAlmostEqual(by_total[1]["useful_compute"], max(0.0, 7.0 - 0.6))
        self.assertIsNone(metrics["gpu_offload_efficiency"])
        self.assertIsNone(metrics["gpu_utilization"])
        self.assertIsNone(metrics["gpu_load_balance"])

    def test_gpu_offload_efficiency_utilization_and_load_balance(self):
        # cpu_only_time (cpu_pure): rank2001 = max(0, 6.5-0.5-1.0) = 5.0,
        # rank2002 = max(0, 7.0-0.6-0.9) = 5.5. max_total = max(6.5, 7.0) = 7.0.
        # GPU-Off = 1 - max(5.0, 5.5) / 7.0.
        gpu_per_rank, _ = stage4_rocprofv3.aggregate_per_rank(os.path.join(COMBINED_DIR, "rocprofv3"))
        gpu_busy = sorted(sum(ft.values()) for ft in gpu_per_rank)

        metrics = pop_table.compute_run_metrics(COMBINED_DIR)
        expected_offload = 1.0 - max(5.0, 5.5) / 7.0
        expected_util = max(gpu_busy) / 7.0
        expected_lb = statistics.mean(gpu_busy) / max(gpu_busy)

        self.assertAlmostEqual(metrics["gpu_offload_efficiency"], expected_offload)
        self.assertAlmostEqual(metrics["gpu_utilization"], expected_util)
        self.assertAlmostEqual(metrics["gpu_load_balance"], expected_lb)
        self.assertAlmostEqual(metrics["total_gpu_busy_time"], sum(gpu_busy))
        self.assertAlmostEqual(metrics["avg_gpu_busy_time"], statistics.mean(gpu_busy))
        for r in metrics["per_rank"]:
            self.assertIsNotNone(r["cpu_only_time"])
            self.assertIsNotNone(r["gpu_busy_time"])


class ComputeScalingMetricsTests(unittest.TestCase):
    def setUp(self):
        self.ref_metrics = pop_table.compute_run_metrics(REF_DIR)
        self.scaled_metrics = pop_table.compute_run_metrics(SCALED_DIR)

    def test_strong_scaling_uses_total_useful_compute(self):
        ref_total = sum([8.0, 9.0])
        scaled_total = sum([4.0, 4.0, 4.5, 4.0])
        result = pop_table.compute_scaling_metrics(self.ref_metrics, self.scaled_metrics, "strong")
        self.assertAlmostEqual(result["computation_efficiency"], ref_total / scaled_total)
        self.assertAlmostEqual(
            result["global_efficiency"],
            self.scaled_metrics["parallel_efficiency"] * (ref_total / scaled_total),
        )

    def test_weak_scaling_uses_average_useful_compute(self):
        ref_avg = statistics.mean([8.0, 9.0])
        scaled_avg = statistics.mean([4.0, 4.0, 4.5, 4.0])
        result = pop_table.compute_scaling_metrics(self.ref_metrics, self.scaled_metrics, "weak")
        self.assertAlmostEqual(result["computation_efficiency"], ref_avg / scaled_avg)

    def test_strong_and_weak_give_different_computation_efficiency(self):
        strong = pop_table.compute_scaling_metrics(self.ref_metrics, self.scaled_metrics, "strong")
        weak = pop_table.compute_scaling_metrics(self.ref_metrics, self.scaled_metrics, "weak")
        self.assertNotAlmostEqual(strong["computation_efficiency"], weak["computation_efficiency"])

    def test_reference_compared_to_itself_is_trivially_one(self):
        result = pop_table.compute_scaling_metrics(self.ref_metrics, self.ref_metrics, "strong")
        self.assertAlmostEqual(result["computation_efficiency"], 1.0)
        self.assertAlmostEqual(result["global_efficiency"], self.ref_metrics["parallel_efficiency"])


class ComputeGpuEfficiencyTests(unittest.TestCase):
    REF = {"total_gpu_busy_time": 10.0, "avg_gpu_busy_time": 5.0}
    SCALED = {"total_gpu_busy_time": 16.0, "avg_gpu_busy_time": 4.0}
    NO_GPU = {"total_gpu_busy_time": None, "avg_gpu_busy_time": None}

    def test_strong_uses_total(self):
        result = pop_table.compute_gpu_efficiency(self.REF, self.SCALED, "strong")
        self.assertAlmostEqual(result, 10.0 / 16.0)

    def test_weak_uses_average(self):
        result = pop_table.compute_gpu_efficiency(self.REF, self.SCALED, "weak")
        self.assertAlmostEqual(result, 5.0 / 4.0)

    def test_strong_and_weak_differ(self):
        strong = pop_table.compute_gpu_efficiency(self.REF, self.SCALED, "strong")
        weak = pop_table.compute_gpu_efficiency(self.REF, self.SCALED, "weak")
        self.assertNotAlmostEqual(strong, weak)

    def test_reference_compared_to_itself_is_one(self):
        self.assertAlmostEqual(pop_table.compute_gpu_efficiency(self.REF, self.REF, "strong"), 1.0)

    def test_none_when_reference_lacks_gpu_data(self):
        self.assertIsNone(pop_table.compute_gpu_efficiency(self.NO_GPU, self.SCALED, "strong"))

    def test_none_when_scaled_run_lacks_gpu_data(self):
        self.assertIsNone(pop_table.compute_gpu_efficiency(self.REF, self.NO_GPU, "strong"))


class FormatMetricsTableTests(unittest.TestCase):
    """Synthetic metrics dicts rather than real fixtures -- format_metrics_table() only reads
    the fields listed below, so a real compute_run_metrics() run isn't needed to exercise its
    column-visibility branches in isolation."""

    def _metrics(self, run_dir, num_ranks=2, load_balance=0.9, comm_e=0.8, pe=0.72,
                 gpu_offload_efficiency=None, gpu_utilization=None, gpu_load_balance=None,
                 total_useful_compute=10.0, avg_useful_compute=5.0,
                 total_gpu_busy_time=None, avg_gpu_busy_time=None):
        return {
            "run_dir": run_dir, "num_ranks": num_ranks, "load_balance": load_balance,
            "communication_efficiency": comm_e, "parallel_efficiency": pe,
            "gpu_offload_efficiency": gpu_offload_efficiency, "gpu_utilization": gpu_utilization,
            "gpu_load_balance": gpu_load_balance, "total_useful_compute": total_useful_compute,
            "avg_useful_compute": avg_useful_compute, "total_gpu_busy_time": total_gpu_busy_time,
            "avg_gpu_busy_time": avg_gpu_busy_time,
        }

    def test_single_run_has_no_gpu_or_scaling_columns(self):
        all_metrics = [self._metrics("/runs/a")]
        table_text, show_gpu_cols, show_gpu_eff = pop_table.format_metrics_table(all_metrics)
        self.assertFalse(show_gpu_cols)
        self.assertFalse(show_gpu_eff)
        header = table_text.splitlines()[1]
        self.assertIn("LB", header)
        self.assertNotIn("GPU-Util", header)
        self.assertNotIn("CompE", header)

    def test_gpu_columns_shown_when_any_run_has_gpu_data(self):
        all_metrics = [self._metrics("/runs/a", gpu_offload_efficiency=0.5, gpu_utilization=0.4, gpu_load_balance=0.9)]
        table_text, show_gpu_cols, _show_gpu_eff = pop_table.format_metrics_table(all_metrics)
        self.assertTrue(show_gpu_cols)
        header = table_text.splitlines()[1]
        self.assertIn("GPU-Util", header)
        self.assertIn("GPU-Off", header)
        self.assertIn("GPU-LB", header)

    def test_multi_run_shows_scaling_columns_and_reference_label(self):
        all_metrics = [self._metrics("/runs/ref"), self._metrics("/runs/scaled", num_ranks=4)]
        table_text, _show_gpu_cols, _show_gpu_eff = pop_table.format_metrics_table(all_metrics, scaling="strong")
        self.assertIn("(reference)", table_text)
        header = table_text.splitlines()[1]
        self.assertIn("CompE", header)
        self.assertIn("GE", header)
        self.assertIn("strong scaling", table_text)

    def test_gpu_eff_only_when_multi_run_and_non_reference_has_gpu_data(self):
        ref = self._metrics("/runs/ref")
        scaled = self._metrics("/runs/scaled", total_gpu_busy_time=5.0, avg_gpu_busy_time=2.5)
        _table_text, _show_gpu_cols, show_gpu_eff = pop_table.format_metrics_table([ref, scaled], scaling="strong")
        self.assertFalse(show_gpu_eff)  # ref itself has no GPU busy time -> compute_gpu_efficiency is None

        ref_with_gpu = self._metrics("/runs/ref", total_gpu_busy_time=4.0, avg_gpu_busy_time=2.0)
        _table_text, _show_gpu_cols, show_gpu_eff = pop_table.format_metrics_table([ref_with_gpu, scaled], scaling="strong")
        self.assertTrue(show_gpu_eff)

    def test_column_order_matches_grouping(self):
        ref = self._metrics(
            "/runs/ref", gpu_offload_efficiency=0.5, gpu_utilization=0.4, gpu_load_balance=0.9,
            total_gpu_busy_time=4.0, avg_gpu_busy_time=2.0,
        )
        scaled = self._metrics(
            "/runs/scaled", gpu_offload_efficiency=0.5, gpu_utilization=0.4, gpu_load_balance=0.9,
            total_gpu_busy_time=5.0, avg_gpu_busy_time=2.5,
        )
        table_text, _show_gpu_cols, _show_gpu_eff = pop_table.format_metrics_table([ref, scaled], scaling="strong")
        header = table_text.splitlines()[1]
        columns = ["LB", "CommE", "PE", "GPU-Util", "GPU-Off", "GPU-LB", "CompE", "GE", "GPU-Eff"]
        positions = [header.index(c) for c in columns]
        self.assertEqual(positions, sorted(positions))


class MetricsLegendTests(unittest.TestCase):
    def test_base_metrics_always_present(self):
        legend = pop_table.metrics_legend(show_gpu_cols=False, show_gpu_eff=False, multi_run=False, scaling=None)
        self.assertIn("LB    =", legend)
        self.assertIn("CommE =", legend)
        self.assertIn("PE    =", legend)
        self.assertNotIn("GPU-Util", legend)
        self.assertIn("CompE, GE need a scaling study", legend)

    def test_gpu_columns_add_their_own_bullets(self):
        legend = pop_table.metrics_legend(show_gpu_cols=True, show_gpu_eff=False, multi_run=False, scaling=None)
        self.assertIn("GPU-Util =", legend)
        self.assertIn("GPU-Off =", legend)
        self.assertIn("GPU-LB =", legend)

    def test_multi_run_strong_vs_weak_wording(self):
        strong = pop_table.metrics_legend(show_gpu_cols=False, show_gpu_eff=False, multi_run=True, scaling="strong")
        self.assertIn("total useful compute time (reference)", strong)
        weak = pop_table.metrics_legend(show_gpu_cols=False, show_gpu_eff=False, multi_run=True, scaling="weak")
        self.assertIn("avg per-rank useful compute time (reference)", weak)

    def test_gpu_eff_bullet_only_when_flagged(self):
        legend = pop_table.metrics_legend(show_gpu_cols=True, show_gpu_eff=True, multi_run=True, scaling="strong")
        self.assertIn("GPU-Eff =", legend)


if __name__ == "__main__":
    unittest.main()
