import importlib.util
import os
import statistics
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "extract_pop_metrics.py")

# extract_pop_metrics.py does a plain top-level "import extract_CPU_hotspots"/
# "import extract_GPU_hotspots", relying on its own directory being on sys.path --
# true automatically when run directly, but not when loaded here by explicit
# file path, so replicate that manually (same as test_extract_hotspots.py).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("extract_pop_metrics", MODULE_PATH)
pop_tool = importlib.util.module_from_spec(spec)
sys.modules["extract_pop_metrics"] = pop_tool
spec.loader.exec_module(pop_tool)

import extract_CPU_hotspots as cpu_tool
import extract_GPU_hotspots as gpu_tool

REF_DIR = os.path.join(FIXTURES, "pop_ref_2rank")
SCALED_DIR = os.path.join(FIXTURES, "pop_scaled_4rank")
COMBINED_DIR = os.path.join(FIXTURES, "pop_combined_2rank")
MISMATCH_DIR = os.path.join(FIXTURES, "pop_combined_mismatch")
GPU_ONLY_DIR = os.path.join(FIXTURES, "pop_gpu_only")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")
FLAT_LAYOUT_DIR = os.path.join(FIXTURES, "mpi_2rank")  # no rocprof-sys/ subdir, files directly in the dir


class ResolveRunDirsTests(unittest.TestCase):
    def test_detects_paired_subdirs(self):
        cpu_dir, gpu_dir = pop_tool.resolve_run_dirs(COMBINED_DIR)
        self.assertEqual(cpu_dir, os.path.join(COMBINED_DIR, "rocprof-sys"))
        self.assertEqual(gpu_dir, os.path.join(COMBINED_DIR, "rocprofv3"))

    def test_falls_back_to_run_dir_itself_when_no_rocprof_sys_subdir(self):
        # mpi_2rank's wall_clock-*.txt files sit directly in the fixture dir --
        # the tool-1-alone layout (no rocprof-sys/ nesting).
        cpu_dir, gpu_dir = pop_tool.resolve_run_dirs(FLAT_LAYOUT_DIR)
        self.assertEqual(cpu_dir, FLAT_LAYOUT_DIR)
        self.assertIsNone(gpu_dir)

    def test_cpu_only_subdir_with_no_gpu_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "rocprof-sys"))
            cpu_dir, gpu_dir = pop_tool.resolve_run_dirs(tmp)
            self.assertEqual(cpu_dir, os.path.join(tmp, "rocprof-sys"))
            self.assertIsNone(gpu_dir)


class ComputeRunMetricsTests(unittest.TestCase):
    def test_per_rank_totals_match_fixture_arithmetic(self):
        # pop_ref_2rank/wall_clock-3001.txt: main(10.0, self 0) with children
        # compute_stencil(8.0, leaf), MPI_Isend(1.0, leaf), PMPI_Waitall(1.0,
        # self 50% = 0.5) -> MPIR_Typerep_icopy(0.5, leaf) as its one child.
        # comm_time = MPI_Isend(1.0) + PMPI_Waitall self(0.5) + MPIR_Typerep_icopy(0.5) = 2.0
        # useful_compute = total_time(10.0) - comm_time(2.0) = 8.0
        # wall_clock-3002.txt is the same shape scaled up: total 12.0, comm 3.0, useful 9.0.
        metrics = pop_tool.compute_run_metrics(REF_DIR)
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
        metrics = pop_tool.compute_run_metrics(REF_DIR)
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
            pop_tool.compute_run_metrics(EMPTY_DIR)

    def test_raises_with_clear_message_on_gpu_only_input(self):
        with self.assertRaises(SystemExit) as ctx:
            pop_tool.compute_run_metrics(GPU_ONLY_DIR)
        self.assertIn("CPU-side timing", str(ctx.exception))


class CombinedPoolTests(unittest.TestCase):
    def test_gpu_kernel_time_and_sync_wait_subtraction_applied_per_rank(self):
        # rocprof-sys/wall_clock-2001.txt: main(6.5) = compute_stencil(5.0) +
        # MPI_Isend(0.5) + hipStreamSynchronize(1.0). comm_time = 0.5 (only the
        # MPI_ label -- hipStreamSynchronize must NOT be misclassified as comm).
        # cpu_pure = max(0, 6.5 - 0.5 - 1.0) = 5.0; useful = cpu_pure + gpu kernel total.
        metrics = pop_tool.compute_run_metrics(COMBINED_DIR)
        self.assertEqual(metrics["gpu_dir"], os.path.join(COMBINED_DIR, "rocprofv3"))

        gpu_per_rank, gpu_scanned = gpu_tool.aggregate_per_rank(os.path.join(COMBINED_DIR, "rocprofv3"))
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
        metrics = pop_tool.compute_run_metrics(MISMATCH_DIR)
        self.assertIsNone(metrics["gpu_dir"])
        by_total = sorted(metrics["per_rank"], key=lambda r: r["total_time"])
        self.assertAlmostEqual(by_total[0]["useful_compute"], max(0.0, 6.5 - 0.5))  # no GPU term added
        self.assertAlmostEqual(by_total[1]["useful_compute"], max(0.0, 7.0 - 0.6))


class ComputeScalingMetricsTests(unittest.TestCase):
    def setUp(self):
        self.ref_metrics = pop_tool.compute_run_metrics(REF_DIR)
        self.scaled_metrics = pop_tool.compute_run_metrics(SCALED_DIR)

    def test_strong_scaling_uses_total_useful_compute(self):
        # Independently recompute from the fixtures' own known per-rank useful
        # compute values rather than hand-typing the ratio.
        ref_total = sum([8.0, 9.0])
        scaled_total = sum([4.0, 4.0, 4.5, 4.0])
        result = pop_tool.compute_scaling_metrics(self.ref_metrics, self.scaled_metrics, "strong")
        self.assertAlmostEqual(result["computation_efficiency"], ref_total / scaled_total)
        self.assertAlmostEqual(
            result["global_efficiency"],
            self.scaled_metrics["parallel_efficiency"] * (ref_total / scaled_total),
        )

    def test_weak_scaling_uses_average_useful_compute(self):
        ref_avg = statistics.mean([8.0, 9.0])
        scaled_avg = statistics.mean([4.0, 4.0, 4.5, 4.0])
        result = pop_tool.compute_scaling_metrics(self.ref_metrics, self.scaled_metrics, "weak")
        self.assertAlmostEqual(result["computation_efficiency"], ref_avg / scaled_avg)

    def test_strong_and_weak_give_different_computation_efficiency(self):
        # Sanity check that the two modes are actually different formulas,
        # not accidentally the same code path.
        strong = pop_tool.compute_scaling_metrics(self.ref_metrics, self.scaled_metrics, "strong")
        weak = pop_tool.compute_scaling_metrics(self.ref_metrics, self.scaled_metrics, "weak")
        self.assertNotAlmostEqual(strong["computation_efficiency"], weak["computation_efficiency"])

    def test_reference_compared_to_itself_is_trivially_one(self):
        result = pop_tool.compute_scaling_metrics(self.ref_metrics, self.ref_metrics, "strong")
        self.assertAlmostEqual(result["computation_efficiency"], 1.0)
        self.assertAlmostEqual(result["global_efficiency"], self.ref_metrics["parallel_efficiency"])


class WriteReportTests(unittest.TestCase):
    def test_single_run_report_has_no_scaling_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            report = pop_tool.write_report([REF_DIR], dest)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("=== Metrics ===", report)
            self.assertIn(pop_tool.run_label(REF_DIR), report)
            header_line = next(line for line in report.splitlines() if "ranks" in line and "LB" in line)
            self.assertNotIn("CompE", header_line)  # no scaling columns in the table itself
            self.assertNotIn("(reference)", report)
            self.assertIn("need a scaling study", report)

    def test_multi_run_report_has_one_merged_table_with_scaling_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            report = pop_tool.write_report([REF_DIR, SCALED_DIR], dest, scaling="strong")
            self.assertIn("strong scaling", report)
            self.assertIn(f"{pop_tool.run_label(REF_DIR)} (reference)", report)
            self.assertIn(pop_tool.run_label(SCALED_DIR), report)
            # exactly one table -- not a separate "Per-run"/"Scaling comparison" pair
            self.assertEqual(report.count("=== Metrics"), 1)
            header_line = next(line for line in report.splitlines() if "ranks" in line and "LB" in line)
            self.assertIn("CompE", header_line)
            self.assertIn("GE", header_line)

    def test_not_computed_and_caveats_sections_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            report = pop_tool.write_report([REF_DIR], dest)
            self.assertIn("Serialisation Efficiency", report)
            self.assertIn("PAPI hardware counters", report)
            self.assertIn("MPICH/Cray-MPICH", report)


class MainCliTests(unittest.TestCase):
    def test_scaling_flag_required_with_scaled_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                pop_tool.main([REF_DIR, SCALED_DIR, "-o", os.path.join(tmp, "out.txt")])

    def test_missing_directory_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            pop_tool.main(["/no/such/directory"])

    def test_single_directory_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            pop_tool.main([REF_DIR, "-o", dest])
            self.assertTrue(os.path.isfile(dest))

    def test_multi_directory_with_scaling_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            pop_tool.main([REF_DIR, SCALED_DIR, "--scaling", "weak", "-o", dest])
            self.assertTrue(os.path.isfile(dest))


if __name__ == "__main__":
    unittest.main()
