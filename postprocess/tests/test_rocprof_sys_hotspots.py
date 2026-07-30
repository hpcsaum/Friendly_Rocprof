import importlib.util
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "rocprof_sys_hotspots.py")

spec = importlib.util.spec_from_file_location("rocprof_sys_hotspots", MODULE_PATH)
hotspots = importlib.util.module_from_spec(spec)
sys.modules["rocprof_sys_hotspots"] = hotspots
spec.loader.exec_module(hotspots)


class CleanLabelTests(unittest.TestCase):
    def test_single_rank_no_indent(self):
        self.assertEqual(hotspots.clean_label("00>>>main"), "main")

    def test_single_rank_with_indent(self):
        self.assertEqual(hotspots.clean_label("00>>>|_compute_stencil"), "compute_stencil")

    def test_mpi_rank_prefix_no_indent(self):
        self.assertEqual(hotspots.clean_label("00|00>>>main"), "main")

    def test_mpi_rank_prefix_with_indent(self):
        self.assertEqual(hotspots.clean_label("00|00>>>|_compute_stencil"), "compute_stencil")


class ParseTableFileTests(unittest.TestCase):
    def test_flat_single_rank_file(self):
        path = os.path.join(FIXTURES, "single_rank", "wall_clock-1234.txt")
        rows = hotspots.parse_table_file(path)
        self.assertIsNotNone(rows)
        labels = {r["label"] for r in rows}
        self.assertEqual(labels, {"main", "compute_stencil", "apply_boundary", "hipLaunchKernel", "hipMemcpy"})
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["compute_stencil"]["count"], 1000)
        self.assertAlmostEqual(by_label["compute_stencil"]["sum"], 9.812345)

    def test_hierarchical_mpi_rank_file(self):
        path = os.path.join(FIXTURES, "mpi_2rank", "wall_clock-2001.txt")
        rows = hotspots.parse_table_file(path)
        self.assertIsNotNone(rows)
        labels = {r["label"] for r in rows}
        self.assertEqual(labels, {"main", "compute_stencil", "hipMemcpy"})

    def test_non_timing_file_returns_none(self):
        path = os.path.join(FIXTURES, "no_timing_data", "available.txt")
        self.assertIsNone(hotspots.parse_table_file(path))


class IsGpuEntryTests(unittest.TestCase):
    def test_hip_prefix_is_gpu(self):
        self.assertTrue(hotspots.is_gpu_entry("hipLaunchKernel", "wall_clock-1.txt"))

    def test_hsa_prefix_is_gpu(self):
        self.assertTrue(hotspots.is_gpu_entry("hsa_amd_memory_pool_allocate", "wall_clock-1.txt"))

    def test_plain_function_is_cpu(self):
        self.assertFalse(hotspots.is_gpu_entry("compute_stencil", "wall_clock-1.txt"))

    def test_roctracer_filename_forces_gpu(self):
        self.assertTrue(hotspots.is_gpu_entry("some_wrapped_call", "roctracer-1.txt"))


class AggregateTests(unittest.TestCase):
    def test_single_rank_bucketing(self):
        cpu, gpu, scanned = hotspots.aggregate(os.path.join(FIXTURES, "single_rank"))
        self.assertEqual(len(scanned), 1)
        cpu_labels = {e["label"] for e in cpu}
        gpu_labels = {e["label"] for e in gpu}
        self.assertEqual(cpu_labels, {"main", "compute_stencil", "apply_boundary"})
        self.assertEqual(gpu_labels, {"hipLaunchKernel", "hipMemcpy"})

    def test_mpi_ranks_aggregate_by_function_name(self):
        cpu, gpu, scanned = hotspots.aggregate(os.path.join(FIXTURES, "mpi_2rank"))
        self.assertEqual(len(scanned), 2)
        by_label = {e["label"]: e for e in cpu}
        # 9.5 (rank0) + 9.4 (rank1)
        self.assertAlmostEqual(by_label["compute_stencil"]["sum"], 18.9)
        self.assertEqual(by_label["compute_stencil"]["count"], 1000)
        gpu_by_label = {e["label"]: e for e in gpu}
        self.assertAlmostEqual(gpu_by_label["hipMemcpy"]["sum"], 0.0395)

    def test_directory_with_no_timing_files(self):
        cpu, gpu, scanned = hotspots.aggregate(os.path.join(FIXTURES, "no_timing_data"))
        self.assertEqual(scanned, [])
        self.assertEqual(cpu, [])
        self.assertEqual(gpu, [])


class FormatTableTests(unittest.TestCase):
    def test_top_n_sorting_and_truncation(self):
        entries = [
            {"label": "a", "count": 1, "sum": 1.0, "pct_self": 10.0},
            {"label": "b", "count": 1, "sum": 5.0, "pct_self": 10.0},
            {"label": "c", "count": 1, "sum": 3.0, "pct_self": 10.0},
        ]
        table = hotspots.format_table(entries, top_n=2)
        self.assertIn("b", table)
        self.assertIn("c", table)
        self.assertNotIn(" a\n", table)

    def test_empty_entries(self):
        self.assertIn("none found", hotspots.format_table([], top_n=5))


class WriteReportTests(unittest.TestCase):
    def test_end_to_end_on_mpi_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest, top_n=10)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("compute_stencil", report)
            self.assertIn("Top 10 CPU compute hotspots", report)
            self.assertIn("Top 10 GPU API / launch overhead", report)
            self.assertIn("true GPU kernel execution time is not present", report)
            self.assertIn("rocprofv3 --stats --kernel-trace --summary", report)

    def test_raises_when_no_timing_data_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            with self.assertRaises(SystemExit):
                hotspots.write_report(os.path.join(FIXTURES, "no_timing_data"), dest, top_n=10)


if __name__ == "__main__":
    unittest.main()
