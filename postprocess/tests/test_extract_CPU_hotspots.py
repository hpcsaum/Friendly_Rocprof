import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "extract_CPU_hotspots.py")

spec = importlib.util.spec_from_file_location("extract_CPU_hotspots", MODULE_PATH)
hotspots = importlib.util.module_from_spec(spec)
sys.modules["extract_CPU_hotspots"] = hotspots
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
    def test_single_rank_bucketing_and_total_runtime(self):
        cpu, gpu, scanned, total_runtime = hotspots.aggregate(os.path.join(FIXTURES, "single_rank"))
        self.assertEqual(len(scanned), 1)
        cpu_labels = {e["label"] for e in cpu}
        gpu_labels = {e["label"] for e in gpu}
        self.assertEqual(cpu_labels, {"main", "compute_stencil", "apply_boundary"})
        self.assertEqual(gpu_labels, {"hipLaunchKernel", "hipMemcpy"})
        # single file -> total_runtime is just that file's largest SUM ("main")
        self.assertAlmostEqual(total_runtime, 13.360265)
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["compute_stencil"]["pct_total"], 9.812345 / 13.360265 * 100)

    def test_mpi_ranks_aggregate_by_function_name_and_total_runtime(self):
        cpu, gpu, scanned, total_runtime = hotspots.aggregate(os.path.join(FIXTURES, "mpi_2rank"))
        self.assertEqual(len(scanned), 2)
        by_label = {e["label"]: e for e in cpu}
        # 9.5 (rank0) + 9.4 (rank1)
        self.assertAlmostEqual(by_label["compute_stencil"]["sum"], 18.9)
        self.assertEqual(by_label["compute_stencil"]["count"], 1000)
        gpu_by_label = {e["label"]: e for e in gpu}
        self.assertAlmostEqual(gpu_by_label["hipMemcpy"]["sum"], 0.0395)
        # total_runtime = rank0's main (10.924161) + rank1's main (10.900000)
        self.assertAlmostEqual(total_runtime, 21.824161)
        self.assertAlmostEqual(by_label["compute_stencil"]["pct_total"], 18.9 / 21.824161 * 100)

    def test_directory_with_no_timing_files(self):
        cpu, gpu, scanned, total_runtime = hotspots.aggregate(os.path.join(FIXTURES, "no_timing_data"))
        self.assertEqual(scanned, [])
        self.assertEqual(cpu, [])
        self.assertEqual(gpu, [])
        self.assertEqual(total_runtime, 0)


class SelectEntriesTests(unittest.TestCase):
    ENTRIES = [
        {"label": "a", "count": 1, "sum": 1.0, "pct_self": 10.0, "pct_total": 10.0},
        {"label": "b", "count": 1, "sum": 5.0, "pct_self": 10.0, "pct_total": 50.0},
        {"label": "c", "count": 1, "sum": 3.0, "pct_self": 10.0, "pct_total": 30.0},
    ]

    def test_default_top_is_20(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_runtime=10.0)
        self.assertEqual([e["label"] for e in selected], ["b", "c", "a"])
        self.assertIn("top 20", desc)

    def test_top_n_truncates(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_runtime=10.0, top=2)
        self.assertEqual([e["label"] for e in selected], ["b", "c"])
        self.assertIn("top 2 of 3", desc)

    def test_threshold_filters_by_pct_total(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_runtime=10.0, threshold=30.0)
        self.assertEqual([e["label"] for e in selected], ["b", "c"])
        self.assertIn(">= 30% of total runtime (2 of 3 entries)", desc)

    def test_threshold_with_unknown_total_runtime_falls_back_to_all(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_runtime=0, threshold=30.0)
        self.assertEqual(len(selected), 3)
        self.assertIn("total runtime unknown, threshold ignored", desc)

    def test_show_all(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_runtime=10.0, show_all=True)
        self.assertEqual(len(selected), 3)
        self.assertIn("all 3 entries", desc)


class FormatTableTests(unittest.TestCase):
    def test_includes_pct_total_column(self):
        entries = [{"label": "a", "count": 1, "sum": 1.0, "pct_self": 10.0, "pct_total": 25.0}]
        table = hotspots.format_table(entries)
        self.assertIn("%total", table)
        self.assertIn("25.0", table)

    def test_pct_total_none_renders_as_na(self):
        entries = [{"label": "a", "count": 1, "sum": 1.0, "pct_self": 10.0, "pct_total": None}]
        table = hotspots.format_table(entries)
        self.assertIn("n/a", table)

    def test_empty_entries(self):
        self.assertIn("none found", hotspots.format_table([]))


class MetadataGuessingTests(unittest.TestCase):
    def test_guesses_from_mpi_fixture_metadata_json(self):
        metadata = hotspots.load_metadata(os.path.join(FIXTURES, "mpi_2rank"))
        self.assertEqual(hotspots.guess_executable(metadata), "jacobi_mpi")
        self.assertEqual(hotspots.guess_run_datetime(metadata, "irrelevant"), "2026-07-21T07:40:00")
        self.assertEqual(hotspots.guess_total_runtime(metadata), "21.824161 sec")
        # world_size is nested under "settings" -- exercises the one-level-deep search
        self.assertEqual(hotspots.guess_num_ranks(metadata, []), 2)

    def test_missing_metadata_json_leaves_fields_blank(self):
        metadata = hotspots.load_metadata(os.path.join(FIXTURES, "single_rank"))
        self.assertEqual(metadata, {})
        self.assertIsNone(hotspots.guess_executable(metadata))
        self.assertIsNone(hotspots.guess_total_runtime(metadata))

    def test_num_ranks_falls_back_to_distinct_pids_in_filenames(self):
        metadata = {}
        scanned = ["/x/wall_clock-1001.txt", "/x/wall_clock-1002.txt", "/x/roctracer-1001.txt"]
        self.assertEqual(hotspots.guess_num_ranks(metadata, scanned), 2)

    def test_run_datetime_falls_back_to_output_dir_timestamp_pattern(self):
        metadata = {}
        output_dir = "/some/rocprof-sys-app-output/2025-01-21_07.40"
        self.assertEqual(hotspots.guess_run_datetime(metadata, output_dir), "2025-01-21_07.40")

    def test_gather_run_info_end_to_end_with_metadata(self):
        info = hotspots.gather_run_info(os.path.join(FIXTURES, "mpi_2rank"), [])
        self.assertEqual(info["executable"], "jacobi_mpi")
        self.assertEqual(info["num_ranks"], 2)

    def test_gather_run_info_end_to_end_without_metadata(self):
        scanned = [os.path.join(FIXTURES, "single_rank", "wall_clock-1234.txt")]
        info = hotspots.gather_run_info(os.path.join(FIXTURES, "single_rank"), scanned)
        self.assertIsNone(info["executable"])
        self.assertIsNone(info["run_datetime"])
        self.assertIsNone(info["total_runtime"])
        self.assertEqual(info["num_ranks"], 1)  # one distinct pid (1234) among scanned files


class WriteReportTests(unittest.TestCase):
    def test_end_to_end_on_mpi_fixture_default_top20(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("compute_stencil", report)
            self.assertIn("showing top 20 of", report)
            self.assertIn("true GPU kernel execution time is not present", report)
            self.assertIn("scripts/profile_GPU_hotspots.sh", report)
            self.assertIn("executable: jacobi_mpi", report)
            self.assertIn("run date/time: 2026-07-21T07:40:00", report)
            self.assertIn("total runtime: 21.824161 sec", report)
            self.assertIn("MPI ranks: 2", report)

    def test_end_to_end_on_single_rank_fixture_blank_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "single_rank"), dest)
            self.assertIn("executable: \n", report)
            self.assertIn("run date/time: \n", report)
            self.assertIn("total runtime: \n", report)
            self.assertIn("MPI ranks: 1\n", report)

    def test_threshold_selection_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest, threshold=50.0)
            self.assertIn(">= 50% of total runtime", report)
            self.assertNotIn("apply_boundary", report)  # not present in this fixture anyway, sanity check

    def test_show_all_selection_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest, show_all=True)
            self.assertIn("showing all", report)

    def test_raises_when_no_timing_data_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            with self.assertRaises(SystemExit):
                hotspots.write_report(os.path.join(FIXTURES, "no_timing_data"), dest)


if __name__ == "__main__":
    unittest.main()
