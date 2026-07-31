import importlib.util
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "extract_GPU_hotspots.py")

spec = importlib.util.spec_from_file_location("extract_GPU_hotspots", MODULE_PATH)
hotspots = importlib.util.module_from_spec(spec)
sys.modules["extract_GPU_hotspots"] = hotspots
spec.loader.exec_module(hotspots)


class ParseKernelStatsCsvTests(unittest.TestCase):
    def test_single_rank_file(self):
        path = os.path.join(FIXTURES, "rocprofv3_single_rank", "myhost", "1234_kernel_stats.csv")
        rows = hotspots.parse_kernel_stats_csv(path)
        self.assertIsNotNone(rows)
        labels = {r["label"] for r in rows}
        self.assertEqual(labels, {"JacobiIterationKernel", "BoundaryKernel", "__hipRegisterFatBinary"})
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["JacobiIterationKernel"]["count"], 1000)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["total_ns"], 537449866)

    def test_scientific_notation_value_parses(self):
        path = os.path.join(FIXTURES, "rocprofv3_single_rank", "myhost", "1234_kernel_stats.csv")
        rows = hotspots.parse_kernel_stats_csv(path)
        by_label = {r["label"]: r for r in rows}
        # TotalDurationNs=9000 here is plain, but this row's Percentage/StdDev are in
        # scientific notation in the fixture -- confirms DictReader/our code don't choke on it.
        self.assertAlmostEqual(by_label["__hipRegisterFatBinary"]["total_ns"], 9000)

    def test_non_matching_csv_returns_none(self):
        path = os.path.join(FIXTURES, "rocprofv3_no_data", "myhost", "1_agent_info.csv")
        self.assertIsNone(hotspots.parse_kernel_stats_csv(path))


class AggregateTests(unittest.TestCase):
    def test_single_rank(self):
        entries, scanned, total_ns = hotspots.aggregate(os.path.join(FIXTURES, "rocprofv3_single_rank"))
        self.assertEqual(len(scanned), 1)
        by_label = {e["label"]: e for e in entries}
        expected_total_ns = 537449866 + 58000000 + 9000
        self.assertAlmostEqual(total_ns, expected_total_ns)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["sum"], 537449866 / 1e9)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["pct_total"], 537449866 / expected_total_ns * 100)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["avg_us"], 537449866 / 1000 / 1000)

    def test_mpi_ranks_aggregate_by_kernel_name(self):
        entries, scanned, total_ns = hotspots.aggregate(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"))
        self.assertEqual(len(scanned), 2)
        by_label = {e["label"]: e for e in entries}
        jacobi_ns = 268724933 + 268000000
        boundary_ns = 16000000 + 17000000
        expected_total_ns = jacobi_ns + boundary_ns
        self.assertAlmostEqual(total_ns, expected_total_ns)
        self.assertEqual(by_label["JacobiIterationKernel"]["count"], 1000)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["sum"], jacobi_ns / 1e9)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["pct_total"], jacobi_ns / expected_total_ns * 100)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["avg_us"], jacobi_ns / 1000 / 1000)

    def test_no_kernel_stats_csv_found(self):
        entries, scanned, total_ns = hotspots.aggregate(os.path.join(FIXTURES, "rocprofv3_no_data"))
        self.assertEqual(entries, [])
        self.assertEqual(scanned, [])
        self.assertEqual(total_ns, 0)


class SelectEntriesTests(unittest.TestCase):
    ENTRIES = [
        {"label": "a", "count": 1, "sum": 1.0, "avg_us": 1.0, "pct_total": 10.0},
        {"label": "b", "count": 1, "sum": 5.0, "avg_us": 1.0, "pct_total": 50.0},
        {"label": "c", "count": 1, "sum": 3.0, "avg_us": 1.0, "pct_total": 30.0},
    ]

    def test_default_top_is_20(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_ns=10.0)
        self.assertEqual([e["label"] for e in selected], ["b", "c", "a"])
        self.assertIn("top 20", desc)

    def test_top_n_truncates(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_ns=10.0, top=2)
        self.assertEqual([e["label"] for e in selected], ["b", "c"])

    def test_threshold_filters(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_ns=10.0, threshold=30.0)
        self.assertEqual([e["label"] for e in selected], ["b", "c"])
        self.assertIn(">= 30% of total runtime (2 of 3 entries)", desc)

    def test_threshold_unknown_total_falls_back_to_all(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_ns=0, threshold=30.0)
        self.assertEqual(len(selected), 3)
        self.assertIn("total runtime unknown", desc)

    def test_show_all(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_ns=10.0, show_all=True)
        self.assertEqual(len(selected), 3)
        self.assertIn("all 3 entries", desc)


class FormatTableTests(unittest.TestCase):
    def test_includes_expected_columns(self):
        entries = [{"label": "k", "count": 2, "sum": 0.001, "avg_us": 500.0, "pct_total": 12.5}]
        table = hotspots.format_table(entries)
        self.assertIn("%total", table)
        self.assertIn("avg(us)", table)
        self.assertIn("12.5", table)
        self.assertIn("500.00", table)

    def test_empty(self):
        self.assertIn("none found", hotspots.format_table([]))


class ConfigJsonGuessingTests(unittest.TestCase):
    def test_guesses_from_config_json(self):
        config = hotspots.load_config_json(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"))
        self.assertEqual(hotspots.guess_executable(config), "jacobi_hip")
        self.assertEqual(hotspots.guess_run_datetime(config), "2026-07-25T09:15:00")
        # "elapsed" is nested under "timing" -- exercises the one-level-deep search
        self.assertEqual(hotspots.guess_total_runtime(config), "5.980000 sec")

    def test_missing_config_json_leaves_fields_blank(self):
        config = hotspots.load_config_json(os.path.join(FIXTURES, "rocprofv3_single_rank"))
        self.assertEqual(config, {})
        self.assertIsNone(hotspots.guess_executable(config))
        self.assertIsNone(hotspots.guess_total_runtime(config))

    def test_num_ranks_from_distinct_pids(self):
        scanned = [
            "/x/myhost/2001_kernel_stats.csv",
            "/x/myhost/2002_kernel_stats.csv",
        ]
        self.assertEqual(hotspots.guess_num_ranks({}, scanned), 2)

    def test_num_ranks_none_when_no_files(self):
        self.assertIsNone(hotspots.guess_num_ranks({}, []))


class WriteReportTests(unittest.TestCase):
    def test_end_to_end_mpi_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"), dest)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("JacobiIterationKernel", report)
            self.assertIn("showing top 20 of", report)
            self.assertIn("executable: jacobi_hip", report)
            self.assertIn("run date/time: 2026-07-25T09:15:00", report)
            self.assertIn("total runtime: 5.980000 sec", report)
            self.assertIn("MPI ranks: 2", report)
            self.assertIn("GPU kernel execution time only", report)
            self.assertIn("scripts/profile_CPU_hotspots.sh", report)

    def test_end_to_end_single_rank_blank_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "rocprofv3_single_rank"), dest)
            self.assertIn("executable: \n", report)
            self.assertIn("run date/time: \n", report)
            self.assertIn("total runtime: \n", report)
            self.assertIn("MPI ranks: 1\n", report)

    def test_threshold_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"), dest, threshold=90.0)
            self.assertIn(">= 90% of total runtime", report)
            self.assertNotIn("BoundaryKernel", report)

    def test_all_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"), dest, show_all=True)
            self.assertIn("showing all", report)
            self.assertIn("BoundaryKernel", report)

    def test_raises_when_no_kernel_stats_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            with self.assertRaises(SystemExit):
                hotspots.write_report(os.path.join(FIXTURES, "rocprofv3_no_data"), dest)


if __name__ == "__main__":
    unittest.main()
