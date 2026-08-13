import importlib.util
import os
import statistics
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "extract_GPU_hotspots.py")

# extract_GPU_hotspots.py does a plain top-level "from stage1_rocprofv3 import ...",
# relying on its own directory being on sys.path -- true automatically when run
# directly, but not when loaded here by explicit file path, so replicate that
# manually (same as test_extract_hotspots.py).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("extract_GPU_hotspots", MODULE_PATH)
hotspots = importlib.util.module_from_spec(spec)
sys.modules["extract_GPU_hotspots"] = hotspots
spec.loader.exec_module(hotspots)


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


class AggregatePerRankTests(unittest.TestCase):
    def test_single_rank_returns_one_dict(self):
        per_file, scanned = hotspots.aggregate_per_rank(os.path.join(FIXTURES, "rocprofv3_single_rank"))
        self.assertEqual(len(scanned), 1)
        self.assertEqual(len(per_file), 1)
        self.assertAlmostEqual(per_file[0]["JacobiIterationKernel"], 537449866 / 1e9)

    def test_mpi_2rank_keeps_ranks_separate(self):
        per_file, scanned = hotspots.aggregate_per_rank(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"))
        self.assertEqual(len(scanned), 2)
        values = sorted(ft["JacobiIterationKernel"] for ft in per_file)
        self.assertAlmostEqual(values[0], 268000000 / 1e9)
        self.assertAlmostEqual(values[1], 268724933 / 1e9)

    def test_no_kernel_stats_csv_found(self):
        per_file, scanned = hotspots.aggregate_per_rank(os.path.join(FIXTURES, "rocprofv3_no_data"))
        self.assertEqual(per_file, [])
        self.assertEqual(scanned, [])


class ComputeLoadImbalanceTests(unittest.TestCase):
    def test_matches_independently_computed_statistics_on_real_fixture(self):
        per_file, _ = hotspots.aggregate_per_rank(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"))
        selected, _ = hotspots.compute_load_imbalance(per_file, show_all=True)
        by_label = {e["label"]: e for e in selected}

        jacobi_values = [ft["JacobiIterationKernel"] for ft in per_file]
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["avg"], statistics.mean(jacobi_values))
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["std_dev"], statistics.pstdev(jacobi_values))
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["min"], min(jacobi_values))
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["max"], max(jacobi_values))

    def test_missing_rank_scores_zero_not_omitted(self):
        per_file_totals = [{"only_on_rank0": 10.0}, {}]
        selected, _ = hotspots.compute_load_imbalance(per_file_totals, show_all=True)
        entry = next(e for e in selected if e["label"] == "only_on_rank0")
        self.assertAlmostEqual(entry["avg"], 5.0)
        self.assertAlmostEqual(entry["std_dev"], 5.0)
        self.assertAlmostEqual(entry["min"], 0.0)
        self.assertAlmostEqual(entry["max"], 10.0)

    def test_top_n_selects_highest_std_dev(self):
        per_file_totals = [
            {"a": 10.0, "b": 5.0, "c": 100.0},
            {"a": 10.0, "b": 15.0, "c": 100.0},
        ]
        selected, desc = hotspots.compute_load_imbalance(per_file_totals, top=1)
        self.assertEqual([e["label"] for e in selected], ["b"])
        self.assertIn("top 1 of 3", desc)

    def test_threshold_is_coefficient_of_variation(self):
        per_file_totals = [
            {"a": 10.0, "b": 5.0, "c": 100.0},
            {"a": 10.0, "b": 15.0, "c": 100.0},
        ]
        selected, desc = hotspots.compute_load_imbalance(per_file_totals, threshold=10.0)
        self.assertEqual([e["label"] for e in selected], ["b"])
        self.assertIn("coefficient of variation", desc)


class FormatTableLoadImbalanceTests(unittest.TestCase):
    def test_includes_expected_columns(self):
        entries = [{"label": "MyKernel", "avg": 1.0, "std_dev": 0.5, "min": 0.5, "max": 1.5, "cv_pct": 50.0}]
        table = hotspots.format_table_load_imbalance(entries)
        self.assertIn("avg(s)", table)
        self.assertIn("std_dev", table)
        self.assertIn("min(s)", table)
        self.assertIn("max(s)", table)
        self.assertIn("kernel", table)
        self.assertIn("MyKernel", table)

    def test_empty_entries(self):
        self.assertIn("none found", hotspots.format_table_load_imbalance([]))


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

    def test_load_imbalance_table_present_after_hotspots_table_on_mpi_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"), dest)
            i_hotspots = report.index("GPU kernel hotspots")
            i_imbalance = report.index("GPU kernel load imbalance across 2 ranks")
            self.assertTrue(i_hotspots < i_imbalance)
            self.assertIn("JacobiIterationKernel", report[i_imbalance:])

    def test_load_imbalance_table_skipped_on_single_rank_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "rocprofv3_single_rank"), dest)
            self.assertIn("GPU kernel load imbalance across ranks -- skipped: only 1 rank/file found", report)


if __name__ == "__main__":
    unittest.main()
