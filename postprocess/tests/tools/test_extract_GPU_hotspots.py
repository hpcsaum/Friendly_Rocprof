"""Tests for extract_GPU_hotspots.py, the GPU kernel (rocprofv3 kernel_stats.csv) hotspots report tool.

ConfigJsonGuessingTests -- CONFIG_*_KEYS constants match a real *_config.json fixture, num-ranks-from-PIDs
WriteReportTests        -- end-to-end report content: tables, header, threshold/show-all/load imbalance
MainCliTests            -- main()'s own argv wiring: missing-dir error, -o write, --all flag reaches write_report
"""

import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")

# extract_GPU_hotspots.py does a plain top-level "from stage4_rocprofv3 import ...",
# relying on its own directory being on sys.path -- true automatically when run
# directly, but not when loaded here by explicit file path, so replicate that
# manually (same as test_extract_hotspots.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402
from _tools_test_helpers import assert_help_leads_with_explanation  # noqa: E402

hotspots = load_module_by_path("extract_GPU_hotspots", "tools", "extract_GPU_hotspots.py")


class ConfigJsonGuessingTests(unittest.TestCase):
    # Wiring check: "missing data -> None" behavior itself is exhaustively covered generically in
    # test_stage6_run_metadata.py; this confirms this tool's own CONFIG_EXECUTABLE_KEYS/
    # CONFIG_DATETIME_KEYS/CONFIG_RUNTIME_KEYS constants actually match a real *_config.json
    # fixture, not synthetic keys.
    def test_guesses_from_config_json(self):
        config = hotspots.load_json_file(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"), "*_config.json")
        self.assertEqual(hotspots.guess_executable(config, hotspots.CONFIG_EXECUTABLE_KEYS), "jacobi_hip")
        self.assertEqual(
            hotspots.guess_run_datetime(config, hotspots.CONFIG_DATETIME_KEYS), "2026-07-25T09:15:00",
        )
        # "elapsed" is nested under "timing" -- exercises the one-level-deep search
        self.assertEqual(hotspots.guess_total_runtime(config, hotspots.CONFIG_RUNTIME_KEYS), "5.980000 sec")

    def test_num_ranks_from_distinct_pids(self):
        # PID_SUFFIX_RE here (*_kernel_stats.csv) is defined locally in extract_GPU_hotspots.py,
        # not shared -- this is the only coverage of that specific pattern.
        scanned = [
            "/x/myhost/2001_kernel_stats.csv",
            "/x/myhost/2002_kernel_stats.csv",
        ]
        self.assertEqual(hotspots.guess_num_ranks({}, hotspots.PID_SUFFIX_RE, scanned), 2)


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
            self.assertIn("runtime: 5.980000 sec", report)
            self.assertIn("MPI ranks: 2", report)
            self.assertIn("each kernel's share of total measured GPU time", report)
            self.assertIn("scripts/profile_CPU_hotspots.sh", report)

    def test_end_to_end_single_rank_blank_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "rocprofv3_single_rank"), dest)
            self.assertIn("executable: \n", report)
            self.assertIn("run date/time: \n", report)
            self.assertIn("runtime: \n", report)
            self.assertIn("MPI ranks: 1\n", report)

    def test_threshold_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"), dest, threshold=90.0)
            self.assertIn(">= 90% of total measured GPU time", report)
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


class MainCliTests(unittest.TestCase):
    def test_missing_directory_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            hotspots.main(["/no/such/directory"])

    def test_end_to_end_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            hotspots.main([os.path.join(FIXTURES, "rocprofv3_mpi_2rank"), "-o", dest])
            self.assertTrue(os.path.isfile(dest))

    def test_all_flag_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            hotspots.main([os.path.join(FIXTURES, "rocprofv3_mpi_2rank"), "-o", dest, "--all"])
            with open(dest) as f:
                report = f.read()
        self.assertIn("showing all", report)
        self.assertIn("BoundaryKernel", report)


class HelpTextTests(unittest.TestCase):
    def test_help_leads_with_explanation(self):
        assert_help_leads_with_explanation(self, hotspots, "rocprofiler-sdk/en/latest")


if __name__ == "__main__":
    unittest.main()
