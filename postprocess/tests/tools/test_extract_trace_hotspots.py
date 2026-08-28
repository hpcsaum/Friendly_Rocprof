"""Tests for extract_trace_hotspots.py, the combined CPU+GPU hotspots report built from a
rocprof-sys Perfetto trace-CSV export (fused table, load imbalance, POP-style header notes).

WriteReportTests -- default fused table + load imbalance, --show-all, always-present time-range note
MainCliTests     -- CLI entry point: end-to-end write, --time-range windowing, bad --time-range
                    syntax, --top/--threshold/--unfiltered selection wiring
"""

import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402
from _tools_test_helpers import assert_help_leads_with_explanation, clear_agg_cache  # noqa: E402

hotspots = load_module_by_path("extract_trace_hotspots", "tools", "extract_trace_hotspots.py")

import stage6_time_range_config as trc  # noqa: E402

TWO_RANK_DIR = os.path.join(FIXTURES, "trace_cli_two_rank")
TIME_RANGE_DIR = os.path.join(FIXTURES, "trace_cli_time_range")


class WriteReportTests(unittest.TestCase):
    def tearDown(self):
        clear_agg_cache(TWO_RANK_DIR)
        trc.configure(None)

    def test_end_to_end_fused_table_and_load_imbalance(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(TWO_RANK_DIR, dest)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("jacobi_sweep", report)
            self.assertIn("jacobi_kernel.kd", report)  # GPU domain row present in the fused table
            self.assertIn("Load imbalance across 2 ranks", report)
            self.assertIn("MPI ranks: 2", report)

    def test_show_all_selection_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(TWO_RANK_DIR, dest, show_all=True)
            self.assertIn("showing all", report)

    def test_header_always_shows_a_time_range_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(TWO_RANK_DIR, dest)
            self.assertIn("time range:", report)
            self.assertIn("(full run)", report)


class MainCliTests(unittest.TestCase):
    def tearDown(self):
        clear_agg_cache(TWO_RANK_DIR)
        clear_agg_cache(TIME_RANGE_DIR)
        trc.configure(None)

    def test_main_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            hotspots.main([TWO_RANK_DIR, "-o", dest])
            with open(dest) as f:
                report = f.read()
        self.assertIn("jacobi_sweep", report)

    def test_time_range_flag_restricts_the_report(self):
        # The flat hotspots table has no analogous "cut a zero-time subtree" concept (that's the
        # calltree tool's own job -- see stage4_rocprofsys_common.make_zero_time_pruned()) -- a
        # function entirely outside the window still gets its own row here, honestly showing
        # 0 calls / 0.0 self-time rather than being hidden, so this checks the NUMBERS reflect the
        # window, not row presence/absence.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            hotspots.main([TIME_RANGE_DIR, "--time-range", "30:70", "-o", dest])
            with open(dest) as f:
                report = f.read()
        self.assertIn("time range: 30.000s-70.000s", report)
        for line in report.splitlines():
            if "init_phase" in line or "teardown_phase" in line:
                self.assertIn("0.000000", line)
        # compute_phase's exact clipped self-time (35.0s) is already verified directly against
        # build_rank_aggregate() in test_stage4_rocprofsys_trace_aggregate.py's
        # TimeRangeClippingAndExtentTests -- this only confirms the row survives and reads as
        # non-zero end to end through this tool's report, not the clipping arithmetic itself.
        self.assertIn("compute_phase", report)
        compute_line = next(line for line in report.splitlines() if "compute_phase" in line)
        self_time_column = compute_line.split()[1]
        self.assertNotEqual(self_time_column, "0.000000")

    def test_bad_time_range_syntax_is_a_clear_cli_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            with self.assertRaises(SystemExit):
                hotspots.main([TIME_RANGE_DIR, "--time-range", "not-a-range", "-o", dest])

    @staticmethod
    def _hotspots_table_only(report):
        # Scoped to just the "1. CPU+GPU hotspots" table's own rows -- the report also contains
        # a second "2. Load imbalance" table (its own separate top/threshold selection, ranked by
        # std_dev) whose row labels would otherwise leak into an unscoped substring search.
        start = report.index("=== 1. CPU+GPU hotspots")
        end = report.index("=== 2. Load imbalance")
        return report[start:end]

    def test_top_flag_limits_entries_end_to_end(self):
        # By self_sum (default ranking field): jacobi_sweep (12.9s) > jacobi_kernel.kd (7.0s) >
        # main (5.0s) > MPI_Barrier (3.0s) > hipLaunchKernel (1.1s) -- --top 2 keeps only the
        # first two.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            hotspots.main([TWO_RANK_DIR, "-o", dest, "--top", "2"])
            with open(dest) as f:
                report = f.read()
        table = self._hotspots_table_only(report)
        self.assertIn("jacobi_sweep", table)
        self.assertIn("jacobi_kernel.kd", table)
        self.assertNotIn("MPI_Barrier", table)
        self.assertNotIn("hipLaunchKernel", table)

    def test_threshold_flag_filters_by_pct_total_end_to_end(self):
        # pct_total (self_sum / total_runtime): jacobi_sweep 58.6%, jacobi_kernel.kd 31.8%, main
        # 22.7%, MPI_Barrier 13.6%, hipLaunchKernel 5.0% -- threshold=40 keeps only the first two.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            hotspots.main([TWO_RANK_DIR, "-o", dest, "--threshold", "40"])
            with open(dest) as f:
                report = f.read()
        table = self._hotspots_table_only(report)
        self.assertIn("jacobi_sweep", table)
        self.assertNotIn("jacobi_kernel.kd", table)
        self.assertNotIn("MPI_Barrier", table)

    def test_unfiltered_flag_switches_the_ranking_field_to_inclusive_sum(self):
        # Filtered (self_sum) ranks jacobi_sweep (12.9s) above main (5.0s); --unfiltered switches
        # the ranking field to inclusive "sum", where main (22.0s, the whole process) outranks
        # jacobi_sweep (17.0s) -- --top 1 makes the field switch observable as which single row
        # survives.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            hotspots.main([TWO_RANK_DIR, "-o", dest, "--top", "1", "--unfiltered"])
            with open(dest) as f:
                report = f.read()
        table = self._hotspots_table_only(report)
        self.assertIn("main", table)
        self.assertNotIn("jacobi_sweep", table)


class HelpTextTests(unittest.TestCase):
    def test_help_leads_with_explanation(self):
        assert_help_leads_with_explanation(self, hotspots, "rocprofiler-systems/en/latest")


if __name__ == "__main__":
    unittest.main()
