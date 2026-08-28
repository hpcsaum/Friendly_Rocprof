"""Tests for extract_trace_pop_metrics.py, the POP-inspired parallel-efficiency metrics report
computed from one or more rocprof-sys Perfetto trace-CSV exports.

WriteReportTests -- single-run report: GPU columns, unified-source note, always-present time-range note
MainCliTests     -- CLI entry point: scaling-flag requirement with multiple dirs, end-to-end write,
                     --time-range windowing
"""

import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402
from _tools_test_helpers import assert_help_leads_with_explanation, clear_agg_cache  # noqa: E402

pop_metrics = load_module_by_path("extract_trace_pop_metrics", "tools", "extract_trace_pop_metrics.py")

import stage6_time_range_config as trc  # noqa: E402

TWO_RANK_DIR = os.path.join(FIXTURES, "trace_cli_two_rank")
TIME_RANGE_DIR = os.path.join(FIXTURES, "trace_cli_time_range")


class WriteReportTests(unittest.TestCase):
    def tearDown(self):
        clear_agg_cache(TWO_RANK_DIR)
        trc.configure(None)

    def test_single_run_reports_gpu_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            report = pop_metrics.write_report([TWO_RANK_DIR], dest)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("GPU-Util", report)  # GPU visibility is always part of the trace
            self.assertIn("pool: CPU+GPU, single unified trace source", report)
            self.assertIn("MPI ranks: 2", report)

    def test_run_header_always_shows_a_time_range_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            report = pop_metrics.write_report([TWO_RANK_DIR], dest)
            self.assertIn("time range:", report)
            self.assertIn("(full run)", report)


class MainCliTests(unittest.TestCase):
    def tearDown(self):
        clear_agg_cache(TWO_RANK_DIR)
        clear_agg_cache(TIME_RANGE_DIR)
        trc.configure(None)

    def test_main_requires_scaling_flag_with_scaled_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            with self.assertRaises(SystemExit):
                pop_metrics.main([TWO_RANK_DIR, TWO_RANK_DIR, "-o", dest])

    def test_main_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            pop_metrics.main([TWO_RANK_DIR, "-o", dest])
            with open(dest) as f:
                report = f.read()
        self.assertIn("=== Metrics ===", report)

    def test_time_range_flag_restricts_the_metrics_and_notes_the_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            pop_metrics.main([TIME_RANGE_DIR, "--time-range", "30:70", "-o", dest])
            with open(dest) as f:
                report = f.read()
        self.assertIn("time range: 30.000s-70.000s", report)


class HelpTextTests(unittest.TestCase):
    def test_help_leads_with_explanation(self):
        assert_help_leads_with_explanation(self, pop_metrics, "rocprofiler-systems/en/latest")


if __name__ == "__main__":
    unittest.main()
