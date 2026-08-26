import importlib.util
import glob
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "tools", "extract_trace_hotspots.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("extract_trace_hotspots", MODULE_PATH)
hotspots = importlib.util.module_from_spec(spec)
sys.modules["extract_trace_hotspots"] = hotspots
spec.loader.exec_module(hotspots)

import stage6_time_range_config as trc  # noqa: E402

TWO_RANK_DIR = os.path.join(FIXTURES, "trace_cli_two_rank")
TIME_RANGE_DIR = os.path.join(FIXTURES, "trace_cli_time_range")


def _clear_cache(directory=TWO_RANK_DIR):
    for f in glob.glob(os.path.join(directory, "*.agg.json")):
        os.remove(f)


class WriteReportTests(unittest.TestCase):
    def tearDown(self):
        _clear_cache()
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
        _clear_cache()
        _clear_cache(TIME_RANGE_DIR)
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
        self.assertIn("compute_phase", report)
        self.assertIn("35.000000", report)  # compute_phase's clipped self time

    def test_bad_time_range_syntax_is_a_clear_cli_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            with self.assertRaises(SystemExit):
                hotspots.main([TIME_RANGE_DIR, "--time-range", "not-a-range", "-o", dest])


if __name__ == "__main__":
    unittest.main()
