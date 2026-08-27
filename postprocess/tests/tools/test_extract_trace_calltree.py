import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402
from _tools_test_helpers import clear_agg_cache  # noqa: E402

calltree = load_module_by_path("extract_trace_calltree", "tools", "extract_trace_calltree.py")

import stage6_time_range_config as trc  # noqa: E402

TWO_RANK_DIR = os.path.join(FIXTURES, "trace_cli_two_rank")
NOISE_DIR = os.path.join(FIXTURES, "trace_cli_calltree_noise")
TIME_RANGE_DIR = os.path.join(FIXTURES, "trace_cli_time_range")


class WriteReportTests(unittest.TestCase):
    def tearDown(self):
        clear_agg_cache(TWO_RANK_DIR)
        clear_agg_cache(NOISE_DIR)
        trc.configure(None)

    def test_end_to_end_tree_hides_gpu_api_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            report = calltree.write_report(TWO_RANK_DIR, dest)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("jacobi_sweep", report)
            self.assertNotIn("hipLaunchKernel", report)
            self.assertIn("Ranks aggregated: 0, 1", report)

    def test_show_gpu_api_reveals_the_gpu_subtree(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            report = calltree.write_report(TWO_RANK_DIR, dest, show_gpu_api=True)
            self.assertIn("hipLaunchKernel", report)
            self.assertIn("jacobi_kernel.kd", report)

    def test_show_all_internals_reveals_every_tier(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            report = calltree.write_report(
                NOISE_DIR, dest, show_gpu_api=True, show_rocprofsys_internals=True,
                show_mpi_internals=True, show_compiler_runtime=True,
            )
            self.assertIn("gotcha_call", report)
            self.assertIn("posix_memalign", report)


class MainCliTests(unittest.TestCase):
    def tearDown(self):
        clear_agg_cache(TWO_RANK_DIR)
        clear_agg_cache(TIME_RANGE_DIR)
        trc.configure(None)

    def test_show_all_internals_flag_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            calltree.main([TWO_RANK_DIR, "-o", dest, "--show-all-internals"])
            with open(dest) as f:
                report = f.read()
        self.assertIn("hipLaunchKernel", report)
        self.assertIn("shown), rocprof-sys internals (shown)", report)

    def test_time_range_flag_cuts_out_of_range_subtrees_and_notes_the_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            calltree.main([TIME_RANGE_DIR, "--time-range", "30:70", "-o", dest])
            with open(dest) as f:
                report = f.read()
        self.assertIn("time range: 30.000s-70.000s", report)
        self.assertNotIn("init_phase", report)
        self.assertNotIn("teardown_phase", report)
        self.assertIn("compute_phase", report)

    def test_no_time_range_flag_shows_the_full_run_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            calltree.main([TIME_RANGE_DIR, "-o", dest])
            with open(dest) as f:
                report = f.read()
        self.assertIn("time range: 0.000s-100.000s (full run)", report)
        self.assertIn("init_phase", report)
        self.assertIn("teardown_phase", report)


if __name__ == "__main__":
    unittest.main()
