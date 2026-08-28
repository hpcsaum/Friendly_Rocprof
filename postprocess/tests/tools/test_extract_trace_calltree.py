"""Tests for extract_trace_calltree.py, the trace-CSV-driven call tree tool (exact GPU kernel
placement via corr_id, unlike the sampling-based extract_calltree.py/extract_wallclock_calltree.py).

WriteReportTests -- default GPU-API hiding, --show-gpu-api/--show-all-internals tier reveal
MainCliTests     -- CLI entry point: --show-all-internals/--time-range/--max-depth/each show-*
                    flag alone/--extra-noise-config, all end to end
"""

import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402
from _tools_test_helpers import assert_help_leads_with_explanation, clear_agg_cache  # noqa: E402
from _stage6_test_helpers import write_noise_config  # noqa: E402

calltree = load_module_by_path("extract_trace_calltree", "tools", "extract_trace_calltree.py")

import stage6_noise_config  # noqa: E402
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

    def test_max_depth_flag_truncates_and_notes_hidden_count(self):
        # TWO_RANK_DIR's tree is main -> jacobi_sweep -> {hipLaunchKernel, MPI_Barrier} --
        # --max-depth 1 keeps jacobi_sweep but truncates its children.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            calltree.main([TWO_RANK_DIR, "-o", dest, "--max-depth", "1"])
            with open(dest) as f:
                report = f.read()
        self.assertIn("jacobi_sweep", report)
        self.assertNotIn("MPI_Barrier", report)
        self.assertIn("more node(s) hidden", report)

    def test_each_show_flag_alone_reveals_only_its_own_tier(self):
        # mpi_internals has no distinguishing label in NOISE_DIR (MPI_Barrier, the mpi_territory
        # row itself, is always shown regardless of this flag -- see
        # test_stage5_trace_calltree_view.py's own equivalent note), so it's exercised for
        # wiring/no-crash only, not for a label it alone reveals.
        flag_to_label = {
            "--show-gpu-api": "hipLaunchKernel",
            "--show-rocprofsys-internals": "gotcha_call",
            "--show-compiler-runtime": "posix_memalign",
        }
        for flag, own_label in flag_to_label.items():
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as tmp:
                    dest = os.path.join(tmp, "calltree.txt")
                    calltree.main([NOISE_DIR, "-o", dest, flag])
                    with open(dest) as f:
                        report = f.read()
                self.assertIn(own_label, report)
                for other_label in set(flag_to_label.values()) - {own_label}:
                    self.assertNotIn(other_label, report)

        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            calltree.main([NOISE_DIR, "-o", dest, "--show-mpi-internals"])
            self.assertTrue(os.path.isfile(dest))

    def test_extra_noise_config_flag_splices_a_configured_wrapper_noise_label(self):
        # The trace pipeline's "other" tag is purely category-derived (see
        # stage3_rocprofsys_trace.py's module docstring: "nothing here for a user to tune") --
        # --extra-noise-config's add/remove only ever reaches the delegated wrapper_noise/
        # compiler_runtime_noise/wrapper_branch_noise patterns, so this configures jacobi_sweep
        # as wrapper_noise instead: spliced out (fold=False), its child MPI_Barrier reparented
        # up to main rather than disappearing with it.
        self.addCleanup(stage6_noise_config.configure, None)
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            config_path = write_noise_config(tmp, {"add": {"wrapper_noise": ["jacobi_sweep"]}})
            calltree.main([TWO_RANK_DIR, "-o", dest, "--extra-noise-config", config_path])
            with open(dest) as f:
                report = f.read()
        self.assertNotIn("jacobi_sweep", report)
        self.assertIn("main", report)
        self.assertIn("MPI_Barrier", report)


class HelpTextTests(unittest.TestCase):
    def test_help_leads_with_explanation(self):
        assert_help_leads_with_explanation(self, calltree, "rocprofiler-systems/en/latest")


if __name__ == "__main__":
    unittest.main()
