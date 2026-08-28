"""Tests for extract_wallclock_calltree.py, the timemory-text-driven call tree tool (exact where
GOTCHA-instrumented, falling back to sampling per rank; GPU kernel placement is name-match-then-
structural-guess, not per-dispatch-exact like extract_trace_calltree.py).

HeaderProseTests -- report header states the aggregated rank count
MainCliTests     -- the shared extract_*_calltree.py CLI contract (assert_extract_tool_cli_contract()),
                    plus --show-gpu-api and --max-depth's own real effect on the written report
"""

import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")

# extract_wallclock_calltree.py does a plain top-level "from stage1_run_dirs import ...",
# relying on its own directory being on sys.path -- true automatically when run
# directly, but not when loaded here by explicit file path, so replicate that
# manually (same as test_extract_hotspots.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402
from _tools_test_helpers import assert_extract_tool_cli_contract, assert_help_leads_with_explanation  # noqa: E402

ct_tool = load_module_by_path("extract_wallclock_calltree", "tools", "extract_wallclock_calltree.py")

import stage6_noise_config  # noqa: E402  (needs sys.path insert above first)

MPI_2RANK_DIR = os.path.join(FIXTURES, "mpi_2rank")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")


class HeaderProseTests(unittest.TestCase):
    def test_header_states_ranks_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "wallclock_calltree.txt")
            report = ct_tool.write_report(MPI_2RANK_DIR, None, dest)
        self.assertIn("MPI ranks: 2", report)


class MainCliTests(unittest.TestCase):
    # The baseline CLI contract every extract_*_calltree.py tool shares (missing/empty input,
    # end-to-end write, explicit two directories, --extra-noise-config) -- see
    # assert_extract_tool_cli_contract()'s own docstring for the full shape.
    def test_cli_contract(self):
        self.addCleanup(stage6_noise_config.configure, None)
        assert_extract_tool_cli_contract(self, ct_tool, MPI_2RANK_DIR, EMPTY_DIR, "compute_stencil")

    def test_show_gpu_api_flag_reveals_the_gpu_subtree(self):
        # hipMemcpy is gpu-api-tagged and hidden by default -- see
        # test_stage5_wallclock_calltree_view.py::RenderTreeTests's equivalent build_calltree_view()
        # -level check; this confirms main()'s own --show-gpu-api wiring reaches the same result.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "wallclock_calltree.txt")
            ct_tool.main([MPI_2RANK_DIR, "-o", dest, "--show-gpu-api"])
            with open(dest) as f:
                report = f.read()
        self.assertIn("hipMemcpy", report)

    def test_max_depth_flag_actually_truncates_the_tree(self):
        # assert_extract_tool_cli_contract()'s own "end_to_end_writes_file" subTest already passes
        # --max-depth 1 but only checks the file was written -- this confirms the flag actually
        # truncates, not just that main() accepts it without crashing.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "wallclock_calltree.txt")
            ct_tool.main([MPI_2RANK_DIR, "-o", dest, "--max-depth", "0"])
            with open(dest) as f:
                report = f.read()
        self.assertIn("hidden below this point", report)
        self.assertNotIn("compute_stencil", report)


class HelpTextTests(unittest.TestCase):
    def test_help_leads_with_explanation(self):
        assert_help_leads_with_explanation(self, ct_tool, "rocprofiler-systems/en/latest")


if __name__ == "__main__":
    unittest.main()
