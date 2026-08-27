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
from _tools_test_helpers import assert_extract_tool_cli_contract  # noqa: E402

ct_tool = load_module_by_path("extract_wallclock_calltree", "tools", "extract_wallclock_calltree.py")

import stage6_noise_config  # noqa: E402  (needs sys.path insert above first)

MPI_2RANK_DIR = os.path.join(FIXTURES, "mpi_2rank")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_anchor")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")


class ResolveTwoDirsTests(unittest.TestCase):
    # extract_wallclock_calltree.py imports resolve_two_dirs() from stage1_run_dirs.py (which has
    # its own thorough direct tests) -- this just confirms the import wires through correctly.
    def test_detects_paired_subdirs(self):
        cpu_dir, gpu_dir = ct_tool.resolve_two_dirs(KERNEL_ANCHOR_DIR, None)
        self.assertEqual(cpu_dir, os.path.join(KERNEL_ANCHOR_DIR, "rocprof-sys"))
        self.assertEqual(gpu_dir, os.path.join(KERNEL_ANCHOR_DIR, "rocprofv3"))

    def test_falls_back_to_run_dir_itself_when_flat(self):
        cpu_dir, gpu_dir = ct_tool.resolve_two_dirs(MPI_2RANK_DIR, None)
        self.assertEqual(cpu_dir, MPI_2RANK_DIR)
        self.assertIsNone(gpu_dir)


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


if __name__ == "__main__":
    unittest.main()
