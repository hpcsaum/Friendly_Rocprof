import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")

# extract_calltree.py does a plain top-level "from stage1_run_dirs import ...",
# relying on its own directory being on sys.path -- true automatically when run
# directly, but not when loaded here by explicit file path, so replicate that
# manually (same as test_extract_hotspots.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402
from _tools_test_helpers import assert_extract_tool_cli_contract  # noqa: E402

ct_tool = load_module_by_path("extract_calltree", "tools", "extract_calltree.py")

import stage6_noise_config  # noqa: E402  (needs sys.path insert above first)

FILTERS_DIR = os.path.join(FIXTURES, "calltree_sampling_filters")
MULTI_RANK_DIR = os.path.join(FIXTURES, "calltree_sampling_multi_rank")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")


class HeaderProseTests(unittest.TestCase):
    def test_header_states_ranks_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            report = ct_tool.write_report(MULTI_RANK_DIR, None, dest)
        self.assertIn("MPI ranks: 3", report)


class ShowAllInternalsTests(unittest.TestCase):
    def test_equivalent_to_all_four_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            ct_tool.main([FILTERS_DIR, "-o", dest, "--show-all-internals"])
            with open(dest) as f:
                report = f.read()
        for label in ("hipLaunchKernel", "__libc_start_main", "MPIR_Allreduce_cdesc", "posix_memalign"):
            self.assertIn(label, report)


class MainCliTests(unittest.TestCase):
    # The baseline CLI contract every extract_*_calltree.py tool shares (missing/empty input,
    # end-to-end write, explicit two directories, --extra-noise-config) -- see
    # assert_extract_tool_cli_contract()'s own docstring for the full shape.
    def test_cli_contract(self):
        self.addCleanup(stage6_noise_config.configure, None)
        assert_extract_tool_cli_contract(self, ct_tool, FILTERS_DIR, EMPTY_DIR, "foo_normal_call")


if __name__ == "__main__":
    unittest.main()
