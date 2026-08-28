"""Tests for extract_calltree.py, the sampling-based call-tree report tool.

HeaderProseTests             -- multi-rank header states how many MPI ranks were aggregated
GpuKernelNestingTests        -- main()/write_report() wiring reaches a real paired rocprofv3 dir
ShowAllInternalsTests        -- --show-all-internals is equivalent to enabling all four individual show flags
IndividualNoiseTierFlagsTests -- each of the four --show-* flags works independently through main()
MainCliTests                 -- baseline extract-tool CLI contract, via assert_extract_tool_cli_contract()
"""

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
from _tools_test_helpers import assert_extract_tool_cli_contract, assert_help_leads_with_explanation  # noqa: E402

ct_tool = load_module_by_path("extract_calltree", "tools", "extract_calltree.py")

import stage6_noise_config  # noqa: E402  (needs sys.path insert above first)

FILTERS_DIR = os.path.join(FIXTURES, "calltree_sampling_filters")
MULTI_RANK_DIR = os.path.join(FIXTURES, "calltree_sampling_multi_rank")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_sampling_kernel_anchor")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")


class HeaderProseTests(unittest.TestCase):
    def test_header_states_ranks_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            report = ct_tool.write_report(MULTI_RANK_DIR, None, dest)
        self.assertIn("MPI ranks: 3", report)


class GpuKernelNestingTests(unittest.TestCase):
    # build_calltree_view()'s own kernel-anchor attribution math is exhaustively unit-tested in
    # test_stage4_rocprofsys_sample_tree.py and end-to-end against stage5_calltree_view.py directly
    # in test_stage5_calltree_view.py::KernelAnchorBroadeningTests -- this only confirms this
    # tool's own main()/write_report() wiring actually reaches a real paired rocprofv3 directory
    # and nests the GPU kernel data into the written report, not just an unpaired CPU-only run.
    def test_kernel_data_nested_under_its_launching_subroutine_in_the_written_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            ct_tool.main([KERNEL_ANCHOR_DIR, "-o", dest])
            with open(dest) as f:
                report = f.read()
        self.assertIn("[GPU kernels -- rocprofv3", report)
        self.assertNotIn("no owning subroutine or launch call site found", report)


class ShowAllInternalsTests(unittest.TestCase):
    def test_equivalent_to_all_four_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            ct_tool.main([FILTERS_DIR, "-o", dest, "--show-all-internals"])
            with open(dest) as f:
                report = f.read()
        for label in ("hipLaunchKernel", "__libc_start_main", "MPIR_Allreduce_cdesc", "posix_memalign"):
            self.assertIn(label, report)


class IndividualNoiseTierFlagsTests(unittest.TestCase):
    # ShowAllInternalsTests above confirms --show-all-internals turns on all four tiers together;
    # this confirms main() also wires each flag independently (not only reachable bundled), by
    # checking each flag alone reveals only its own tier's label, not the other three.
    FLAG_TO_LABEL = {
        "--show-gpu-api": "hipLaunchKernel",
        "--show-rocprofsys-internals": "__libc_start_main",
        "--show-mpi-internals": "MPIR_Allreduce_cdesc",
        "--show-compiler-runtime": "posix_memalign",
    }

    def test_each_flag_alone_reveals_only_its_own_tier(self):
        for flag, own_label in self.FLAG_TO_LABEL.items():
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as tmp:
                    dest = os.path.join(tmp, "calltree.txt")
                    ct_tool.main([FILTERS_DIR, "-o", dest, flag])
                    with open(dest) as f:
                        report = f.read()
                self.assertIn(own_label, report)
                other_labels = set(self.FLAG_TO_LABEL.values()) - {own_label}
                for other_label in other_labels:
                    self.assertNotIn(other_label, report)


class MainCliTests(unittest.TestCase):
    # The baseline CLI contract every extract_*_calltree.py tool shares (missing/empty input,
    # end-to-end write, explicit two directories, --extra-noise-config) -- see
    # assert_extract_tool_cli_contract()'s own docstring for the full shape.
    def test_cli_contract(self):
        self.addCleanup(stage6_noise_config.configure, None)
        assert_extract_tool_cli_contract(self, ct_tool, FILTERS_DIR, EMPTY_DIR, "foo_normal_call")


class HelpTextTests(unittest.TestCase):
    def test_help_leads_with_explanation(self):
        assert_help_leads_with_explanation(self, ct_tool, "rocprofiler-systems/en/latest")


if __name__ == "__main__":
    unittest.main()
