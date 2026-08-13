import importlib.util
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "extract_calltree.py")

# extract_calltree.py does a plain top-level "import extract_CPU_hotspots"/
# "import extract_GPU_hotspots", relying on its own directory being on
# sys.path -- true automatically when run directly, but not when loaded here
# by explicit file path, so replicate that manually (same as
# test_extract_hotspots.py).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("extract_calltree", MODULE_PATH)
ct_tool = importlib.util.module_from_spec(spec)
sys.modules["extract_calltree"] = ct_tool
spec.loader.exec_module(ct_tool)

FILTERS_DIR = os.path.join(FIXTURES, "calltree_sampling_filters")
FALLBACK_DIR = os.path.join(FIXTURES, "calltree_sampling_fallback")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_sampling_kernel_anchor")
MULTI_RANK_DIR = os.path.join(FIXTURES, "calltree_sampling_multi_rank")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")


def render(run_dir, **kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "calltree.txt")
        return ct_tool.write_report(run_dir, dest, **kwargs)


def _line_for(report, label):
    for line in report.splitlines():
        if label in line:
            return line
    raise AssertionError(f"no rendered line for {label!r} found")


class GpuNoiseTierTests(unittest.TestCase):
    def test_hidden_by_default(self):
        report = render(FILTERS_DIR)
        self.assertNotIn("hipLaunchKernel", report)
        self.assertNotIn("rocprofiler::hip::something", report)

    def test_shown_with_flag(self):
        report = render(FILTERS_DIR, show_gpu_api=True)
        self.assertIn("hipLaunchKernel", report)
        self.assertIn("rocprofiler::hip::something", report)  # broadened substring match, not just prefix

    def test_omp_target_offload_internals_hidden_by_default(self):
        # __tgt_target_kernel (LLVM libomptarget's launch entry point) and the
        # AMDGPU-offload-plugin chain beneath it, plus AMD's GPU-kernel-JIT-
        # compilation noise (clang/LLVM/comgr) -- all confirmed unfiltered in
        # real test_apps HPC data before this fix.
        report = render(FILTERS_DIR)
        self.assertNotIn("__tgt_target_kernel", report)
        self.assertNotIn("llvm::omp::target::plugin::GenericPluginTy::load_binary", report)
        self.assertNotIn("clang::CodeGen::mergeDefaultFunctionDefinition", report)

    def test_omp_target_offload_internals_shown_with_flag(self):
        report = render(FILTERS_DIR, show_gpu_api=True)
        self.assertIn("__tgt_target_kernel", report)
        self.assertIn("llvm::omp::target::plugin::GenericPluginTy::load_binary", report)
        self.assertIn("clang::CodeGen::mergeDefaultFunctionDefinition", report)


class RocprofsysWrapperSpliceTests(unittest.TestCase):
    def test_wrapper_frames_hidden_by_default(self):
        report = render(FILTERS_DIR)
        self.assertNotIn("__libc_start_main", report)
        self.assertNotIn("rocprofsys_main", report)
        self.assertNotIn("gotcha_wrapper_call", report)
        self.assertNotIn("tim::wrapped_call", report)

    def test_whole_ancestor_chain_wrapper_promotes_new_root(self):
        # __libc_start_main -> rocprofsys_main -> main: both ancestors are
        # wrapper frames, so main itself must become a top-level root once
        # they're spliced out -- not just have its parent link changed.
        ranks = ct_tool.load_rank_trees(FILTERS_DIR, show_rocprofsys_internals=False)
        self.assertEqual(len(ranks), 1)
        _rank_key, _rows, roots = ranks[0]
        # "start_thread" (thread 3) is also its own independent root (a
        # separate, unrelated background thread) -- see
        # UntetheredThreadRootGpuPropagationTests for why it's still hidden
        # at render time despite being a real root here.
        self.assertEqual({r["label"] for r in roots}, {"main", "start_thread"})

    def test_mid_tree_wrapper_reparents_its_child_not_drops_it(self):
        # gotcha_wrapper_call -> tim::wrapped_call -> real_child_under_wrapper:
        # both wrapper frames removed, but the real leaf underneath must still
        # appear, reparented up to "main" (spliced, not pruned).
        report = render(FILTERS_DIR)
        self.assertIn("real_child_under_wrapper", report)

    def test_wrapper_frames_shown_with_flag(self):
        report = render(FILTERS_DIR, show_rocprofsys_internals=True)
        self.assertIn("__libc_start_main", report)
        self.assertIn("gotcha_wrapper_call", report)


class WrapperContaminatedBranchTests(unittest.TestCase):
    # "std::pair<std::_Rb_tree_iterator<int>, bool> noise_top" is a sibling
    # of main's other real children (foo_normal_call etc.) with get_library
    # nested one level inside it -- real shape confirmed via amd test_apps
    # HPC data: rocprof-sys/GOTCHA's own startup bookkeeping is mostly
    # generic std::set/std::map container internals that don't match
    # ROCPROFSYS_WRAPPER_SUBSTRINGS on their own, only a frame like
    # get_library genuinely deep inside does -- see mark_wrapper_
    # contaminated_branches()'s own docstring.
    def test_contaminated_branch_hidden_by_default(self):
        report = render(FILTERS_DIR)
        self.assertNotIn("noise_top", report)
        self.assertNotIn("get_library", report)

    def test_contaminated_branch_shown_with_flag(self):
        report = render(FILTERS_DIR, show_rocprofsys_internals=True)
        self.assertIn("noise_top", report)
        self.assertIn("get_library", report)

    def test_real_sibling_branches_unaffected(self):
        report = render(FILTERS_DIR)
        self.assertIn("foo_normal_call", report)


class UntetheredThreadRootGpuPropagationTests(unittest.TestCase):
    # calltree_sampling_filters also has a thread-3 "start_thread" root with
    # no parent at all (DEPTH resets to 0, same shape a real background
    # HIP/ROCr event-loop thread samples as) whose own content -- past
    # rocprof-sys's pthread_create wrapper hop -- is ROCm-runtime noise
    # (rocr::os::ThreadTrampoline). Confirmed against real Heat_Convection_Solver
    # data: without this propagation, the untethered root itself rendered
    # unfiltered (with a large SELF/TOTAL time) even though its own child was
    # already correctly GPU-classified.
    def test_untethered_root_hidden_by_default(self):
        report = render(FILTERS_DIR)
        self.assertNotIn("start_thread", report)
        self.assertNotIn("ThreadTrampoline", report)

    def test_untethered_root_shown_with_gpu_api_flag(self):
        report = render(FILTERS_DIR, show_gpu_api=True)
        self.assertIn("start_thread", report)
        self.assertIn("ThreadTrampoline", report)

    def test_untethered_root_reclassified_regardless_of_wrapper_visibility(self):
        # --show-rocprofsys-internals alone (not --show-gpu-api) must NOT
        # reveal this subtree -- it's GPU noise at heart, gated by its own flag.
        report = render(FILTERS_DIR, show_rocprofsys_internals=True)
        self.assertNotIn("start_thread", report)


class MpiCollapseTierTests(unittest.TestCase):
    def test_first_mpi_frame_shown_deeper_internals_hidden(self):
        report = render(FILTERS_DIR)
        self.assertIn("mpi_allreduce_f08ts_", report)
        self.assertNotIn("MPIR_Allreduce_cdesc", report)
        self.assertNotIn("PMPI_Allreduce", report)

    def test_deeper_internals_shown_with_flag(self):
        report = render(FILTERS_DIR, show_mpi_internals=True)
        self.assertIn("MPIR_Allreduce_cdesc", report)
        self.assertIn("PMPI_Allreduce", report)

    def test_open_mpi_prefixes_matched_probably(self):
        # ompi_/opal_/orte_ -- no real Open MPI test_apps capture exists yet,
        # added as a "most probable" list per real Open MPI naming
        # conventions. startswith-based, so this must NOT match a Open-MPI
        # opaque-handle typename appearing mid-string inside rocprof-sys's own
        # generic GOTCHA-wrapper template signature.
        self.assertTrue(ct_tool.is_mpi_territory("ompi_request_complete"))
        self.assertTrue(ct_tool.is_mpi_territory("opal_progress"))
        self.assertTrue(ct_tool.is_mpi_territory("orte_grpcomm_base_pack"))
        self.assertFalse(ct_tool.is_mpi_territory(
            "tim::component::gotcha<101ul, int, ompi_group_t**>::construct"
        ))


class CompilerRuntimeTierTests(unittest.TestCase):
    def test_hidden_by_default_whole_subtree(self):
        report = render(FILTERS_DIR)
        self.assertNotIn("posix_memalign", report)
        self.assertNotIn("should_not_appear_child", report)  # PRUNE: child hidden too, not just the node

    def test_shown_with_flag(self):
        report = render(FILTERS_DIR, show_compiler_runtime=True)
        self.assertIn("posix_memalign", report)
        self.assertIn("should_not_appear_child", report)


class ShowAllInternalsTests(unittest.TestCase):
    def test_equivalent_to_all_four_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            ct_tool.main([FILTERS_DIR, "-o", dest, "--show-all-internals"])
            with open(dest) as f:
                report = f.read()
        for label in ("hipLaunchKernel", "__libc_start_main", "MPIR_Allreduce_cdesc", "posix_memalign"):
            self.assertIn(label, report)


class SamplingMissingFallbackTests(unittest.TestCase):
    def test_rank_without_sampling_file_falls_back_to_wall_clock(self):
        ranks = ct_tool.load_rank_trees(FALLBACK_DIR, show_rocprofsys_internals=False)
        self.assertEqual(len(ranks), 2)
        labels_by_rank = {rk: {r["label"] for r in rows} for rk, rows, _roots in ranks}
        self.assertIn({"main", "sampled_leaf"}, labels_by_rank.values())
        self.assertIn({"main_fallback", "instrumented_leaf"}, labels_by_rank.values())

    def test_fallback_rank_renders_correctly(self):
        report = render(FALLBACK_DIR)
        self.assertIn("main_fallback", report)
        self.assertIn("instrumented_leaf", report)


class KernelAnchorBroadeningTests(unittest.TestCase):
    def test_namespace_qualified_and_cray_acc_both_resolve_as_anchors(self):
        report = render(KERNEL_ANCHOR_DIR)
        self.assertIn("[GPU kernels -- rocprofv3", report)
        self.assertNotIn("no owning subroutine or launch call site found", report)
        # compute_a issued 100/150 launch calls, compute_b issued 50/150.
        self.assertIn("~67% estimate: this site issued 100/150", report)
        self.assertIn("~33% estimate: this site issued 50/150", report)
        i_a = report.index("compute_a")
        i_b = report.index("compute_b")
        i_kernel_a = report.index("[GPU kernels -- rocprofv3", i_a)
        self.assertTrue(i_a < i_kernel_a < i_b)


class AggregationTests(unittest.TestCase):
    # calltree_sampling_multi_rank: three ranks share the same "main" ->
    # "compute_stencil" structure with different self-times (1.0/3.0/5.0s),
    # plus a "rare_error_path" only rank 9203 hits -- exercises both the
    # avg/std_dev/min/max load-balance math and the "a rank that never
    # reached this node counts as 0, not omitted" rule.
    def test_no_per_rank_sections(self):
        report = render(MULTI_RANK_DIR)
        self.assertNotIn("=== Rank", report)

    def test_header_states_ranks_aggregated(self):
        report = render(MULTI_RANK_DIR)
        self.assertIn("ranks aggregated: 3", report)

    def test_load_balance_columns_for_node_present_on_every_rank(self):
        report = render(MULTI_RANK_DIR)
        line = _line_for(report, "compute_stencil")
        self.assertIn("20.0", line)        # CALLS avg: (10+20+30)/3
        self.assertIn("3.000000", line)    # SELF-AVG: (1+3+5)/3
        self.assertIn("1.632993", line)    # SELF-STD: pstdev([1,3,5])
        self.assertIn("1.000000", line)    # SELF-MIN
        self.assertIn("5.000000", line)    # SELF-MAX
        # TOTAL-AVG is also 3.000000 here -- already covered by SELF-AVG's
        # identical value in this fixture (self% is 100 on every rank), not
        # asserted separately to avoid a coincidental-match false positive.

    def test_load_balance_columns_for_node_missing_on_some_ranks(self):
        # rare_error_path only exists on rank 9203 -- ranks 9201/9202 count
        # as 0 for it, not omitted, so the average is pulled down accordingly.
        report = render(MULTI_RANK_DIR)
        line = _line_for(report, "rare_error_path")
        self.assertIn("0.3", line)         # CALLS avg: (0+0+1)/3
        self.assertIn("0.200000", line)    # SELF-AVG: (0+0+0.6)/3
        self.assertIn("0.282843", line)    # SELF-STD: pstdev([0,0,0.6])
        self.assertIn("0.600000", line)    # SELF-MAX


class MainCliTests(unittest.TestCase):
    def test_missing_directory_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            ct_tool.main(["/no/such/directory"])

    def test_empty_input_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            ct_tool.main([EMPTY_DIR])

    def test_end_to_end_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            ct_tool.main([FILTERS_DIR, "-o", dest, "--max-depth", "1"])
            self.assertTrue(os.path.isfile(dest))


if __name__ == "__main__":
    unittest.main()
