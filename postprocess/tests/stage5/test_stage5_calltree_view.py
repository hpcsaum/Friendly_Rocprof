import json
import os
import re
import sys
import tempfile
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

from stage1_run_dirs import resolve_run_dirs  # noqa: E402  (needs sys.path insert above first)
from stage5_calltree_view import build_calltree_view, strip_wrapper_noise  # noqa: E402
from stage4_rocprofsys_sample_tree import load_rank_trees  # noqa: E402
import stage6_noise_config  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
FILTERS_DIR = os.path.join(FIXTURES, "calltree_sampling_filters")
FALLBACK_DIR = os.path.join(FIXTURES, "calltree_sampling_fallback")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_sampling_kernel_anchor")
MULTI_RANK_DIR = os.path.join(FIXTURES, "calltree_sampling_multi_rank")


def render(run_dir, **kwargs):
    cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
    view = build_calltree_view(run_dir, cpu_dir, gpu_dir, **kwargs)
    return view["tree_text"] + view["fallback_text"]


def _line_for(report, label):
    for line in report.splitlines():
        if label in line:
            return line
    raise AssertionError(f"no rendered line for {label!r} found")


_NUMERIC_SUFFIX_RE = re.compile(r"(?:\s{2,}-?[\d.]+)+\s*$")


def _labels_only(report):
    """Strips each physical line's trailing numeric-cell columns (if any), then joins every line
    back together with no separator -- reconstructs each row's own label text contiguously, so a
    test can search for a label substring regardless of exactly where
    stage5_tree_render.wrap_leading_labels() cut a too-long label across physical lines."""
    return "".join(_NUMERIC_SUFFIX_RE.sub("", line) for line in report.splitlines())


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
        # This label is long enough that stage5_tree_render.wrap_leading_labels() hard-wraps it
        # across 2 physical lines -- reconstruct labels-only text first so the substring search
        # sees it whole.
        flat = _labels_only(report)
        self.assertIn("llvm::omp::target::plugin::GenericPluginTy::load_binary", flat)
        self.assertIn("clang::CodeGen::mergeDefaultFunctionDefinition", flat)


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
        ranks = load_rank_trees(
            FILTERS_DIR, "sampling_wall_clock-*.txt", "wall_clock-*.txt", postprocess=strip_wrapper_noise
        )
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


class OtherTagConfigTests(unittest.TestCase):
    def tearDown(self):
        stage6_noise_config.configure(None)

    def test_other_tagged_node_spliced_out_with_fold(self):
        # foo_normal_call is a real, otherwise-untagged sibling branch (see
        # test_real_sibling_branches_unaffected above) -- confirms "other"'s own default
        # treatment (splice, fold=True) actually reaches this tool via strip_wrapper_noise().
        report_default = render(FILTERS_DIR)
        self.assertIn("foo_normal_call", report_default)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "noise_config.json")
            with open(path, "w") as f:
                json.dump({"add": {"other": ["foo_normal_call"]}}, f)
            stage6_noise_config.configure(path)
            report_configured = render(FILTERS_DIR)
        self.assertNotIn("foo_normal_call", report_configured)


class UntetheredThreadRootGpuPropagationTests(unittest.TestCase):
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


class CompilerRuntimeTierTests(unittest.TestCase):
    def test_hidden_by_default_whole_subtree(self):
        report = render(FILTERS_DIR)
        self.assertNotIn("posix_memalign", report)
        self.assertNotIn("should_not_appear_child", report)  # PRUNE: child hidden too, not just the node

    def test_shown_with_flag(self):
        report = render(FILTERS_DIR, show_compiler_runtime=True)
        self.assertIn("posix_memalign", report)
        self.assertIn("should_not_appear_child", report)


class SamplingMissingFallbackTests(unittest.TestCase):
    def test_rank_without_sampling_file_falls_back_to_wall_clock(self):
        ranks = load_rank_trees(
            FALLBACK_DIR, "sampling_wall_clock-*.txt", "wall_clock-*.txt", postprocess=strip_wrapper_noise
        )
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
        # compute_a issued 100/150 launch calls, compute_b issued 50/150. This bracketed label is
        # long enough that wrap_leading_labels() may hard-wrap it across physical lines --
        # reconstruct labels-only text first so the substring search sees it whole regardless of
        # the exact cut point.
        flat = _labels_only(report)
        self.assertIn("~67% estimate: this site issued 100/150", flat)
        self.assertIn("~33% estimate: this site issued 50/150", flat)
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

    def test_load_balance_columns_for_node_present_on_every_rank(self):
        report = render(MULTI_RANK_DIR)
        line = _line_for(report, "compute_stencil")
        self.assertIn("20.0", line)        # CALLS avg: (10+20+30)/3
        self.assertIn("3.000000", line)    # SELF-AVG: (1+3+5)/3
        self.assertIn("1.632993", line)    # SELF-STD: pstdev([1,3,5])
        self.assertIn("1.000000", line)    # SELF-MIN
        self.assertIn("5.000000", line)    # SELF-MAX

    def test_load_balance_columns_for_node_missing_on_some_ranks(self):
        # rare_error_path only exists on rank 9203 -- ranks 9201/9202 count
        # as 0 for it, not omitted, so the average is pulled down accordingly.
        report = render(MULTI_RANK_DIR)
        line = _line_for(report, "rare_error_path")
        self.assertIn("0.3", line)         # CALLS avg: (0+0+1)/3
        self.assertIn("0.200000", line)    # SELF-AVG: (0+0+0.6)/3
        self.assertIn("0.282843", line)    # SELF-STD: pstdev([0,0,0.6])
        self.assertIn("0.600000", line)    # SELF-MAX


if __name__ == "__main__":
    unittest.main()
