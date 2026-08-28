"""Tests for stage5_wallclock_calltree_view.py's build_calltree_view() -- the traced/wall_clock-based
calltree tool's (extract_wallclock_calltree.py) single --show-gpu-api prune tier, .kd-artifact
filtering, GPU-kernel-anchor attribution, and cross-rank aggregation. Renders through
render()/labels_only(), thin wrappers around the shared _stage5_test_helpers.py plumbing also used
by test_stage5_calltree_view.py.

RenderTreeTests           -- --show-gpu-api gating, max-depth truncation, tree connectors
KdArtifactFilteringTests  -- .kd-suffixed rows hidden by default, shown with --show-gpu-api
KernelIntegrationTests    -- single/multiple kernel-anchor attribution, no-anchor fallback section
AggregationTests          -- cross-rank merge, no per-rank sections, no node duplication
OtherTagConfigTests       -- user-configured "other" tags reach this tool via _splice_other()
NoDataFoundTests          -- SystemExit when no timemory text table is found under run_dir
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _stage5_test_helpers import labels_only, render_calltree_view  # noqa: E402

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

from stage5_wallclock_calltree_view import build_calltree_view  # noqa: E402
import stage6_noise_config  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
MPI_2RANK_DIR = os.path.join(FIXTURES, "mpi_2rank")
GPU_API_NESTED_CHAIN_DIR = os.path.join(FIXTURES, "gpu_api_nested_chain")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_anchor")
KERNEL_MULTI_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_multi_anchor")
KERNEL_NO_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_no_anchor")
KD_ARTIFACT_DIR = os.path.join(FIXTURES, "calltree_kd_artifact")


def render(run_dir, **kwargs):
    return render_calltree_view(build_calltree_view, run_dir, **kwargs)


class RenderTreeTests(unittest.TestCase):
    def test_gpu_api_hidden_by_default(self):
        report = render(MPI_2RANK_DIR)
        self.assertNotIn("hipMemcpy", report)

    def test_gpu_api_shown_with_flag(self):
        report = render(MPI_2RANK_DIR, show_gpu_api=True)
        self.assertIn("hipMemcpy", report)

    def test_gpu_api_chain_pruned_wholesale(self):
        # gpu_api_nested_chain: hipStreamCreate -> hip::hipStreamCreate(...) ->
        # hip::ihipStreamCreate(...) -- none should appear by default, all three
        # should appear when --show-gpu-api is passed.
        default_report = render(GPU_API_NESTED_CHAIN_DIR)
        for label in ("hipStreamCreate", "hip::hipStreamCreate", "hip::ihipStreamCreate"):
            self.assertNotIn(label, default_report)
        shown_report = render(GPU_API_NESTED_CHAIN_DIR, show_gpu_api=True)
        for label in ("hipStreamCreate", "hip::hipStreamCreate", "hip::ihipStreamCreate"):
            self.assertIn(label, shown_report)

    def test_max_depth_truncates_with_stated_count(self):
        report = render(MPI_2RANK_DIR, max_depth=0)
        self.assertIn("hidden below this point", report)
        self.assertIn("1 more node(s)", report)  # compute_stencil only -- hipMemcpy is gpu-hidden regardless
        self.assertNotIn("compute_stencil", report)

    def test_max_depth_hidden_count_matches_show_gpu_api(self):
        report = render(MPI_2RANK_DIR, max_depth=0, show_gpu_api=True)
        self.assertIn("2 more node(s)", report)

    def test_no_max_depth_prints_whole_tree(self):
        report = render(MPI_2RANK_DIR)
        self.assertIn("compute_stencil", report)
        self.assertNotIn("hidden below this point", report)

    def test_real_columns_not_bracketed_string(self):
        # Wiring check: format_aligned_rows()'s own column-formatting behavior (incl. the
        # "[calls=" old-format regression) is exhaustively covered directly in
        # test_stage5_tree_render.py -- this only confirms this tool's report actually uses the
        # real REPORT_HEADERS constant, not a leftover hand-rolled string.
        report = render(MPI_2RANK_DIR)
        self.assertIn("calls", report)
        self.assertIn("self-avg(s)", report)
        self.assertIn("total-avg(s)", report)
        self.assertNotIn("[calls=", report)  # old per-line bracketed format, must be gone

    def test_tree_connectors_present(self):
        # mpi_2rank's main has only one child, so it's rendered "└── " (last
        # child) -- "├── " needs a node with 2+ children, like gpu_api_nested_chain's
        # main (compute_stencil + hipStreamCreate).
        report = render(GPU_API_NESTED_CHAIN_DIR, show_gpu_api=True)
        self.assertIn("├── ", report)
        self.assertIn("└── ", report)


class KdArtifactFilteringTests(unittest.TestCase):
    def test_kd_suffixed_row_hidden_by_default(self):
        report = render(KD_ARTIFACT_DIR)
        self.assertIn("compute_stencil", report)
        self.assertNotIn("some_kernel_name.kd", report)

    def test_kd_suffixed_row_shown_with_flag(self):
        report = render(KD_ARTIFACT_DIR, show_gpu_api=True)
        self.assertIn("some_kernel_name.kd", report)


class KernelIntegrationTests(unittest.TestCase):
    def test_single_anchor_gets_full_attribution(self):
        report = render(KERNEL_ANCHOR_DIR)
        self.assertIn("[GPU kernels -- rocprofv3]", report)
        self.assertIn("JacobiIterationKernel", report)
        i_parent = report.index("compute_stencil")
        i_kernel = report.index("[GPU kernels -- rocprofv3]")
        i_leaf = report.index("JacobiIterationKernel")
        self.assertTrue(i_parent < i_kernel < i_leaf)  # nested under compute_stencil, not top-level
        self.assertNotIn("no launch call site found", report)

    def test_multiple_anchors_split_proportionally(self):
        # Wiring check: the 60%/40% (300/500, 200/500) split arithmetic itself is exhaustively
        # covered directly against KernelAnchorAttributionTests in
        # test_stage4_rocprofsys_sample_tree.py -- this only confirms both anchors' labels
        # actually land at the right two tree positions in this tool's rendered report.
        report = render(KERNEL_MULTI_ANCHOR_DIR)
        # fixture: compute_a issued 300 launch calls, compute_b issued 200 (of 500 total). This
        # bracketed label is long enough that wrap_leading_labels() may hard-wrap it across
        # physical lines -- reconstruct labels-only text first so the substring search sees it
        # whole regardless of the exact cut point.
        flat = labels_only(report)
        self.assertIn("~60% estimate: this site issued 300/500", flat)
        self.assertIn("~40% estimate: this site issued 200/500", flat)
        # both anchors get their own nested kernel breakdown, not one shared full total
        self.assertEqual(report.count("JacobiIterationKernel"), 2)

    def test_no_anchor_falls_back_to_top_level_section(self):
        report = render(KERNEL_NO_ANCHOR_DIR)
        self.assertIn("=== GPU kernels (rocprofv3) -- no owning subroutine or launch call site found in CPU tree ===", report)
        self.assertIn("JacobiIterationKernel", report)
        # the fallback section's kernel data must NOT also appear nested inside a rank tree
        i_fallback = report.index("=== GPU kernels")
        self.assertNotIn("[GPU kernels -- rocprofv3", report[:i_fallback])


class AggregationTests(unittest.TestCase):
    # mpi_2rank: two ranks (2001, 2002), each with its own real timing for
    # "main"/"compute_stencil" -- exercises the cross-rank merge + load
    # balance columns end to end (stage4_rocprofsys_sample_tree's own math is tested
    # directly and more exhaustively in test_stage4_rocprofsys_sample_tree.py).
    def test_no_per_rank_sections(self):
        report = render(MPI_2RANK_DIR)
        self.assertNotIn("=== Rank", report)

    def test_single_merged_main_node_not_duplicated_per_rank(self):
        report = render(MPI_2RANK_DIR)
        self.assertEqual(report.count("\nmain "), 1)


class OtherTagConfigTests(unittest.TestCase):
    def tearDown(self):
        stage6_noise_config.configure(None)

    def test_other_tagged_node_spliced_out_with_fold(self):
        # This tool has no wrapper-noise handling of its own -- confirms "other"'s default
        # treatment (splice, fold=True) still lands here via _splice_other().
        # compute_stencil is main's real, otherwise-untagged child (mpi_2rank's tree shape is
        # covered directly in test_stage4_rocprofsys_sample_tree.py::LoadRankTreesTests).
        report_default = render(MPI_2RANK_DIR)
        self.assertIn("compute_stencil", report_default)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "noise_config.json")
            with open(path, "w") as f:
                json.dump({"add": {"other": ["compute_stencil"]}}, f)
            stage6_noise_config.configure(path)
            report_configured = render(MPI_2RANK_DIR)
        self.assertNotIn("compute_stencil", report_configured)


class NoDataFoundTests(unittest.TestCase):
    def test_empty_run_dir_raises_system_exit(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(SystemExit) as ctx:
                render(empty_dir)
        self.assertIn("no rocprof-sys timemory text table found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
