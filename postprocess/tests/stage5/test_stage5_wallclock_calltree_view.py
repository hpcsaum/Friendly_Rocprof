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
from stage5_wallclock_calltree_view import build_calltree_view  # noqa: E402
from stage4_rocprofsys_sample_tree import load_rank_trees  # noqa: E402
import stage6_noise_config  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
MPI_2RANK_DIR = os.path.join(FIXTURES, "mpi_2rank")
GPU_SPAWNED_THREAD_DIR = os.path.join(FIXTURES, "gpu_spawned_thread")
MULTI_METRIC_RANK_DIR = os.path.join(FIXTURES, "multi_metric_rank")
GPU_API_NESTED_CHAIN_DIR = os.path.join(FIXTURES, "gpu_api_nested_chain")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_anchor")
KERNEL_MULTI_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_multi_anchor")
KERNEL_NO_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_no_anchor")
KD_ARTIFACT_DIR = os.path.join(FIXTURES, "calltree_kd_artifact")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")


def render(run_dir, **kwargs):
    cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
    view = build_calltree_view(run_dir, cpu_dir, gpu_dir, **kwargs)
    return view["tree_text"] + view["fallback_text"]


_NUMERIC_SUFFIX_RE = re.compile(r"(?:\s{2,}-?[\d.]+)+\s*$")


def _labels_only(report):
    """Strips each physical line's trailing numeric-cell columns (if any), then joins every line
    back together with no separator -- reconstructs each row's own label text contiguously, so a
    test can search for a label substring regardless of exactly where
    stage5_tree_render.wrap_leading_labels() cut a too-long label across physical lines."""
    return "".join(_NUMERIC_SUFFIX_RE.sub("", line) for line in report.splitlines())


class LoadRankTreesTests(unittest.TestCase):
    def test_basic_two_rank_tree(self):
        cpu_dir, _gpu_dir = resolve_run_dirs(MPI_2RANK_DIR)
        ranks = load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        self.assertEqual(len(ranks), 2)
        for rank_key, rows, roots in ranks:
            self.assertEqual(len(roots), 1)
            self.assertEqual(roots[0]["label"], "main")
            # compute_stencil and hipMemcpy are main's only two children
            children = [r for r in rows if r["parent"] is roots[0]]
            self.assertEqual({c["label"] for c in children}, {"compute_stencil", "hipMemcpy"})

    def test_raises_on_empty_input_is_just_empty_list(self):
        # load_rank_trees() itself doesn't raise -- build_calltree_view() does, once it
        # sees an empty list. Confirmed here so that distinction stays intentional.
        cpu_dir, _gpu_dir = resolve_run_dirs(EMPTY_DIR)
        self.assertEqual(load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt"), [])


class MultiRootDetectionTests(unittest.TestCase):
    def test_is_thread_root_flagged_case(self):
        # gpu_spawned_thread: start_thread's DEPTH nests one level under its
        # spawning pthread_create call -- is_thread_root correctly fires, but
        # root-enumeration here relies only on parent is None, not that flag.
        cpu_dir, _gpu_dir = resolve_run_dirs(GPU_SPAWNED_THREAD_DIR)
        ranks = load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        self.assertEqual(len(ranks), 1)
        _rank_key, _rows, roots = ranks[0]
        # main, hipRuntimeGetVersion, compute_stencil are three independent
        # DEPTH-0 roots in this fixture -- all three must be detected.
        self.assertEqual({r["label"] for r in roots}, {"main", "hipRuntimeGetVersion", "compute_stencil"})

    def test_depth_resets_to_zero_case(self):
        # multi_metric_rank: a second OS thread's own root row sits at DEPTH 0,
        # the same depth as thread 0's other roots -- parent is None still
        # correctly separates it without relying on is_thread_root.
        cpu_dir, _gpu_dir = resolve_run_dirs(MULTI_METRIC_RANK_DIR)
        ranks = load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        _rank_key, _rows, roots = ranks[0]
        labels = [r["label"] for r in roots]
        self.assertEqual(labels.count("worker_loop"), 2)  # two distinct thread roots, not merged


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
        report = render(KERNEL_MULTI_ANCHOR_DIR)
        # fixture: compute_a issued 300 launch calls, compute_b issued 200 (of 500 total). This
        # bracketed label is long enough that wrap_leading_labels() may hard-wrap it across
        # physical lines -- reconstruct labels-only text first so the substring search sees it
        # whole regardless of the exact cut point.
        flat = _labels_only(report)
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
        # compute_stencil is main's real, otherwise-untagged child (see
        # LoadRankTreesTests.test_basic_two_rank_tree above).
        report_default = render(MPI_2RANK_DIR)
        self.assertIn("compute_stencil", report_default)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "noise_config.json")
            with open(path, "w") as f:
                json.dump({"add": {"other": ["compute_stencil"]}}, f)
            stage6_noise_config.configure(path)
            report_configured = render(MPI_2RANK_DIR)
        self.assertNotIn("compute_stencil", report_configured)


if __name__ == "__main__":
    unittest.main()
