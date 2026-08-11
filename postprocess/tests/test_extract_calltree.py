import importlib.util
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "extract_calltree.py")

# extract_calltree.py does a plain top-level "import extract_CPU_hotspots"/
# "import extract_GPU_hotspots", relying on its own directory being on sys.path --
# true automatically when run directly, but not when loaded here by explicit
# file path, so replicate that manually (same as test_extract_hotspots.py).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("extract_calltree", MODULE_PATH)
ct_tool = importlib.util.module_from_spec(spec)
sys.modules["extract_calltree"] = ct_tool
spec.loader.exec_module(ct_tool)

MPI_2RANK_DIR = os.path.join(FIXTURES, "mpi_2rank")
GPU_SPAWNED_THREAD_DIR = os.path.join(FIXTURES, "gpu_spawned_thread")
MULTI_METRIC_RANK_DIR = os.path.join(FIXTURES, "multi_metric_rank")
GPU_API_NESTED_CHAIN_DIR = os.path.join(FIXTURES, "gpu_api_nested_chain")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_anchor")
KERNEL_MULTI_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_multi_anchor")
KERNEL_NO_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_no_anchor")
KD_ARTIFACT_DIR = os.path.join(FIXTURES, "calltree_kd_artifact")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")


class ResolveRunDirsTests(unittest.TestCase):
    def test_detects_paired_subdirs(self):
        cpu_dir, gpu_dir = ct_tool.resolve_run_dirs(KERNEL_ANCHOR_DIR)
        self.assertEqual(cpu_dir, os.path.join(KERNEL_ANCHOR_DIR, "rocprof-sys"))
        self.assertEqual(gpu_dir, os.path.join(KERNEL_ANCHOR_DIR, "rocprofv3"))

    def test_falls_back_to_run_dir_itself_when_flat(self):
        cpu_dir, gpu_dir = ct_tool.resolve_run_dirs(MPI_2RANK_DIR)
        self.assertEqual(cpu_dir, MPI_2RANK_DIR)
        self.assertIsNone(gpu_dir)


class LoadRankTreesTests(unittest.TestCase):
    def test_basic_two_rank_tree(self):
        ranks = ct_tool.load_rank_trees(MPI_2RANK_DIR)
        self.assertEqual(len(ranks), 2)
        for rank_key, rows, roots in ranks:
            self.assertEqual(len(roots), 1)
            self.assertEqual(roots[0]["label"], "main")
            # compute_stencil and hipMemcpy are main's only two children
            children = [r for r in rows if r["parent"] is roots[0]]
            self.assertEqual({c["label"] for c in children}, {"compute_stencil", "hipMemcpy"})

    def test_raises_on_empty_input_is_just_empty_list(self):
        # load_rank_trees() itself doesn't raise -- write_report() does, once it
        # sees an empty list. Confirmed here so that distinction stays intentional.
        self.assertEqual(ct_tool.load_rank_trees(EMPTY_DIR), [])


class MultiRootDetectionTests(unittest.TestCase):
    def test_is_thread_root_flagged_case(self):
        # gpu_spawned_thread: start_thread's DEPTH nests one level under its
        # spawning pthread_create call -- is_thread_root correctly fires, but
        # root-enumeration here relies only on parent is None, not that flag.
        ranks = ct_tool.load_rank_trees(GPU_SPAWNED_THREAD_DIR)
        self.assertEqual(len(ranks), 1)
        _rank_key, _rows, roots = ranks[0]
        # main, hipRuntimeGetVersion, compute_stencil are three independent
        # DEPTH-0 roots in this fixture -- all three must be detected.
        self.assertEqual({r["label"] for r in roots}, {"main", "hipRuntimeGetVersion", "compute_stencil"})

    def test_depth_resets_to_zero_case(self):
        # multi_metric_rank: a second OS thread's own root row sits at DEPTH 0,
        # the same depth as thread 0's other roots -- parent is None still
        # correctly separates it without relying on is_thread_root.
        ranks = ct_tool.load_rank_trees(MULTI_METRIC_RANK_DIR)
        _rank_key, _rows, roots = ranks[0]
        labels = [r["label"] for r in roots]
        self.assertEqual(labels.count("worker_loop"), 2)  # two distinct thread roots, not merged


class RenderTreeTests(unittest.TestCase):
    def _render(self, run_dir, max_depth=None, show_gpu_api=False):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            return ct_tool.write_report(run_dir, dest, max_depth=max_depth, show_gpu_api=show_gpu_api)

    def test_gpu_api_hidden_by_default(self):
        report = self._render(MPI_2RANK_DIR)
        self.assertNotIn("hipMemcpy", report)

    def test_gpu_api_shown_with_flag(self):
        report = self._render(MPI_2RANK_DIR, show_gpu_api=True)
        self.assertIn("hipMemcpy", report)

    def test_gpu_api_chain_pruned_wholesale(self):
        # gpu_api_nested_chain: hipStreamCreate -> hip::hipStreamCreate(...) ->
        # hip::ihipStreamCreate(...) -- none should appear by default, all three
        # should appear when --show-gpu-api is passed.
        default_report = self._render(GPU_API_NESTED_CHAIN_DIR)
        for label in ("hipStreamCreate", "hip::hipStreamCreate", "hip::ihipStreamCreate"):
            self.assertNotIn(label, default_report)
        shown_report = self._render(GPU_API_NESTED_CHAIN_DIR, show_gpu_api=True)
        for label in ("hipStreamCreate", "hip::hipStreamCreate", "hip::ihipStreamCreate"):
            self.assertIn(label, shown_report)

    def test_max_depth_truncates_with_stated_count(self):
        report = self._render(MPI_2RANK_DIR, max_depth=0)
        self.assertIn("hidden below this point", report)
        self.assertIn("1 more node(s)", report)  # compute_stencil only -- hipMemcpy is gpu-hidden regardless
        self.assertNotIn("compute_stencil", report)

    def test_max_depth_hidden_count_matches_show_gpu_api(self):
        report = self._render(MPI_2RANK_DIR, max_depth=0, show_gpu_api=True)
        self.assertIn("2 more node(s)", report)

    def test_no_max_depth_prints_whole_tree(self):
        report = self._render(MPI_2RANK_DIR)
        self.assertIn("compute_stencil", report)
        self.assertNotIn("hidden below this point", report)

    def test_real_columns_not_bracketed_string(self):
        report = self._render(MPI_2RANK_DIR)
        self.assertIn("CALLS", report)
        self.assertIn("SELF(s)", report)
        self.assertIn("TOTAL(s)", report)
        self.assertNotIn("[calls=", report)  # old per-line bracketed format, must be gone

    def test_tree_connectors_present(self):
        # mpi_2rank's main has only one child, so it's rendered "└── " (last
        # child) -- "├── " needs a node with 2+ children, like gpu_api_nested_chain's
        # main (compute_stencil + hipStreamCreate).
        report = self._render(GPU_API_NESTED_CHAIN_DIR, show_gpu_api=True)
        self.assertIn("├── ", report)
        self.assertIn("└── ", report)


class KdArtifactFilteringTests(unittest.TestCase):
    def test_is_kernel_descriptor_artifact(self):
        self.assertTrue(ct_tool.is_kernel_descriptor_artifact("some_kernel_name.kd"))
        self.assertFalse(ct_tool.is_kernel_descriptor_artifact("some_kernel_name"))
        self.assertFalse(ct_tool.is_kernel_descriptor_artifact("hipLaunchKernel"))

    def test_kd_suffixed_row_hidden_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            report = ct_tool.write_report(KD_ARTIFACT_DIR, dest)
        self.assertIn("compute_stencil", report)
        self.assertNotIn("some_kernel_name.kd", report)

    def test_kd_suffixed_row_shown_with_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            report = ct_tool.write_report(KD_ARTIFACT_DIR, dest, show_gpu_api=True)
        self.assertIn("some_kernel_name.kd", report)


class KernelIntegrationTests(unittest.TestCase):
    def test_single_anchor_gets_full_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            report = ct_tool.write_report(KERNEL_ANCHOR_DIR, dest)
        self.assertIn("[GPU kernels -- rocprofv3]", report)
        self.assertIn("JacobiIterationKernel", report)
        i_parent = report.index("compute_stencil")
        i_kernel = report.index("[GPU kernels -- rocprofv3]")
        i_leaf = report.index("JacobiIterationKernel")
        self.assertTrue(i_parent < i_kernel < i_leaf)  # nested under compute_stencil, not top-level
        self.assertNotIn("no launch call site found", report)

    def test_multiple_anchors_split_proportionally(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            report = ct_tool.write_report(KERNEL_MULTI_ANCHOR_DIR, dest)
        # fixture: compute_a issued 300 launch calls, compute_b issued 200 (of 500 total)
        self.assertIn("~60% estimate: this site issued 300/500", report)
        self.assertIn("~40% estimate: this site issued 200/500", report)
        # both anchors get their own nested kernel breakdown, not one shared full total
        self.assertEqual(report.count("JacobiIterationKernel"), 2)

    def test_no_anchor_falls_back_to_top_level_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            report = ct_tool.write_report(KERNEL_NO_ANCHOR_DIR, dest)
        self.assertIn("=== GPU kernels (rocprofv3) -- no launch call site found in CPU tree ===", report)
        self.assertIn("JacobiIterationKernel", report)
        # the fallback section's kernel data must NOT also appear nested inside a rank tree
        i_fallback = report.index("=== GPU kernels")
        self.assertNotIn("[GPU kernels -- rocprofv3", report[:i_fallback])


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
            ct_tool.main([MPI_2RANK_DIR, "-o", dest, "--max-depth", "1"])
            self.assertTrue(os.path.isfile(dest))


if __name__ == "__main__":
    unittest.main()
