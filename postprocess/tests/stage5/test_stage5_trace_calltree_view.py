import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

view = load_module_by_path("stage5_trace_calltree_view", "stage5", "stage5_trace_calltree_view.py")

import stage6_time_range_config as trc  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
NOISE_DIR = os.path.join(FIXTURES, "trace_calltree_noise")
RANK_INPUTS = [("r0", os.path.join(NOISE_DIR, "rank0.csv"))]
TIME_RANGE_DIR = os.path.join(FIXTURES, "trace_time_range")
TIME_RANGE_RANK_INPUTS = [("r0", os.path.join(TIME_RANGE_DIR, "rank0.csv"))]


class BuildCalltreeViewTests(unittest.TestCase):
    def test_default_hides_gpu_api_wrapper_noise_compiler_runtime_and_other(self):
        text = view.build_calltree_view(RANK_INPUTS)["tree_text"]
        self.assertIn("main", text)
        self.assertIn("jacobi_sweep", text)
        self.assertIn("MPI_Barrier", text)
        self.assertNotIn("gotcha_call", text)
        self.assertNotIn("hipLaunchKernel", text)
        self.assertNotIn("jacobi_kernel.kd", text)
        self.assertNotIn("posix_memalign", text)
        self.assertNotIn("numa_migration_event", text)

    def test_show_gpu_api_reveals_the_gpu_subtree(self):
        text = view.build_calltree_view(RANK_INPUTS, show_gpu_api=True)["tree_text"]
        self.assertIn("hipLaunchKernel", text)
        self.assertIn("jacobi_kernel.kd", text)

    def test_show_rocprofsys_internals_reveals_wrapper_noise(self):
        text = view.build_calltree_view(RANK_INPUTS, show_rocprofsys_internals=True)["tree_text"]
        self.assertIn("gotcha_call", text)

    def test_show_compiler_runtime_reveals_compiler_runtime_noise(self):
        text = view.build_calltree_view(RANK_INPUTS, show_compiler_runtime=True)["tree_text"]
        self.assertIn("posix_memalign", text)

    def test_other_is_folded_whenever_the_wrapper_noise_postprocess_step_runs(self):
        # "other" has no --show-* flag of its own -- it's always folded into its parent
        # (fold=True) as part of the same postprocess step that strips wrapper_noise, same as the
        # sample pipeline's own strip_wrapper_noise(). That step itself only runs when
        # show_rocprofsys_internals is False (matching stage5_calltree_view.py's identical
        # gating), so "other" only disappears in that case -- confirmed separately below that
        # show_rocprofsys_internals=True (which skips the whole postprocess step) also reveals it,
        # matching existing precedent rather than being its own bug.
        text = view.build_calltree_view(
            RANK_INPUTS, show_gpu_api=True, show_mpi_internals=True, show_compiler_runtime=True,
        )["tree_text"]
        self.assertNotIn("numa_migration_event", text)

    def test_show_rocprofsys_internals_also_reveals_other_since_it_skips_the_whole_postprocess_step(self):
        text = view.build_calltree_view(RANK_INPUTS, show_rocprofsys_internals=True)["tree_text"]
        self.assertIn("numa_migration_event", text)

    def test_mpi_territory_row_itself_always_shown_even_by_default(self):
        text = view.build_calltree_view(RANK_INPUTS)["tree_text"]
        self.assertIn("MPI_Barrier", text)

    def test_return_shape_has_no_fallback_text_key(self):
        result = view.build_calltree_view(RANK_INPUTS)
        self.assertEqual(set(result.keys()), {"rank_keys", "tree_text"})


class TimeRangePruningTests(unittest.TestCase):
    def tearDown(self):
        trc.configure(None)
        for f in glob.glob(os.path.join(TIME_RANGE_DIR, "*.agg.json")):
            os.remove(f)

    def test_no_active_range_leaves_zero_time_subtrees_untouched(self):
        # Not that there'd be any zero-time subtrees without a range active -- confirms the
        # pruning predicate simply isn't composed in at all when there's nothing to filter.
        trc.configure(None)
        text = view.build_calltree_view(TIME_RANGE_RANK_INPUTS)["tree_text"]
        self.assertIn("init_phase", text)
        self.assertIn("teardown_phase", text)
        self.assertIn("compute_phase", text)

    def test_entirely_out_of_range_subtrees_are_cut(self):
        trc.configure("30:70")
        text = view.build_calltree_view(TIME_RANGE_RANK_INPUTS)["tree_text"]
        self.assertNotIn("init_phase", text)
        self.assertNotIn("teardown_phase", text)
        self.assertIn("compute_phase", text)
        self.assertIn("main", text)

    def test_ancestor_chain_to_a_surviving_descendant_stays_intact(self):
        # hipLaunchKernel's own span never touches [30,70], but the kernel it dispatched
        # (reparented via corr_id, independently timed) does -- the launch call must stay visible
        # as the connecting ancestor, not be pruned just because its OWN contribution is zero.
        trc.configure("30:70")
        text = view.build_calltree_view(TIME_RANGE_RANK_INPUTS, show_gpu_api=True)["tree_text"]
        self.assertIn("hipLaunchKernel", text)
        self.assertIn("jacobi_kernel.kd", text)

    def test_pruning_composes_with_existing_tag_based_pruning(self):
        # gpu_api noise stays hidden by default, independent of the new range-based predicate --
        # the two compose via OR, neither one disabling the other.
        trc.configure("30:70")
        text = view.build_calltree_view(TIME_RANGE_RANK_INPUTS)["tree_text"]
        self.assertNotIn("hipLaunchKernel", text)


if __name__ == "__main__":
    unittest.main()
