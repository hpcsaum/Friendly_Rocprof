import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

s4c = load_module_by_path("stage4_rocprofsys_common", "stage4", "stage4_rocprofsys_common.py")


def make_label_parent_row(label, parent=None):
    """A minimal row for caller_chains_for_label() -- it only ever reads "label"/"parent"."""
    return {"label": label, "parent": parent}


class MergeRankTreesTests(unittest.TestCase):
    def test_merges_same_label_across_ranks_by_structural_position(self):
        # Two independent single-rank trees (distinct row objects, as if
        # parsed from two separate files) sharing the same "main" ->
        # "compute" structure must merge into ONE node per position, with
        # both ranks' contributions kept separately in per_rank.
        main_a = {"label": "main", "parent": None, "count": 1, "self_sum": 0.0, "sum": 10.0, "gpu": False}
        compute_a = {"label": "compute", "parent": main_a, "count": 5, "self_sum": 1.0, "sum": 1.0, "gpu": False}
        main_b = {"label": "main", "parent": None, "count": 1, "self_sum": 0.0, "sum": 20.0, "gpu": False}
        compute_b = {"label": "compute", "parent": main_b, "count": 7, "self_sum": 3.0, "sum": 3.0, "gpu": False}

        ranks = [
            ("rankA", [main_a, compute_a], [main_a]),
            ("rankB", [main_b, compute_b], [main_b]),
        ]
        merged_roots = s4c.merge_rank_trees(ranks)
        self.assertEqual(len(merged_roots), 1)
        merged_main = merged_roots[0]
        self.assertEqual(merged_main["label"], "main")
        self.assertEqual(set(merged_main["per_rank"].keys()), {"rankA", "rankB"})

        merged_compute = merged_main["children"]["compute"]
        self.assertEqual(merged_compute["per_rank"]["rankA"]["self_sum"], 1.0)
        self.assertEqual(merged_compute["per_rank"]["rankB"]["self_sum"], 3.0)
        self.assertIs(merged_compute["parent"], merged_main)

    def test_tags_unioned_across_ranks(self):
        row_a = {"label": "start_thread", "parent": None, "count": 1, "self_sum": 1.0, "sum": 1.0, "tags": set()}
        row_b = {"label": "start_thread", "parent": None, "count": 1, "self_sum": 1.0, "sum": 1.0, "tags": {"gpu_api"}}
        ranks = [("rankA", [row_a], [row_a]), ("rankB", [row_b], [row_b])]
        merged_roots = s4c.merge_rank_trees(ranks)
        self.assertEqual(merged_roots[0]["tags"], {"gpu_api"})

    def test_single_rank_collapses_repeated_same_position_calls(self):
        # A single rank's own repeated calls at the same tree position (e.g. many identical
        # hipLaunchKernel siblings under one shared parent object) collapse into one merged
        # node -- the same by-id(parent)+label mechanism this function already uses across
        # ranks, exercised here with len(ranks) == 1.
        main = {"label": "main", "parent": None, "count": 1, "self_sum": 0.0, "sum": 10.0}
        launch_1 = {"label": "hipLaunchKernel", "parent": main, "count": 1, "self_sum": 1.0, "sum": 1.0}
        launch_2 = {"label": "hipLaunchKernel", "parent": main, "count": 1, "self_sum": 2.0, "sum": 2.0}
        launch_3 = {"label": "hipLaunchKernel", "parent": main, "count": 1, "self_sum": 3.0, "sum": 3.0}
        rows = [main, launch_1, launch_2, launch_3]

        merged_roots = s4c.merge_rank_trees([("r0", rows, [main])])
        self.assertEqual(len(merged_roots), 1)
        merged_main = merged_roots[0]
        self.assertEqual(len(merged_main["children"]), 1)
        merged_launch = merged_main["children"]["hipLaunchKernel"]
        entry = merged_launch["per_rank"]["r0"]
        self.assertEqual(entry["count"], 3)
        self.assertAlmostEqual(entry["self_sum"], 6.0)
        self.assertAlmostEqual(entry["sum"], 6.0)

    def test_label_key_reads_display_name_from_a_different_column(self):
        # Trace-pipeline rows use "name", not "label" (see stage1_rocprofsys_trace.LABEL_KEY) --
        # merge_rank_trees() must read whichever key label_key names, not assume "label".
        main = {"name": "main", "parent": None, "count": 1, "self_sum": 0.0, "sum": 1.0}
        child = {"name": "jacobi_sweep", "parent": main, "count": 1, "self_sum": 1.0, "sum": 1.0}
        rows = [main, child]

        merged_roots = s4c.merge_rank_trees([("r0", rows, [main])], label_key="name")
        self.assertEqual(merged_roots[0]["label"], "main")
        self.assertEqual(merged_roots[0]["children"]["jacobi_sweep"]["label"], "jacobi_sweep")


class AggregateNodeStatsTests(unittest.TestCase):
    def test_missing_rank_counts_as_zero_not_omitted(self):
        per_rank = {"r0": {"count": 10, "self_sum": 2.0, "sum": 2.0}}
        stats = s4c.aggregate_node_stats(per_rank, ["r0", "r1", "r2"])
        self.assertAlmostEqual(stats["self_avg"], 2.0 / 3)
        self.assertAlmostEqual(stats["self_min"], 0.0)
        self.assertAlmostEqual(stats["self_max"], 2.0)

    def test_std_dev_and_min_max(self):
        per_rank = {
            "r0": {"count": 1, "self_sum": 1.0, "sum": 1.0},
            "r1": {"count": 1, "self_sum": 3.0, "sum": 3.0},
            "r2": {"count": 1, "self_sum": 5.0, "sum": 5.0},
        }
        stats = s4c.aggregate_node_stats(per_rank, ["r0", "r1", "r2"])
        self.assertAlmostEqual(stats["self_avg"], 3.0)
        self.assertAlmostEqual(stats["self_std"], 1.632993161855452)
        self.assertAlmostEqual(stats["self_min"], 1.0)
        self.assertAlmostEqual(stats["self_max"], 5.0)


class CallerChainsForLabelTests(unittest.TestCase):
    def test_single_call_site_returns_one_root_to_target_chain(self):
        main = make_label_parent_row("main")
        compute = make_label_parent_row("compute", parent=main)
        target = make_label_parent_row("hot_function", parent=compute)
        rows = [main, compute, target]

        chains = s4c.caller_chains_for_label(rows, "hot_function")
        self.assertEqual(len(chains), 1)
        self.assertEqual([n["label"] for n in chains[0]], ["main", "compute", "hot_function"])

    def test_two_distinct_call_sites_return_two_chains(self):
        main = make_label_parent_row("main")
        compute_a = make_label_parent_row("compute_a", parent=main)
        compute_b = make_label_parent_row("compute_b", parent=main)
        target_a = make_label_parent_row("hot_function", parent=compute_a)
        target_b = make_label_parent_row("hot_function", parent=compute_b)
        rows = [main, compute_a, compute_b, target_a, target_b]

        chains = s4c.caller_chains_for_label(rows, "hot_function")
        self.assertEqual(len(chains), 2)
        labels = {tuple(n["label"] for n in chain) for chain in chains}
        self.assertEqual(labels, {
            ("main", "compute_a", "hot_function"),
            ("main", "compute_b", "hot_function"),
        })

    def test_no_match_returns_empty_list(self):
        main = make_label_parent_row("main")
        rows = [main]
        self.assertEqual(s4c.caller_chains_for_label(rows, "never_called"), [])

    def test_target_is_a_root_returns_single_node_chain(self):
        main = make_label_parent_row("main")
        rows = [main]
        chains = s4c.caller_chains_for_label(rows, "main")
        self.assertEqual(len(chains), 1)
        self.assertEqual([n["label"] for n in chains[0]], ["main"])
        self.assertIsNone(chains[0][0]["parent"])


class KernelOwnerLabelTests(unittest.TestCase):
    def test_strips_ck_suffix(self):
        self.assertEqual(
            s4c.kernel_owner_label("jacobi_sweep$pressure_solver_mod_$ck_L36_1_cce$noloop$form"),
            "jacobi_sweep$pressure_solver_mod_",
        )

    def test_unchanged_without_ck_marker(self):
        self.assertEqual(s4c.kernel_owner_label("JacobiIterationKernel"), "JacobiIterationKernel")

    # The following mirror the real rocprofv3 kernel_stats.csv "Name" values observed in
    # test_apps/results/ for each language/compiler combo (Cray Fortran, covered above, is the
    # only one NOT using this generic shape).
    def test_amd_c_omp_offloading_plain_name(self):
        self.assertEqual(
            s4c.kernel_owner_label("__omp_offloading_4f_8fb8827_launch_omp_kernel_l6"),
            "launch_omp_kernel",
        )

    def test_cray_c_omp_offloading_plain_name_with_compiler_suffix(self):
        self.assertEqual(
            s4c.kernel_owner_label("__omp_offloading_34_8fb8827_launch_omp_kernel_l6_cce$noloop$form"),
            "launch_omp_kernel",
        )

    def test_amd_cpp_omp_offloading_itanium_mangled_name(self):
        self.assertEqual(
            s4c.kernel_owner_label("__omp_offloading_4f_8fb8861__Z17launch_omp_kernelPdi_l6"),
            "launch_omp_kernel",
        )

    def test_cray_cpp_omp_offloading_itanium_mangled_name_with_compiler_suffix(self):
        self.assertEqual(
            s4c.kernel_owner_label(
                "__omp_offloading_34_8fb8861__Z17launch_omp_kernelPdi_l6_cce$noloop$form",
            ),
            "launch_omp_kernel",
        )

    def test_amd_fortran_omp_offloading_flang_mangled_name(self):
        self.assertEqual(
            s4c.kernel_owner_label(
                "__omp_offloading_4f_8fb7ea6__QMkernel_omp_modPlaunch_omp_kernel_l12",
            ),
            "launch_omp_kernel",
        )

    def test_hip_kernel_name_matches_neither_convention(self):
        # test_apps' native HIP kernel ("stencil_kernel") isn't affected by the OMPT-shared-entry-
        # point bug this decoding exists for -- its corr_id ancestry is already call-site-exact --
        # so it must return unchanged, same as any other unrecognized name.
        self.assertEqual(s4c.kernel_owner_label("stencil_kernel"), "stencil_kernel")


class ExpandLabelsWithAncestorsTests(unittest.TestCase):
    def test_depth_1_adds_only_the_immediate_parent(self):
        main = make_label_parent_row("main")
        caller = make_label_parent_row("caller", parent=main)
        target = make_label_parent_row("target", parent=caller)
        rows = [main, caller, target]

        added = s4c.expand_labels_with_ancestors(rows, {"target"}, depth=1)
        self.assertEqual(added, {"caller"})

    def test_depth_2_reaches_the_grandparent_too(self):
        main = make_label_parent_row("main")
        caller = make_label_parent_row("caller", parent=main)
        target = make_label_parent_row("target", parent=caller)
        rows = [main, caller, target]

        added = s4c.expand_labels_with_ancestors(rows, {"target"}, depth=2)
        self.assertEqual(added, {"caller", "main"})

    def test_depth_0_adds_nothing(self):
        main = make_label_parent_row("main")
        target = make_label_parent_row("target", parent=main)
        added = s4c.expand_labels_with_ancestors([main, target], {"target"}, depth=0)
        self.assertEqual(added, set())

    def test_already_selected_ancestor_is_not_reported_twice(self):
        main = make_label_parent_row("main")
        caller = make_label_parent_row("caller", parent=main)
        target = make_label_parent_row("target", parent=caller)
        rows = [main, caller, target]

        # "caller" is already in the base selection -- expanding "target" must not re-report it,
        # even though expansion also runs for "caller" itself and legitimately adds "main".
        added = s4c.expand_labels_with_ancestors(rows, {"target", "caller"}, depth=1)
        self.assertNotIn("caller", added)
        self.assertEqual(added, {"main"})

    def test_multiple_call_sites_each_contribute_their_own_immediate_parent(self):
        main = make_label_parent_row("main")
        caller_a = make_label_parent_row("caller_a", parent=main)
        caller_b = make_label_parent_row("caller_b", parent=main)
        target_a = make_label_parent_row("target", parent=caller_a)
        target_b = make_label_parent_row("target", parent=caller_b)
        rows = [main, caller_a, caller_b, target_a, target_b]

        added = s4c.expand_labels_with_ancestors(rows, {"target"}, depth=1)
        self.assertEqual(added, {"caller_a", "caller_b"})

    def test_label_never_sampled_contributes_nothing(self):
        main = make_label_parent_row("main")
        added = s4c.expand_labels_with_ancestors([main], {"never_seen"}, depth=1)
        self.assertEqual(added, set())


class ResolveKernelOwnersTests(unittest.TestCase):
    def test_decoded_owners_are_returned(self):
        owners = s4c.resolve_kernel_owners([
            "jacobi_sweep$pressure_solver_mod_$ck_L36_1_cce$noloop$form",
            "__omp_offloading_4f_8fb8827_launch_omp_kernel_l6",
        ])
        self.assertEqual(owners, {"jacobi_sweep$pressure_solver_mod_", "launch_omp_kernel"})

    def test_unrecognized_kernel_names_contribute_nothing(self):
        owners = s4c.resolve_kernel_owners(["stencil_kernel", "JacobiIterationKernel"])
        self.assertEqual(owners, set())

    def test_duplicate_owners_collapse(self):
        owners = s4c.resolve_kernel_owners([
            "foo$mod_$ck_L1_1",
            "foo$mod_$ck_L2_2",
        ])
        self.assertEqual(owners, {"foo$mod_"})


def make_merge_row(label, parent=None, self_sum=0.0):
    return {"label": label, "parent": parent, "count": 1, "self_sum": self_sum, "sum": self_sum}


class MakeZeroTimePrunedTests(unittest.TestCase):
    def test_entirely_zero_leaf_subtree_is_pruned(self):
        main = make_merge_row("main", self_sum=2.0)
        other_func = make_merge_row("other_func", parent=main, self_sum=0.0)
        rows = [main, other_func]
        merged_roots = s4c.merge_rank_trees([("r0", rows, [main])])
        by_label = {n["label"]: n for n in merged_roots[0]["children"].values()}
        by_label["main"] = merged_roots[0]

        is_pruned = s4c.make_zero_time_pruned(merged_roots)
        self.assertTrue(is_pruned(by_label["other_func"]))
        self.assertFalse(is_pruned(by_label["main"]))

    def test_zero_launch_call_with_a_nonzero_descendant_is_not_pruned(self):
        # The corr_id/owner-reanchored-kernel case: a launch call's own contribution is zero, but
        # a real, independently-timed descendant (a concurrently-executing GPU kernel) is not --
        # the whole chain down to it must stay visible.
        main = make_merge_row("main", self_sum=1.0)
        launch_call = make_merge_row("hipLaunchKernel", parent=main, self_sum=0.0)
        kernel = make_merge_row("jacobi_kernel.kd", parent=launch_call, self_sum=5.0)
        rows = [main, launch_call, kernel]
        merged_roots = s4c.merge_rank_trees([("r0", rows, [main])])
        merged_main = merged_roots[0]
        merged_launch = merged_main["children"]["hipLaunchKernel"]
        merged_kernel = merged_launch["children"]["jacobi_kernel.kd"]

        is_pruned = s4c.make_zero_time_pruned(merged_roots)
        self.assertFalse(is_pruned(merged_main))
        self.assertFalse(is_pruned(merged_launch))
        self.assertFalse(is_pruned(merged_kernel))

    def test_sibling_after_a_zero_subtree_is_still_visited_and_correctly_judged(self):
        # Regression: an early implementation used all(visit(c) for c in children) directly in a
        # generator, which short-circuits on the first False and skips visit() on later siblings
        # entirely -- silently leaving them out of the pruned-id set (so they'd wrongly render as
        # "not pruned" even when genuinely fully zero) instead of ever really judging them.
        main = make_merge_row("main", self_sum=1.0)
        zero_first = make_merge_row("zero_first", parent=main, self_sum=0.0)
        zero_second = make_merge_row("zero_second", parent=main, self_sum=0.0)
        rows = [main, zero_first, zero_second]
        merged_roots = s4c.merge_rank_trees([("r0", rows, [main])])
        merged_main = merged_roots[0]

        is_pruned = s4c.make_zero_time_pruned(merged_roots)
        self.assertTrue(is_pruned(merged_main["children"]["zero_first"]))
        self.assertTrue(is_pruned(merged_main["children"]["zero_second"]))

    def test_entire_tree_zero_prunes_every_node_including_the_root(self):
        main = make_merge_row("main", self_sum=0.0)
        leaf = make_merge_row("leaf", parent=main, self_sum=0.0)
        rows = [main, leaf]
        merged_roots = s4c.merge_rank_trees([("r0", rows, [main])])

        is_pruned = s4c.make_zero_time_pruned(merged_roots)
        self.assertTrue(is_pruned(merged_roots[0]))
        self.assertTrue(is_pruned(merged_roots[0]["children"]["leaf"]))

    def test_no_zero_time_anywhere_prunes_nothing(self):
        main = make_merge_row("main", self_sum=1.0)
        child = make_merge_row("child", parent=main, self_sum=2.0)
        rows = [main, child]
        merged_roots = s4c.merge_rank_trees([("r0", rows, [main])])

        is_pruned = s4c.make_zero_time_pruned(merged_roots)
        self.assertFalse(is_pruned(merged_roots[0]))
        self.assertFalse(is_pruned(merged_roots[0]["children"]["child"]))


if __name__ == "__main__":
    unittest.main()
